"""Offline root SEARCH-value audit on the pre-locked val_tune QIDs.

    .venv/bin/python -m scripts.audit_rl_search_value

This is a privileged, gold-side diagnostic, never a policy or training input. It reads
only val_strict artifacts and only val_tune label and reader-cache rows. The frozen
protocol and its input fingerprints must match before any oracle scoring begins.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from src.data import artifacts
from src.data.schemas import ScenarioRecord
from src.decision.teacher import OracleConfig, oracle_episode
from src.eval import splitting
from src.pipeline import build_retrieval, cache_reader, train_rl, worlds
from src.qa.reader import ReaderCache, ReaderPrediction
from src.qa.state_cache import load_state_cache_meta
from src.rl.env import AdaptiveQAEnv, EpisodeWorld
from src.rl.reward import RewardCalculator, RewardConfig
from src.utils.config import load_config, resolve
from src.utils.hashing import sha256_file, sha256_json
from src.utils.io import read_yaml

SPLIT = "val_strict"
PROTOCOL = Path("outputs/experiments/2026-10-10-rl-audit/reports/val_protocol_locked.json")
RESULT = PROTOCOL.with_name("val_tune_search_value.json")
PROTOCOL_SHA256 = "b2c9eb40c427292b12865920a25e84c69a5d421d65904b4525939956106dabb7"
TOLERANCE = 1e-8


def verify_protocol(cfg: dict, protocol: dict) -> tuple[list[str], dict[str, str]]:
    if sha256_file(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("locked validation protocol changed")
    if protocol.get("criterion") != (
        "SEARCH helps iff max(Q_SEARCH_BM25,Q_SEARCH_DENSE) - "
        "max(Q_ANSWER_if_legal,Q_ABSTAIN) > 1e-8; tie does not count"
    ):
        raise ValueError("locked SEARCH criterion changed")
    if protocol.get("root") != (
        "after forced BM25 bootstrap; max_optional_search=2; "
        "gold visible only to offline oracle"
    ):
        raise ValueError("locked root definition changed")
    if protocol.get("report_labels_access") is not False:
        raise ValueError("protocol does not reserve report labels")
    if protocol.get("test_labels_access") is not False:
        raise ValueError("protocol does not prohibit test labels")
    config = worlds.env_config(cfg)
    if config.bootstrap_tool != "bm25" or config.max_optional_search != 2:
        raise ValueError("environment differs from locked root")

    processed = resolve(cfg, "data_processed")
    cache_path, meta_path = cache_reader.state_cache_paths(cfg, SPLIT)
    paths = {
        "partition_code": Path(train_rl.__file__),
        "questions_val": artifacts.questions_path(processed, SPLIT),
        "scenarios_val": artifacts.scenarios_path(resolve(cfg, "data_scenarios"), SPLIT),
        "rankings_val": build_retrieval.rankings_path(cfg, SPLIT),
        "reader_cache_val": cache_path,
        "reader_cache_meta_val": meta_path,
    }
    for name, path in paths.items():
        if sha256_file(path) != protocol["input_hashes"][name]:
            raise ValueError(f"frozen {name} hash changed: {path}")
    if RewardConfig.from_dict(cfg).spec_hash() != protocol["input_hashes"]["reward_config"]:
        raise ValueError("frozen reward changed")

    tune = protocol["tune_qids"]
    report = protocol["report_qids"]
    validation = protocol["validation"]
    if (len(tune) != len(set(tune)) or len(report) != len(set(report))
            or set(tune) & set(report)):
        raise ValueError("duplicate or overlapping validation QIDs")
    for name, qids in (("tune", tune), ("report", report)):
        if len(qids) != validation[f"n_{name}"] or sha256_json(qids) != validation[f"{name}_qid_sha256"]:
            raise ValueError(f"frozen {name} partition changed")

    # Public grouping metadata only: do not load the report partition's gold columns.
    group_rows = pq.read_table(str(paths["questions_val"]), columns=["qid", "title_group"]).to_pylist()
    groups = {row["qid"]: row["title_group"] for row in group_rows}
    if len(groups) != len(group_rows) or set(groups) != set(tune) | set(report):
        raise ValueError("validation QIDs are incomplete or duplicated")
    if sha256_json(groups) != protocol["title_groups_digest"]:
        raise ValueError("frozen title-group mapping changed")
    expected_tune, expected_report = splitting.group_holdout(
        groups, fraction=validation["report_fraction"], seed=validation["group_seed"])
    if tune != expected_tune or report != expected_report:
        raise ValueError("validation grouping no longer reproduces the locked partition")
    tune_groups = {groups[qid] for qid in tune}
    report_groups = {groups[qid] for qid in report}
    if tune_groups & report_groups or len(tune_groups) != validation["n_tune_groups"] or len(report_groups) != validation["n_report_groups"]:
        raise ValueError("validation title groups overlap or changed")
    return tune, groups


def tune_worlds(cfg: dict, tune: list[str]) -> list[EpisodeWorld]:
    processed = resolve(cfg, "data_processed")
    wanted = set(tune)
    labels = artifacts.questions_by_qid(pq.read_table(
        str(artifacts.questions_path(processed, SPLIT)),
        columns=["qid", "question", "is_impossible", "gold_doc_id", "answer_texts_display"],
        filters=[("qid", "in", tune)],
    ).to_pylist())
    if set(labels) != wanted:
        raise ValueError("val_tune label rows missing or unexpected")
    docs = artifacts.read_docs(processed, SPLIT)
    rankings = worlds._load_rankings(cfg, SPLIT)
    state_path, meta_path = cache_reader.state_cache_paths(cfg, SPLIT)
    meta = load_state_cache_meta(meta_path)
    config = worlds.env_config(cfg)
    worlds._check_state_cache_budget(meta, config)
    worlds._check_state_cache_sources(cfg, SPLIT, meta)
    worlds._check_state_cache_reader(cfg, meta)

    # Load only the frozen tune QIDs' reader predictions, not the reserved group's rows.
    cache_rows = pq.read_table(str(state_path), filters=[("qid", "in", tune)]).to_pylist()
    entries = {}
    for row in cache_rows:
        if row["schema_version"] != "v1" or row["qid"] not in wanted or row["cache_key"] in entries:
            raise ValueError("reader cache schema, QID, or key is invalid")
        entries[row["cache_key"]] = ReaderPrediction(
            qid=row["qid"], doc_id=row["doc_id"], answer_text=row["answer_text"],
            start_char=row["start_char"], end_char=row["end_char"],
            non_null_score=float(row["non_null_score"]), null_score=float(row["null_score"]),
            null_margin=float(row["null_margin"]), reader_model_hash=row["reader_model_hash"],
            provenance_ok=bool(row["provenance_ok"]))
    if not entries:
        raise ValueError("no val_tune reader cache entries")
    reader = ReaderCache(entries)

    selected = []
    with artifacts.scenarios_path(resolve(cfg, "data_scenarios"), SPLIT).open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["qid"] not in wanted:
                continue
            if row["split"] != SPLIT:
                raise ValueError("scenario split mismatch")
            selected.append(ScenarioRecord(
                scenario_id=row["scenario_id"], qid=row["qid"], split=row["split"],
                corpus_doc_ids=tuple(row["corpus_doc_ids"]),
                scenario_seed=int(row["scenario_seed"]), gold_doc_id=row.get("gold_doc_id"),
                reachable_bm25=row.get("reachable_bm25"),
                reachable_dense=row.get("reachable_dense"),
                difficulty=row.get("difficulty") or {}))
    if len(selected) != len(tune) or {s.qid for s in selected} != wanted:
        raise ValueError("val_tune scenarios missing or duplicated")

    result = []
    for scenario in sorted(selected, key=lambda s: s.qid):
        gold = labels[scenario.qid]
        if not gold["is_impossible"] and gold["gold_doc_id"] != scenario.gold_doc_id:
            raise ValueError(f"{scenario.qid}: answerable scenario/label gold document mismatch")
        row = rankings.get(scenario.scenario_id)
        if row is None or row["qid"] != scenario.qid:
            raise ValueError(f"{scenario.qid}: missing or mismatched ranking")
        ranking = {tool: list(row["rankings"][tool]) for tool in worlds.TOOLS}
        if any(doc_id not in docs for doc_id in scenario.corpus_doc_ids):
            raise ValueError(f"{scenario.qid}: corpus document missing from val_strict")
        corpus = {doc_id: docs[doc_id] for doc_id in scenario.corpus_doc_ids}
        world = EpisodeWorld(
            qid=scenario.qid, question=gold["question"], split=SPLIT,
            scenario_id=scenario.scenario_id, corpus=corpus, rankings=ranking,
            private=worlds._private(scenario, gold, corpus, ranking, config),
            reader=worlds._reader(reader, meta, scenario.qid))
        scores = row.get("scores") or {}
        world._scores_bm25 = dict(scores.get("bm25", {}))
        world._scores_dense = dict(scores.get("dense", {}))
        result.append(world)
    return result


def measure(cfg: dict, tune: list[str], groups: dict[str, str], items: list[EpisodeWorld]) -> dict:
    reward = RewardCalculator(RewardConfig.from_dict(cfg))
    env = AdaptiveQAEnv(items, reward, worlds.env_config(cfg))
    oracle_cfg = OracleConfig(max_states_per_question=1)
    counts = Counter()
    by_group = defaultdict(Counter)
    per_qid = []
    for world in items:
        target = oracle_episode(env, world.qid, oracle_cfg).targets[0]
        values = target.q_values
        searches = {action: values[action] for action in ("SEARCH_BM25", "SEARCH_DENSE") if action in values}
        terminal = {action: values[action] for action in ("ANSWER", "ABSTAIN") if action in values}
        best_search = max(searches.values()) if searches else None
        best_terminal = max(terminal.values())
        advantage = best_search - best_terminal if best_search is not None else None
        beneficial = advantage is not None and advantage > TOLERANCE
        tied = advantage is not None and abs(advantage) <= TOLERANCE
        category = "impossible" if world.private.is_impossible else "answerable"
        label = "beneficial" if beneficial else "tie" if tied else "not_beneficial"
        counts[label] += 1
        counts[f"{category}_{label}"] += 1
        counts[category] += 1
        counts[f"oracle_{target.expected}"] += 1
        by_group[groups[world.qid]][label] += 1
        by_group[groups[world.qid]]["total"] += 1
        if beneficial:
            for action in searches:
                if searches[action] - best_terminal > TOLERANCE:
                    counts[f"beneficial_{action}"] += 1
            counts[f"beneficial_best_terminal_{max(terminal, key=terminal.get)}"] += 1
            if "ANSWER" in values and values["ANSWER"] == reward.config.wrong_answer:
                counts["beneficial_root_answer_wrong"] += 1
        per_qid.append({
            "qid": world.qid, "title_group": groups[world.qid], "category": category,
            "gold_reachable": world.private.gold_reachable, "q_root": values,
            "best_search_minus_best_terminal": advantage,
            "beneficial_search": beneficial, "root_tie_search_terminal": tied,
            "oracle_root_action": target.expected,
        })
    if len(per_qid) != len(tune) or {row["qid"] for row in per_qid} != set(tune):
        raise ValueError("oracle did not evaluate every frozen tune QID exactly once")
    if sum(counts[name] for name in ("beneficial", "tie", "not_beneficial")) != len(tune):
        raise AssertionError("oracle classification is incomplete")
    return {
        "protocol_sha256": PROTOCOL_SHA256, "reward_spec_hash": reward.config.spec_hash(),
        "criterion": "strict root SEARCH advantage > 1e-8; ties excluded",
        "n_tune": len(tune), "n_groups": len(by_group),
        "counts": dict(sorted(counts.items())),
        "beneficial_fraction": counts["beneficial"] / len(tune),
        "by_title_group": {key: dict(value) for key, value in sorted(by_group.items())},
        "per_qid": per_qid,
        "interpretation_limits": (
            "Privileged, deterministic in-corpus oracle on previously exposed validation; "
            "not a learned-policy score, generalization estimate, or held-out test result. "
            "Q values exclude the common forced bootstrap cost, which cancels in comparisons. "
            "A beneficial first SEARCH may be followed by another SEARCH before termination."
        ),
    }


def main() -> None:
    if RESULT.exists():
        raise FileExistsError(f"refusing to overwrite audit result: {RESULT}")
    cfg = load_config("medium", extra=read_yaml("configs/rl_audit_medium.yaml"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    tune, groups = verify_protocol(cfg, protocol)
    items = tune_worlds(cfg, tune)
    result = measure(cfg, tune, groups, items)
    with RESULT.open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    print(f"{result['counts']['beneficial']}/{result['n_tune']} root SEARCH beneficial "
          f"({result['beneficial_fraction']:.2%}); groups={result['n_groups']}; result={RESULT}")


if __name__ == "__main__":
    main()
