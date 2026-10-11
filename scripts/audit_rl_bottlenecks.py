"""Gold-side retrieval/reader diagnosis on medium train and locked val_tune only.

Run with .venv/bin/python -m scripts.audit_rl_bottlenecks. Never feeds labels or
oracle results to the policy. Does not open reserved validation/test label columns.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from scripts.audit_rl_search_value import PROTOCOL, verify_protocol, tune_worlds
from src.data import artifacts
from src.pipeline import cache_reader, train_reader, worlds
from src.qa.metrics import f1_score
from src.qa.reader import ReaderCache, ReaderPrediction
from src.qa.state_cache import load_state_cache_meta
from src.retrieval.states import reachable_states
from src.rl.env import EpisodeWorld
from src.rl.reward import RewardConfig
from src.utils.config import load_config, resolve
from src.utils.hashing import sha256_file, sha256_json
from src.utils.io import read_yaml

TRAIN = "train_strict"
OUTPUT = Path("outputs/experiments/2026-10-11-rl-bottleneck/diagnosis.json")


def candidate_label(prediction: ReaderPrediction, answer_texts: tuple[str, ...],
                    *, threshold: float, impossible: bool) -> tuple[str, float]:
    if prediction.answer_text is None:
        return ("null" if impossible else "no_candidate"), 0.0
    if not prediction.provenance_ok:
        return "invalid_provenance", 0.0
    if impossible:
        return "false_answer", 0.0
    score = f1_score(prediction.answer_text, list(answer_texts))
    if score >= threshold - 1e-12:
        return "correct", score
    return ("wrong_zero_f1" if score == 0 else "partial_below_threshold"), score


def classify(world: EpisodeWorld, *, threshold: float,
             max_optional_search: int, bootstrap_tool: str = "bm25",
             bootstrap_passages: int = 1, passage_per_search_action: int = 1) -> dict:
    """Partition by actual retrieval histories, never by gold-chosen tool at inference."""
    states = reachable_states(
        world.rankings, bootstrap_tool=bootstrap_tool, bootstrap_passages=bootstrap_passages,
        passage_per_search_action=passage_per_search_action,
        max_optional_search=max_optional_search)
    gold = world.private.gold_doc_id if not world.private.is_impossible else None
    if world.private.is_impossible:
        gold_position = "not_applicable"
    elif gold in states[0]:
        gold_position = "bootstrap"
    elif any(gold in state for state in states if len(state) - len(states[0]) == 1):
        gold_position = "first_search"
    elif any(gold in state for state in states if len(state) - len(states[0]) == 2):
        gold_position = "second_search"
    else:
        gold_position = "unreachable"
    evaluated = []
    for state in states:
        prediction = world.reader(state)
        label, score = candidate_label(prediction, world.private.answer_texts,
                                       threshold=threshold, impossible=world.private.is_impossible)
        evaluated.append((state, label, score, prediction.doc_id))
    bootstrap_reader = evaluated[0][1]
    with_gold = [entry for entry in evaluated if gold is not None and gold in entry[0]]
    labels = [entry[1] for entry in with_gold]
    if not with_gold:
        best_with_gold = "unreachable" if gold is not None else "not_applicable"
    elif "correct" in labels:
        best_with_gold = "correct"
    elif "partial_below_threshold" in labels:
        best_with_gold = "partial_below_threshold"
    elif "wrong_zero_f1" in labels:
        best_with_gold = "wrong_zero_f1"
    elif "invalid_provenance" in labels:
        best_with_gold = "invalid_provenance"
    else:
        best_with_gold = "no_candidate"
    search_recovery = (not world.private.is_impossible and bootstrap_reader != "correct"
                       and any(label == "correct" and len(state) > len(states[0])
                               for state, label, _, _ in evaluated))
    return {
        "qid": world.qid, "is_impossible": world.private.is_impossible,
        "gold_position": gold_position, "bootstrap_reader": bootstrap_reader,
        "best_with_gold_reader": best_with_gold,
        "any_reachable_reader_correct": any(e[1] == "correct" for e in evaluated),
        "search_recovers_reader": search_recovery,
        "gold_passage_selected_at_some_correct_state": any(
            label == "correct" and doc_id == gold for _, label, _, doc_id in with_gold),
        "n_reachable_states": len(evaluated),
    }


def summarize(rows: list[dict]) -> dict:
    answerable = [row for row in rows if not row["is_impossible"]]
    negative = [row for row in rows if row["is_impossible"]]
    return {
        "n": len(rows), "n_answerable": len(answerable), "n_impossible": len(negative),
        "gold_position": dict(Counter(row["gold_position"] for row in rows)),
        "bootstrap_answerable": dict(Counter(row["bootstrap_reader"] for row in answerable)),
        "bootstrap_impossible": dict(Counter(row["bootstrap_reader"] for row in negative)),
        "best_with_gold_reader": dict(Counter(row["best_with_gold_reader"] for row in answerable)),
        "search_recovers_reader": sum(row["search_recovers_reader"] for row in answerable),
        "search_recovery_by_gold_position": dict(Counter(
            row["gold_position"] for row in answerable if row["search_recovers_reader"])),
        "any_reachable_reader_correct": sum(row["any_reachable_reader_correct"] for row in answerable),
    }


def train_worlds(cfg: dict) -> tuple[list[EpisodeWorld], set[str], dict[str, str]]:
    """Build only train_strict; never use build_worlds (it reads test documents)."""
    processed = resolve(cfg, "data_processed")
    scenarios = artifacts.read_scenarios(resolve(cfg, "data_scenarios"), TRAIN)
    wanted = {s.qid for s in scenarios}
    if len(wanted) != len(scenarios):
        raise ValueError("duplicate train scenario QID")
    rows = pq.read_table(str(artifacts.questions_path(processed, TRAIN)),
                         filters=[("qid", "in", sorted(wanted))],
                         columns=["qid", "question", "title_group", "is_impossible",
                                  "gold_doc_id", "answer_texts_display"]).to_pylist()
    labels = artifacts.questions_by_qid(rows)
    if set(labels) != wanted:
        raise ValueError("train scenario missing gold label")
    docs = artifacts.read_docs(processed, TRAIN)
    rankings = worlds._load_rankings(cfg, TRAIN)
    cache_path, meta_path = cache_reader.state_cache_paths(cfg, TRAIN)
    meta = load_state_cache_meta(meta_path)
    config = worlds.env_config(cfg)
    worlds._check_state_cache_budget(meta, config)
    worlds._check_state_cache_sources(cfg, TRAIN, meta)
    worlds._check_state_cache_reader(cfg, meta)
    entries = {}
    for row in pq.read_table(str(cache_path), filters=[("qid", "in", sorted(wanted))]).to_pylist():
        if row["schema_version"] != "v1" or row["qid"] not in wanted or row["cache_key"] in entries:
            raise ValueError("train reader cache schema, QID, or key invalid")
        entries[row["cache_key"]] = ReaderPrediction(
            qid=row["qid"], doc_id=row["doc_id"], answer_text=row["answer_text"],
            start_char=row["start_char"], end_char=row["end_char"],
            non_null_score=float(row["non_null_score"]), null_score=float(row["null_score"]),
            null_margin=float(row["null_margin"]), reader_model_hash=row["reader_model_hash"],
            provenance_ok=bool(row["provenance_ok"]))
    cache = ReaderCache(entries)
    result = []
    for scenario in sorted(scenarios, key=lambda s: s.qid):
        row = rankings.get(scenario.scenario_id)
        label = labels[scenario.qid]
        if not label["is_impossible"] and label["gold_doc_id"] != scenario.gold_doc_id:
            raise ValueError(f"{scenario.qid}: train gold mismatch")
        if row is None or row["qid"] != scenario.qid:
            raise ValueError(f"{scenario.qid}: missing train ranking")
        if any(doc_id not in docs for doc_id in scenario.corpus_doc_ids):
            raise ValueError(f"{scenario.qid}: train document missing")
        corpus = {doc_id: docs[doc_id] for doc_id in scenario.corpus_doc_ids}
        ranking = {tool: list(row["rankings"][tool]) for tool in worlds.TOOLS}
        result.append(EpisodeWorld(
            qid=scenario.qid, question=label["question"], split=TRAIN,
            scenario_id=scenario.scenario_id, corpus=corpus, rankings=ranking,
            private=worlds._private(scenario, label, corpus, ranking, config),
            reader=worlds._reader(cache, meta, scenario.qid)))
    flagged = set(worlds.load_ambiguous_qids(cfg, TRAIN))
    return result, flagged, {row["qid"]: row["title_group"] for row in rows}


def gold_context_val(cfg: dict, tune: list[str], items: list[EpisodeWorld], threshold: float) -> dict:
    """Gold-only reader condition, not reachable-state result or policy input."""
    path = train_reader.reader_dir(cfg) / train_reader.PREDICTIONS_FILE
    rows = pq.read_table(str(path), filters=[("qid", "in", tune),
                                             ("condition", "==", "oracle_context")]).to_pylist()
    # One gold-context document per QID; null predictions are not marked is_candidate.
    chosen = {row["qid"]: row for row in rows}
    if len(rows) != len(items) or len(chosen) != len(items):
        raise ValueError("gold-context reader rows missing or duplicated for val_tune")
    counts = Counter()
    for world in items:
        row = chosen[world.qid]
        category, _ = candidate_label(
            ReaderPrediction(qid=world.qid, doc_id=row["doc_id"],
                             answer_text=row["answer_text"], start_char=row["start_char"],
                             end_char=row["end_char"], non_null_score=float(row["non_null_score"]),
                             null_score=float(row["null_score"]), null_margin=float(row["null_margin"]),
                             reader_model_hash=row["reader_model_hash"],
                             provenance_ok=bool(row["provenance_ok"])),
            world.private.answer_texts, threshold=threshold, impossible=world.private.is_impossible)
        counts[("impossible" if world.private.is_impossible else "answerable", category)] += 1
    return {name: {label: n for (kind, label), n in counts.items() if kind == name}
            for name in ("answerable", "impossible")}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite: {OUTPUT}")
    cfg = load_config("medium", extra=read_yaml("configs/rl_audit_medium.yaml"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    tune, groups = verify_protocol(cfg, protocol)
    train, flagged, train_groups = train_worlds(cfg)
    val = tune_worlds(cfg, tune)
    reward = RewardConfig.from_dict(cfg)
    config = worlds.env_config(cfg)
    options = {"threshold": reward.candidate_f1_threshold,
               "max_optional_search": config.max_optional_search,
               "bootstrap_tool": config.bootstrap_tool,
               "bootstrap_passages": config.bootstrap_passages,
               "passage_per_search_action": config.passage_per_search_action}
    train_rows = [classify(w, **options) for w in train]
    val_rows = [classify(w, **options) for w in val]
    clean = [row for row in train_rows if row["qid"] not in flagged]
    if {r["qid"] for r in val_rows} != set(tune):
        raise AssertionError("val_tune QIDs drifted")
    processed = resolve(cfg, "data_processed")
    cache_train, meta_train = cache_reader.state_cache_paths(cfg, TRAIN)
    sources = {
        "train_questions": artifacts.questions_path(processed, TRAIN),
        "train_scenarios": artifacts.scenarios_path(resolve(cfg, "data_scenarios"), TRAIN),
        "train_rankings": worlds.build_retrieval.rankings_path(cfg, TRAIN),
        "train_reader_cache": cache_train, "train_reader_meta": meta_train,
        "val_oracle_context_predictions": train_reader.reader_dir(cfg) / train_reader.PREDICTIONS_FILE,
    }
    report = {
        "protocol_sha256": sha256_file(PROTOCOL), "reward_spec_hash": reward.spec_hash(),
        "inputs_sha256": {name: sha256_file(path) for name, path in sources.items()},
        "train_qids_sha256": sha256_json([w.qid for w in train]),
        "train_groups": len(set(train_groups.values())),
        "val_tune_groups": len({groups[qid] for qid in tune}),
        "train_flagged_in_scenarios": len({w.qid for w in train} & flagged),
        "train_all": summarize(train_rows), "train_clean": summarize(clean),
        "val_tune": summarize(val_rows),
        "val_tune_gold_context": gold_context_val(cfg, tune, val, reward.candidate_f1_threshold),
        "definitions": {
            "gold_position": "first search = any valid single optional tool action reveals gold; second = only a two-search branch; bootstrap is forced BM25",
            "best_with_gold_reader": "best category over reachable cached reader states containing gold doc; not a gold-only model prediction",
            "search_recovers_reader": "bootstrap cached candidate not F1>=0.8 but at least one longer reachable state is F1>=0.8 with provenance; privileged upper bound, not policy success",
            "gold_context": "existing frozen reader on the source passage alone; available only for val_tune (no train gold-only inference run)",
            "negative_flags": "heuristic ambiguous negatives excluded only from train_clean denominator; not human-verified errors",
        },
        "per_qid": {"train": train_rows, "val_tune": val_rows},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({key: report[key] for key in ("train_flagged_in_scenarios", "train_clean",
                                                    "val_tune", "val_tune_gold_context")},
                     ensure_ascii=False, indent=2))
    print("wrote", OUTPUT)


if __name__ == "__main__":
    main()
