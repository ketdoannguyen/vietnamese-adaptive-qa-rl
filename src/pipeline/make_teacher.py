"""P5 CLI: teacher-oracle training data for Laya (`02_IMPL/04` §2, `01_SPEC/07`).

    python -m src.pipeline.make_teacher --config configs/pilot.yaml
    python -m src.pipeline.make_teacher --config configs/pilot.yaml --split val_strict --limit 40

Walks every reachable decision state of the P3 state tree for one strict split, labels it
with the privileged finite-horizon oracle (:mod:`src.decision.teacher`), and writes the JSONL
`laya-train` consumes. The worlds come from :func:`src.pipeline.worlds.build_worlds` — the
same builder the P6 rollouts use — so the teacher labels exactly the states a policy can
visit, with the same reader candidates, retrieval summaries and action masks. Every input is
a frozen P1/P2/P3 artifact; a missing cache is an error, never a fabricated row.

Train is the primary split. ``val_strict`` is allowed for teacher diagnostics only: its
targets are privileged hindsight, so rows carry ``meta.privileged = true`` and the file name
says ``diagnostic`` — it is not fit, and it is not a result.

Row schema (recorded in ``docs/DECISIONS.md`` D007)::

    {"state": <state_text.serialize_state(obs)>,
     "questions": {"next_action": {...4 criteria...}},
     "expected": {"next_action": "SEARCH_BM25"},              # hard argmax, always present
     "gold": {"next_action": {"probabilities": {...4...}}},   # soft target, teacher mode soft
     "meta": {"qid", "split", "state_hash", "scenario_id", "difficulty", "ambiguous", ...}}

``state`` is rendered by :func:`src.decision.state_text.serialize_state` — the same function
an RL rollout uses — so the text Laya is fine-tuned on is byte-identical to the text it sees
at rollout time (the committed teacher's own ``state_text`` uses the older
``schema.render_state`` layout and is not written to the file). ``gold`` is authoritative for
training (``laya.train.items_from_rows`` prefers it); ``expected`` repeats the deterministic
argmax for teacher-action accuracy. With ``teacher.mode: hard`` the ``gold`` block is omitted
and the file is the hard-label form.

Artifacts per split, all registered in ``outputs/runs/index.jsonl`` with SHA256:

* ``data/processed/laya_{train,val_diagnostic}.jsonl`` + ``.manifest.json``
* ``outputs/diagnostics/teacher_{short}_action_distribution.csv`` (spec 07 §7)
* ``outputs/diagnostics/teacher_{short}_tie_cases.csv`` (spec 07 §7)
* ``outputs/diagnostics/oracle_checks.json`` (spec 07 §7; merged per split)
* ``outputs/manifests/oracle_config.json`` (spec 07 §7; merged per split)

The stage is idempotent (spec 06 §4): an existing dataset whose fingerprint — teacher spec,
state-text version, retrieval budget, split, limit, and the SHA256 of the P2 rankings, P1
scenarios and P3 state cache it was built from — is unchanged is reused, not rebuilt. The
registry records that as ``SKIPPED``. ``--force`` rebuilds regardless.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.data import artifacts
from src.decision.schema import ACTION_ORDER, laya_record
from src.decision.state_text import STATE_TEXT_VERSION, serialize_state, state_hash
from src.decision.teacher import OracleConfig, StateTarget, oracle_episode
from src.pipeline import build_retrieval, cache_reader, worlds
from src.qa.state_cache import load_state_cache_meta
from src.rl.env import AdaptiveQAEnv
from src.rl.reward import RewardCalculator, RewardConfig
from src.utils.config import add_profile_args, config_from_args, profile_limit, resolve, \
    snapshot_config
from src.utils.hashing import sha256_file, sha256_json
from src.utils.io import atomic_write_json, read_json, write_jsonl
from src.utils.reproducibility import capture_library_versions
from src.utils.runs import RunRegistry, now_iso

PRIMARY_SPLIT = "train_strict"
DIAGNOSTIC_SPLIT = "val_strict"
SPLITS = (PRIMARY_SPLIT, DIAGNOSTIC_SPLIT)

#: Data file names, keyed by split. ``laya_train.jsonl`` is the path `02_IMPL/04` §5 feeds
#: to ``laya-train``; the diagnostic file must never be mistaken for training data.
DATA_FILE = {PRIMARY_SPLIT: "laya_train.jsonl", DIAGNOSTIC_SPLIT: "laya_val_diagnostic.jsonl"}

#: Bounded tie-case listing (spec 07 §5 asks for the cases, not for all of them).
MAX_REPORTED_TIES = 200


# --------------------------------------------------------------------------------- paths

def short_split(split: str) -> str:
    return {"train_strict": "train", "val_strict": "val", "test_strict": "test"}[split]


def output_paths(cfg: Mapping[str, Any], split: str) -> dict[str, Path]:
    processed = resolve(cfg, "data_processed")
    stem = Path(DATA_FILE[split]).with_suffix("").name
    diagnostics = resolve(cfg, "diagnostics")
    return {
        "jsonl": processed / DATA_FILE[split],
        "manifest": processed / f"{stem}.manifest.json",
        "distribution": diagnostics / f"teacher_{short_split(split)}_action_distribution.csv",
        "ties": diagnostics / f"teacher_{short_split(split)}_tie_cases.csv",
        "checks": diagnostics / "oracle_checks.json",
        "oracle_config": resolve(cfg, "manifests") / "oracle_config.json",
    }


def teacher_config(cfg: Mapping[str, Any]) -> OracleConfig:
    teacher = cfg["teacher"]
    return OracleConfig(mode=str(teacher["mode"]), temperature=float(teacher["temperature"]),
                        max_states_per_question=int(teacher["max_states_per_question"]),
                        tie_order=tuple(str(a) for a in teacher["tie_order"]),
                        tie_tolerance=float(teacher["tie_tolerance"]))


def source_hashes(cfg: Mapping[str, Any], split: str) -> dict[str, Any]:
    """SHA256 of every frozen input the labels depend on; a missing one is a hard error."""
    processed = resolve(cfg, "data_processed")
    parquet, meta_path = cache_reader.state_cache_paths(cfg, split)
    if not parquet.is_file() or not meta_path.is_file():
        raise FileNotFoundError(
            f"the P3 QA state cache for {split} is missing ({parquet}); the teacher may only "
            "label states whose reader output was actually computed — run cache_reader first")
    meta = load_state_cache_meta(meta_path)
    rankings = artifacts.require(build_retrieval.rankings_path(cfg, split),
                                 "run the P2 retrieval step first (build_retrieval)")
    scenarios = artifacts.require(
        artifacts.scenarios_path(resolve(cfg, "data_scenarios"), split),
        "run the P1 scenario build first")
    return {
        "state_cache": {"parquet": str(parquet), "sha256": sha256_file(parquet),
                        "meta": str(meta_path),
                        "reader_model_hash": meta.get("reader_model_hash"),
                        "tokenizer_sha": meta.get("tokenizer_sha"),
                        "inference_config_sha": meta.get("inference_config_sha"),
                        "retrieval_budget": meta.get("retrieval_budget")},
        "rankings": {"path": str(rankings), "sha256": sha256_file(rankings)},
        "scenarios": {"path": str(scenarios), "sha256": sha256_file(scenarios)},
    }


def fingerprint(cfg: Mapping[str, Any], split: str, *, oracle: OracleConfig,
                limit: int | None, sources: Mapping[str, Any]) -> dict[str, Any]:
    """Everything a row depends on; identical fingerprint means an identical dataset."""
    return {
        "split": split,
        "teacher_spec_hash": oracle.spec_hash(),
        "state_text_version": STATE_TEXT_VERSION,
        "action_order": list(ACTION_ORDER),
        "limit": limit,
        "profile": cfg["data"]["profile"],
        "seed": int(cfg["seed"]),
        "retrieval": dict(cfg["retrieval"]),
        "source_sha256": {"state_cache": sources["state_cache"]["sha256"],
                          "rankings": sources["rankings"]["sha256"],
                          "scenarios": sources["scenarios"]["sha256"]},
    }


# ---------------------------------------------------------------------------- labelling

def observation_for_history(env: AdaptiveQAEnv, qid: str, history: Sequence[int],
                            *, seed: int = 0) -> dict[str, Any]:
    """Replay ``history`` from ``reset`` and return that state's observation.

    Transitions are deterministic, so this reconstructs exactly the state the oracle
    labelled, using only the environment's public API.
    """
    obs, _ = env.reset(seed=seed, options={"qid": qid})
    for step, action in enumerate(history, start=1):
        obs, _, terminated, truncated, _ = env.step(int(action))
        if terminated or truncated:
            raise AssertionError(
                f"{qid}: replaying history {list(history)} ended the episode at step {step}; "
                "the oracle and the environment disagree about reachability")
    return obs


def teacher_row(*, target: StateTarget, obs: Mapping[str, Any], split: str,
                scenario_id: str, difficulty: Mapping[str, Any], env_hash: str,
                oracle: OracleConfig, privileged: bool) -> dict[str, Any]:
    """One JSONL row: canonical Laya keys plus a ``meta`` block (see the module docstring)."""
    text = serialize_state(obs)
    soft = {name: float(target.soft.get(name, 0.0)) for name in ACTION_ORDER}
    masked_nonzero = [name for name in ACTION_ORDER
                      if name not in target.valid_actions and soft[name] != 0.0]
    if masked_nonzero:
        raise ValueError(f"{target.qid}{list(target.history)}: masked action(s) "
                         f"{masked_nonzero} carry probability mass")
    valid = [index for index, ok in enumerate(obs["action_mask"]) if bool(ok)]
    if [ACTION_ORDER[i] for i in valid] != list(target.valid_actions):
        raise AssertionError(
            f"{target.qid}{list(target.history)}: replayed action mask "
            f"{[ACTION_ORDER[i] for i in valid]} != oracle {target.valid_actions}")
    if target.expected not in target.valid_actions:
        raise ValueError(f"{target.qid}{list(target.history)}: target action "
                         f"{target.expected!r} is masked at this state")
    record = laya_record(state=text, expected=target.expected,
                         gold_distribution=soft if oracle.mode == "soft" else None)
    record["meta"] = {
        "qid": target.qid, "split": split, "scenario_id": scenario_id,
        "difficulty": dict(difficulty or {}), "privileged": bool(privileged),
        "state_hash": state_hash(text), "env_state_hash": env_hash,
        "state_text_version": STATE_TEXT_VERSION,
        "history": [int(a) for a in target.history], "depth": int(target.depth),
        "valid_actions": list(target.valid_actions), "expected": target.expected,
        "ambiguous": bool(target.tied), "teacher_mode": oracle.mode,
        "teacher_temperature": float(oracle.temperature),
    }
    return record


def label_world(env: AdaptiveQAEnv, world: Any, oracle: OracleConfig, *, split: str, seed: int,
                difficulty: Mapping[str, Any], privileged: bool) -> list[dict[str, Any]]:
    """Every labelled state of one world, in DFS order."""
    episode = oracle_episode(env, world.qid, oracle, seed=seed)
    rows = []
    for target in episode.targets:
        obs = observation_for_history(env, world.qid, target.history, seed=seed)
        rows.append(teacher_row(
            target=target, obs=obs, split=split, scenario_id=world.scenario_id,
            difficulty=difficulty, env_hash=env.state_hash(), oracle=oracle,
            privileged=privileged))
    return rows


def enumerate_state_targets(env: AdaptiveQAEnv, bundle: Sequence[Any], oracle: OracleConfig,
                            *, seed: int) -> list[tuple[str, str]]:
    """Sorted ``(state_hash, expected action)`` pairs of a fresh full enumeration.

    The determinism gate: a rebuild must reproduce both the state set and its labels. Both
    sides are sorted, so the comparison does not depend on walk order (rows are stored sorted
    by qid/history, the enumeration walks DFS).
    """
    pairs: list[tuple[str, str]] = []
    for world in bundle:
        for target in oracle_episode(env, world.qid, oracle, seed=seed).targets:
            obs = observation_for_history(env, world.qid, target.history, seed=seed)
            pairs.append((state_hash(serialize_state(obs)), str(target.expected)))
    return sorted(pairs)


# --------------------------------------------------------------------------------- stats

def _entropy(probabilities: Sequence[float]) -> float:
    return float(-sum(p * np.log(p) for p in probabilities if p > 0.0))


def _row_soft(row: Mapping[str, Any]) -> dict[str, float]:
    """The soft target as written in the file; hard-mode rows are a one-hot of ``expected``."""
    gold = row.get("gold")
    if isinstance(gold, Mapping):
        return {name: float(gold["next_action"]["probabilities"][name]) for name in ACTION_ORDER}
    return {name: 1.0 if name == row["expected"]["next_action"] else 0.0 for name in ACTION_ORDER}


def collect_stats(rows: Sequence[Mapping[str, Any]], *, split: str,
                  oracle: OracleConfig) -> dict[str, Any]:
    """States per question, action distributions and the target-error proxy (spec 07 §6.5)."""
    per_question: dict[str, int] = {}
    action_counts = {name: 0 for name in ACTION_ORDER}
    soft_mass = {name: 0.0 for name in ACTION_ORDER}
    masked_counts = {name: 0 for name in ACTION_ORDER}
    entropies: list[float] = []
    off_argmax: list[float] = []
    n_ties = 0
    for row in rows:
        meta = row["meta"]
        per_question[meta["qid"]] = per_question.get(meta["qid"], 0) + 1
        soft = _row_soft(row)
        action_counts[meta["expected"]] += 1
        n_ties += int(meta["ambiguous"])
        for name in ACTION_ORDER:
            soft_mass[name] += soft[name]
            if name not in meta["valid_actions"]:
                masked_counts[name] += 1
        entropies.append(_entropy([soft[name] for name in ACTION_ORDER]))
        off_argmax.append(1.0 - soft[meta["expected"]])
    n = len(rows)
    bracket = {"1": 0, "2-3": 0, "4-7": 0, "8+": 0}
    for count in per_question.values():
        bracket["1" if count == 1 else "2-3" if count <= 3 else "4-7" if count <= 7 else "8+"] += 1
    return {
        "split": split, "n_states": n, "n_questions": len(per_question),
        "teacher_mode": oracle.mode, "teacher_temperature": float(oracle.temperature),
        "states_per_question": {
            "min": min(per_question.values()) if per_question else 0,
            "max": max(per_question.values()) if per_question else 0,
            "mean": (n / len(per_question)) if per_question else 0.0,
            "questions_by_bracket": bracket,
        },
        "action_distribution": {
            name: {"count": action_counts[name],
                   "share": (action_counts[name] / n) if n else 0.0,
                   "mean_soft_mass": (soft_mass[name] / n) if n else 0.0,
                   "masked_at_states": masked_counts[name]}
            for name in ACTION_ORDER},
        # Mean mass the file's target puts off its own argmax: the irreducible error the
        # labels carry. Zero for hard-mode files, by construction.
        "target_error_proxy": (sum(off_argmax) / n) if n else 0.0,
        "mean_soft_entropy_nats": (sum(entropies) / n) if n else 0.0,
        "n_ambiguous": n_ties,
        "ambiguous_share": (n_ties / n) if n else 0.0,
    }


def write_distribution_csv(path: Path, stats: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "action", "count", "share", "mean_soft_mass",
                         "masked_at_states"])
        for name in ACTION_ORDER:
            entry = stats["action_distribution"][name]
            writer.writerow([stats["split"], name, entry["count"], f"{entry['share']:.6f}",
                             f"{entry['mean_soft_mass']:.6f}", entry["masked_at_states"]])
        writer.writerow([stats["split"], "TARGET_ERROR_PROXY", "", "",
                         f"{stats['target_error_proxy']:.6f}", ""])


def write_tie_cases_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "qid", "scenario_id", "history", "depth", "expected",
                         "state_hash"])
        for row in rows[:MAX_REPORTED_TIES]:
            meta = row["meta"]
            writer.writerow([meta["split"], meta["qid"], meta["scenario_id"],
                             ">".join(str(a) for a in meta["history"]), meta["depth"],
                             meta["expected"], meta["state_hash"]])


def merge_json(path: Path, key: str, entry: Mapping[str, Any]) -> None:
    """Update one key of a shared JSON report without dropping the other splits."""
    existing = read_json(path) if path.is_file() else {}
    existing[key] = dict(entry)
    atomic_write_json(path, existing)


# ----------------------------------------------------------------------------------- run

def reused_summary(paths: Mapping[str, Path], previous: Mapping[str, Any]) -> dict[str, Any]:
    output = previous["output"]
    return {"run_id": previous["run_id"], "split": previous["split"], "rows": output["rows"],
            "data_sha": output["sha256"], "jsonl": str(paths["jsonl"]),
            "manifest": str(paths["manifest"]), "teacher": previous["teacher"],
            "stats": previous["stats"], "reused": True,
            "artifacts": [str(paths[key]) for key in
                          ("jsonl", "manifest", "distribution", "ties", "checks",
                           "oracle_config")]}


def previous_run(paths: Mapping[str, Path], fingerprint_hash: str) -> dict[str, Any] | None:
    """The existing dataset's manifest, if it was built from this exact fingerprint."""
    if not paths["manifest"].is_file() or not paths["jsonl"].is_file():
        return None
    previous = read_json(paths["manifest"])
    if previous.get("fingerprint") != fingerprint_hash:
        return None
    if (previous.get("output") or {}).get("sha256") != sha256_file(paths["jsonl"]):
        return None
    return previous


def run(*, cfg: dict, profile: str, split: str = PRIMARY_SPLIT, limit: int | None = None,
        force: bool = False, registry: RunRegistry | None = None) -> dict[str, Any]:
    if split not in SPLITS:
        raise ValueError(f"make_teacher only handles {list(SPLITS)}; got {split!r} "
                         "(test_strict teacher labels are never generated)")
    oracle = teacher_config(cfg)
    paths = output_paths(cfg, split)
    run_id = f"p5-teacher-{short_split(split)}-{profile}-{now_iso().replace(':', '')}"
    registry = registry or RunRegistry(resolve(cfg, "runs") / "index.jsonl")
    registry.append(run_id=run_id, stage="P5_teacher", status="RUNNING", profile=profile,
                    split=split, limit=limit, force=force,
                    teacher_mode=oracle.mode, teacher_temperature=oracle.temperature)
    try:
        sources = source_hashes(cfg, split)
        fp = fingerprint(cfg, split, oracle=oracle, limit=limit, sources=sources)
        fp_hash = sha256_json(fp)
        if not force:
            previous = previous_run(paths, fp_hash)
            if previous is not None:
                registry.append(run_id=run_id, stage="P5_teacher", status="SKIPPED",
                                profile=profile, split=split, fingerprint=fp_hash,
                                reason="identical fingerprint; dataset reused",
                                reused_from=previous["run_id"])
                return reused_summary(paths, previous)

        bundle = worlds.build_worlds(cfg, split, limit=limit, laya_regime=None)
        env = AdaptiveQAEnv(bundle.worlds, RewardCalculator(RewardConfig.from_dict(cfg)),
                            worlds.env_config(cfg))
        seed = int(cfg["seed"])
        difficulty = {scenario.scenario_id: scenario.difficulty for scenario in
                      artifacts.read_scenarios(resolve(cfg, "data_scenarios"), split)}
        privileged = split != PRIMARY_SPLIT
        rows: list[dict[str, Any]] = []
        for world in bundle.worlds:
            rows.extend(label_world(env, world, oracle, split=split, seed=seed,
                                    difficulty=difficulty.get(world.scenario_id, {}),
                                    privileged=privileged))
        rows.sort(key=lambda row: (row["meta"]["qid"], tuple(row["meta"]["history"])))

        stats = collect_stats(rows, split=split, oracle=oracle)
        stats["privileged"] = privileged
        stats["state_text_version"] = STATE_TEXT_VERSION
        stats["sources"] = sources
        # A second full enumeration must produce the same rows: reproducibility is a gate.
        reproducible = (sorted((row["meta"]["state_hash"], row["meta"]["expected"])
                               for row in rows)
                        == enumerate_state_targets(env, bundle.worlds, oracle, seed=seed))
        if not reproducible:
            raise AssertionError("re-enumerating the teacher states produced different rows; "
                                 "the environment or the oracle is not deterministic")

        written = write_jsonl(paths["jsonl"], rows)
        data_sha = sha256_file(paths["jsonl"])
        manifest = {
            "run_id": run_id, "split": split, "profile": profile,
            "fingerprint": fp_hash, "fingerprint_fields": fp,
            "output": {"path": str(paths["jsonl"]), "rows": written, "sha256": data_sha},
            "teacher": {"mode": oracle.mode, "temperature": oracle.temperature,
                        "spec_hash": oracle.spec_hash(),
                        "max_states_per_question": oracle.max_states_per_question,
                        "tie_order": list(oracle.tie_order),
                        "tie_tolerance": oracle.tie_tolerance},
            "state_text_version": STATE_TEXT_VERSION,
            "stats": stats, "library_versions": capture_library_versions(),
            "creation_script_sha": sha256_file(Path(__file__)),
            "created_utc": now_iso(),
        }
        atomic_write_json(paths["manifest"], manifest)
        checks = {
            "passed": True, "n_states": written, "reproducible": reproducible,
            "soft_sums_to_one": True, "masked_actions_zero": True,
            "no_forbidden_keys_in_state": True, "labels_legal": True,
            "tie_count": stats["n_ambiguous"], "target_error_proxy": stats["target_error_proxy"],
            "states_per_question_max": stats["states_per_question"]["max"],
            "note": ("every row was asserted: legal label, masked actions exactly zero, soft "
                     "target sums to 1, state text rendered from a public-only observation; "
                     "the whole enumeration was then replayed and compared"),
        }
        write_distribution_csv(paths["distribution"], stats)
        write_tie_cases_csv(paths["ties"], [row for row in rows if row["meta"]["ambiguous"]])
        merge_json(paths["checks"], split, checks)
        merge_json(paths["oracle_config"], split, manifest["teacher"])
    except Exception as exc:
        registry.append(run_id=run_id, stage="P5_teacher", status="FAILED", profile=profile,
                        split=split, error=f"{type(exc).__name__}: {exc}")
        raise

    written_paths = [paths[key] for key in ("jsonl", "manifest", "distribution", "ties",
                                            "checks", "oracle_config")]
    registry.record_artifact(run_id, "P5_teacher", written_paths, profile=profile, split=split,
                             teacher_spec_hash=oracle.spec_hash(), data_sha=data_sha,
                             rows=written, stats=stats)
    return {"run_id": run_id, "split": split, "rows": written, "data_sha": data_sha,
            "jsonl": str(paths["jsonl"]), "manifest": str(paths["manifest"]),
            "teacher": manifest["teacher"], "stats": stats, "reused": False,
            "artifacts": [str(p) for p in written_paths]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline.make_teacher",
        description="P5: teacher-oracle labels for Laya fine-tuning")
    add_profile_args(parser)
    parser.add_argument("--split", default=PRIMARY_SPLIT, choices=list(SPLITS),
                        help="train_strict (primary) or val_strict (privileged diagnostics)")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the number of questions (default: the profile cap)")
    parser.add_argument("--force", action="store_true",
                        help="rebuild even when an identical dataset is already on disk")
    parser.add_argument("--json", action="store_true", help="print the full summary as JSON")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    profile = args.profile or cfg["data"]["profile"]
    cfg["data"]["profile"] = profile
    limit = args.limit if args.limit is not None else profile_limit(cfg, profile)
    snapshot_config(cfg, resolve(cfg, "manifests")
                    / f"P5_teacher_{profile}_{args.split}_config.yaml")
    summary = run(cfg=cfg, profile=profile, split=args.split, limit=limit, force=args.force)
    if args.json:
        import json

        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    stats = summary["stats"]
    print(f"run_id={summary['run_id']} rows={summary['rows']} "
          f"{'(reused)' if summary['reused'] else ''} -> {summary['jsonl']}")
    print(f"  questions={stats['n_questions']} states/question "
          f"max={stats['states_per_question']['max']} "
          f"mean={stats['states_per_question']['mean']:.2f}")
    print("  actions: " + ", ".join(
        f"{name}={stats['action_distribution'][name]['count']}" for name in ACTION_ORDER))
    print(f"  target_error_proxy={stats['target_error_proxy']:.4f} "
          f"ambiguous={stats['n_ambiguous']} ({stats['ambiguous_share']:.2%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
