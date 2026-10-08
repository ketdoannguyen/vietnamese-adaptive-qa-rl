"""P3 CLI: exhaustive QA state cache for RL rollouts (`03_...md` §6; `09_...md` §2-3).

    python -m src.pipeline.cache_reader --profile small
    python -m src.pipeline.cache_reader --profile pilot --limit 200   # the 100-500 qid pass

The reader must already be trained and frozen (``reader_inference.json`` next to the
checkpoint). The cache is stamped with the reader/tokenizer/threshold hashes, so retraining
or retuning invalidates every row: the CLI reports it when it detects it, and a lookup under
new hashes fails fast instead of returning stale text.

Per split it writes, under ``outputs/models/qa_reader/``:

* ``reader_qa_state_cache_{split}.parquet`` — one row per reachable visible-doc state
* ``reader_qa_state_cache_{split}.json``    — hashes, source checksums, counts, schema version

Splits default to train_strict + val_strict. test_strict is only cached for the final locked
test, so test rows never sit in the directory the RL stages read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.data import artifacts
from src.qa.infer import ReaderInference, use_fp16
from src.qa.state_cache import SCHEMA_VERSION, build_state_cache
from src.pipeline import train_reader
from src.utils.config import add_profile_args, config_from_args, profile_limit, resolve, \
    snapshot_config
from src.utils.hashing import sha256_file
from src.utils.io import atomic_write_json, read_json, read_jsonl, write_parquet
from src.utils.reproducibility import capture_library_versions
from src.utils.runs import RunRegistry, now_iso

CACHE_SPLITS = ("train_strict", "val_strict")


def state_cache_paths(cfg: Mapping[str, Any], split: str) -> tuple[Path, Path]:
    base = train_reader.reader_dir(cfg) / f"reader_qa_state_cache_{split}"
    return Path(f"{base}.parquet"), Path(f"{base}.json")


def read_contract(cfg: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    """The frozen reader's inference contract written by ``train_reader``."""
    path = artifacts.require(train_reader.reader_dir(cfg) / train_reader.INFERENCE_FILE,
                             "run the P3 reader training step first")
    contract = read_json(path)
    for field in ("reader_model_hash", "tokenizer_sha", "inference_config_sha",
                  "null_score_diff_threshold"):
        if field not in contract:
            raise ValueError(f"{path}: missing {field!r}")
    return path, contract


def load_rankings(cfg: Mapping[str, Any], split: str) -> dict[str, dict[str, list[str]]]:
    """``qid -> {tool: ranking}`` from the P2 cache; a split without rankings fails loudly."""
    path = artifacts.require(train_reader.rankings_path(cfg, split),
                             "run the P2 retrieval step first (build_retrieval)")
    rankings: dict[str, dict[str, list[str]]] = {}
    for row in read_jsonl(path):
        qid = row["qid"]
        if qid in rankings:
            raise ValueError(f"{path}: duplicate qid {qid!r}")
        rankings[qid] = row["rankings"]
    return rankings


def source_checksums(cfg: Mapping[str, Any], split: str) -> dict[str, dict[str, Any]]:
    """The exact inputs a state cache for ``split`` was built from, with their hashes."""
    processed = resolve(cfg, "data_processed")
    paths = {
        "rankings": train_reader.rankings_path(cfg, split),
        "docs": artifacts.docs_path(processed, split),
        "questions": artifacts.questions_path(processed, split),
        "scenarios": artifacts.scenarios_path(resolve(cfg, "data_scenarios"), split),
    }
    return {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()}


def build_split_cache(cfg: Mapping[str, Any], split: str, *, profile: str, limit: int | None,
                      contract: Mapping[str, Any], inference: ReaderInference) -> dict[str, Any]:
    """Cache one split; returns the summary row (paths included)."""
    processed = resolve(cfg, "data_processed")
    docs = artifacts.read_docs(processed, split)
    questions = artifacts.question_texts(processed, split)
    scenarios = sorted(artifacts.read_scenarios(resolve(cfg, "data_scenarios"), split),
                       key=lambda s: s.scenario_id)
    if limit is not None:
        scenarios = scenarios[:int(limit)]
    if not scenarios:
        raise ValueError(f"no {split} scenarios to cache (limit={limit})")
    rankings = load_rankings(cfg, split)

    threshold = float(contract["null_score_diff_threshold"])
    rows, stats = build_state_cache(
        scenarios=scenarios, questions=questions, docs=docs, rankings=rankings,
        inference=inference, threshold=threshold, reader_model_hash=inference.model_hash,
        tokenizer_sha=inference.tokenizer_sha,
        inference_config_sha=contract["inference_config_sha"], retrieval=cfg["retrieval"])

    parquet_path, meta_path = state_cache_paths(cfg, split)
    if meta_path.exists():
        previous = read_json(meta_path)
        if previous.get("reader_model_hash") not in (None, inference.model_hash):
            print(f"{split}: previous cache was built for reader "
                  f"{str(previous['reader_model_hash'])[:12]}…, now "
                  f"{inference.model_hash[:12]}… — invalidating all rows")
    parquet_meta = write_parquet(parquet_path, rows)
    retrieval = cfg["retrieval"]
    meta = {
        "schema_version": SCHEMA_VERSION,
        "split": split, "profile": profile,
        "reader_model_hash": inference.model_hash, "tokenizer_sha": inference.tokenizer_sha,
        "inference_config_sha": contract["inference_config_sha"],
        "null_score_diff_threshold": threshold,
        "reader_contract": {"path": str(train_reader.reader_dir(cfg)
                                       / train_reader.INFERENCE_FILE)},
        "retrieval_budget": {
            "bootstrap_tool": retrieval["initial_bootstrap"],
            "bootstrap_passages": int(retrieval["bootstrap_passages"]),
            "passage_per_search_action": int(retrieval["passage_per_search_action"]),
            "optional_searches": int(retrieval["optional_searches"]),
        },
        "sources": source_checksums(cfg, split),
        "parquet": {"path": str(parquet_path), **parquet_meta},
        "creation_script_sha": sha256_file(Path(__file__)),
        "library_versions": capture_library_versions(),
        **stats,
    }
    atomic_write_json(meta_path, meta)
    return {"split": split, "parquet": str(parquet_path), "meta": str(meta_path),
            **{k: v for k, v in stats.items() if not isinstance(v, list)}}


def run(*, cfg: dict, profile: str, splits: Sequence[str] = CACHE_SPLITS,
        limit_override: int | None = None,
        registry: RunRegistry | None = None) -> dict[str, Any]:
    reader_cfg = cfg["reader"]
    run_id = f"p3-reader-cache-{profile}-{now_iso().replace(':', '')}"
    registry = registry or RunRegistry(resolve(cfg, "runs") / "index.jsonl")
    registry.append(run_id=run_id, stage="P3_reader_cache", status="RUNNING", profile=profile,
                    splits=list(splits), limit=limit_override)
    artifacts_written: list[Path] = []
    try:
        contract_path, contract = read_contract(cfg)
        inference = ReaderInference.load(
            train_reader.reader_dir(cfg), max_length=int(reader_cfg["max_length"]),
            doc_stride=int(reader_cfg["doc_stride"]),
            max_answer_length=int(reader_cfg["max_answer_length"]),
            batch_size=int(reader_cfg["inference_batch_size"]), fp16=use_fp16(reader_cfg))
        if (inference.model_hash != contract["reader_model_hash"]
                or inference.tokenizer_sha != contract["tokenizer_sha"]):
            raise RuntimeError(
                "the saved reader no longer matches reader_inference.json; retrain the reader "
                "(P3) and rebuild the cache — stale reader predictions are never reused")
        limit = limit_override if limit_override is not None else profile_limit(cfg, profile)
        per_split: dict[str, Any] = {}
        for split in splits:
            summary = build_split_cache(cfg, split, profile=profile, limit=limit,
                                        contract=contract, inference=inference)
            per_split[split] = summary
            artifacts_written += [Path(summary["parquet"]), Path(summary["meta"])]
    except Exception as exc:
        registry.append(run_id=run_id, stage="P3_reader_cache", status="FAILED", profile=profile,
                        error=f"{type(exc).__name__}: {exc}")
        raise
    registry.record_artifact(run_id, "P3_reader_cache", artifacts_written, profile=profile,
                             reader_model_hash=inference.model_hash,
                             inference_config_sha=contract["inference_config_sha"],
                             splits={s: {k: v for k, v in row.items()
                                         if k not in ("parquet", "meta")}
                                     for s, row in per_split.items()})
    return {"run_id": run_id, "reader_model_hash": inference.model_hash,
            "inference_config_sha": contract["inference_config_sha"],
            "contract": str(contract_path), "splits": per_split,
            "artifacts": [str(p) for p in artifacts_written]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline.cache_reader",
        description="P3: exhaustive QA state cache over the reachable retrieval tree")
    add_profile_args(parser)
    parser.add_argument("--splits", default=",".join(CACHE_SPLITS),
                        help="comma-separated strict splits to cache")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the number of scenarios per split (default: the profile cap)")
    parser.add_argument("--json", action="store_true", help="print the full summary as JSON")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    profile = args.profile or cfg["data"]["profile"]
    cfg["data"]["profile"] = profile
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    snapshot_config(cfg, resolve(cfg, "manifests") / f"P3_reader_cache_{profile}_config.yaml")
    summary = run(cfg=cfg, profile=profile, splits=splits, limit_override=args.limit)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    print(f"run_id={summary['run_id']} reader={summary['reader_model_hash'][:12]}…")
    for split, row in summary["splits"].items():
        print(f"  {split}: {row['n_states']} states over {row['n_questions']} questions "
              f"({row['n_null_states']} null, {row['n_pairs']} pairs, "
              f"{row['n_missing_ranking']} without ranking) -> {row['parquet']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
