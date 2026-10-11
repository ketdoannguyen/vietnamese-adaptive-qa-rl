"""Verify new strict validation/test artifacts against a frozen split manifest.

Only manifest JSON and file bytes are hashed; private question labels are never parsed.
Usage: python -m scripts.check_split_identity --reference data/processed/split_manifest.json \
    --candidate outputs/experiments/<id>/data/processed/split_manifest.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.config import REPO_ROOT
from src.utils.hashing import sha256_file
from src.utils.io import read_json

FROZEN_ARTIFACTS = tuple(
    f"{kind}_{split}_strict" for split in ("val", "test")
    for kind in ("questions", "docs", "qids")
) + ("scenarios_val", "scenarios_test")


def verify_split_identity(reference_path: str | Path,
                          candidate_path: str | Path) -> dict:
    reference_path = Path(reference_path).resolve()
    candidate_path = Path(candidate_path).resolve()
    if reference_path == candidate_path:
        raise ValueError("candidate manifest aliases reference; cannot verify a new split")
    reference = read_json(reference_path)
    candidate = read_json(candidate_path)
    for field in ("repo_id", "resolved_revision"):
        if reference["source"][field] != candidate["source"][field]:
            raise ValueError(f"source.{field} differs from frozen reference")
    for field in ("train", "validation", "test"):
        if reference["source"]["raw_rows"][field] != candidate["source"]["raw_rows"][field]:
            raise ValueError(f"source.raw_rows.{field} differs from frozen reference")
    sources = []
    for item in (reference, candidate):
        source_path = Path(item["source"]["source_manifest"])
        if not source_path.is_absolute():
            source_path = REPO_ROOT / source_path
        if not source_path.is_file() or sha256_file(source_path) != \
                item["source"]["source_manifest_sha256"]:
            raise ValueError(f"source manifest missing or mismatched: {source_path}")
        source = read_json(source_path)
        if source["resolved_revision"] != item["source"]["resolved_revision"]:
            raise ValueError("source resolved_revision disagrees with source manifest")
        if source["repo_id"] != item["source"]["repo_id"]:
            raise ValueError("source repo_id disagrees with source manifest")
        for field in ("train", "validation", "test"):
            raw = source_path.parent / source["splits"][field]["parquet"]
            if not raw.is_file() or sha256_file(raw) != source["splits"][field]["sha256"]:
                raise ValueError(f"raw {field} artifact missing or mismatched: {raw}")
        sources.append(source)
    for field in ("train", "validation", "test"):
        if sources[0]["splits"][field]["sha256"] != sources[1]["splits"][field]["sha256"]:
            raise ValueError(f"raw {field} differs from frozen reference")
    for field in ("seed", "ratios_target", "near_duplicate_merge_threshold"):
        if reference["strict_split"][field] != candidate["strict_split"][field]:
            raise ValueError(f"strict_split.{field} differs from frozen reference")
    for field in ("version", "scenario_seed", "n_distractors", "hard_negative_ratio",
                  "hard_negative_reference"):
        if reference["scenarios"][field] != candidate["scenarios"][field]:
            raise ValueError(f"scenarios.{field} differs from frozen reference")
    for split in ("val_strict", "test_strict"):
        a = reference["strict_split"]["splits"][split]
        b = candidate["strict_split"]["splits"][split]
        for field in ("rows_pool", "rows_emitted", "answerable_emitted",
                      "impossible_emitted", "title_groups_pool", "docs_emitted"):
            if a[field] != b[field]:
                raise ValueError(f"{split}.{field} differs from frozen reference")
        if b["rows_emitted"] != b["rows_pool"]:
            raise ValueError(f"{split} was capped; frozen validation/test must be complete")
        if reference["scenarios"]["splits"][split] != candidate["scenarios"]["splits"][split]:
            raise ValueError(f"{split} scenario summary differs from frozen reference")
    for name in FROZEN_ARTIFACTS:
        a = reference["artifacts"][name]
        b = candidate["artifacts"][name]
        reference_file = Path(a["path"])
        candidate_file = Path(b["path"])
        if not reference_file.is_absolute():
            reference_file = REPO_ROOT / reference_file
        if not candidate_file.is_absolute():
            candidate_file = REPO_ROOT / candidate_file
        if reference_file.resolve() == candidate_file.resolve():
            raise ValueError(f"{name} candidate aliases historical artifact")
        for item in (a, b):
            path = Path(item["path"])
            if not path.is_absolute():
                path = REPO_ROOT / path
            if not path.is_file() or sha256_file(path) != item["sha256"] or \
                    path.stat().st_size != item["bytes"]:
                raise ValueError(f"{name} artifact missing or does not match its manifest: {path}")
        if (a["sha256"], a["rows"]) != (b["sha256"], b["rows"]):
            raise ValueError(f"{name} differs from frozen reference")
    return {"reference": str(reference_path), "candidate": str(candidate_path),
            "source_revision": candidate["source"]["resolved_revision"],
            "matched_artifacts": len(FROZEN_ARTIFACTS),
            "train_rows": candidate["strict_split"]["splits"]["train_strict"]["rows_emitted"],
            "val_rows": candidate["strict_split"]["splits"]["val_strict"]["rows_emitted"],
            "test_rows": candidate["strict_split"]["splits"]["test_strict"]["rows_emitted"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_split_identity(args.reference, args.candidate), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
