"""Freeze the deterministic, disjoint VixSTORY pilot split manifest (issue #30).

Locks the pilot protocol *before* any GPU run:

* ``pilot_300``  -- 300 official-train QIDs, stratified by ``hop_level``/``question_type``,
  seed 42.
* ``train_dev_200`` -- 200 further official-train QIDs drawn from the remainder (the pilot
  QIDs are excluded), same stratification, seed 43; used only to tune prompt/reader/checkpoint.
* ``validation_246`` -- included only with explicit ``--include-validation``; the default
  manifest is train/dev-only and does not download, read or hash validation.

The manifest stores QIDs (sorted), the dataset revision, source-file sha256, sampling
versions, per-split hashes and a disjointness proof -- never a question, answer or passage.
``data/test.parquet`` is never read. Selection reuses
``scripts.vixstory_suitability.sample_train`` so the split is reproducible from the pinned
revision alone.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from scripts.vixstory_suitability import REPO, sample_train
from src.utils.hashing import sha256_file, sha256_json
from src.utils.io import atomic_write_json

REVISION = "19613082c7ef128647d561472d1e150ca878fb22"
FILES = ("data/train.parquet", "data/validation.parquet")
EXPECTED_SHA256 = {
    "data/train.parquet": "7e0f41ce2896a66bc5ea4e4150303508fe8aa2787db6800c118e8830eb307dad",
    "data/validation.parquet": "db356de42e18b1b4f3a9150852702c406188bbf95a2b1524b52084f310ca6ab7",
}
TRAIN_COUNT = 2553
VALIDATION_COUNT = 246
PILOT_N = 300
DEV_N = 200
PILOT_SEED = 42
DEV_SEED = 43
SCHEMA_VERSION = 1
SAMPLING_VERSION = "sample_train.v1"
RUN_ID = "vixstory-pilot-300-2026-10-11"
OUT = Path("outputs/experiments") / RUN_ID / "manifest.json"


def select_pilot(train_rows: Sequence[Mapping], *, pilot_n: int = PILOT_N,
                 pilot_seed: int = PILOT_SEED, dev_n: int = DEV_N,
                 dev_seed: int = DEV_SEED) -> tuple[list[Mapping], list[Mapping]]:
    """Deterministic, disjoint ``(pilot, train_dev)`` split of the official train rows."""
    if len({row["id"] for row in train_rows}) != len(train_rows):
        raise ValueError("duplicate QA ids in official train")
    pilot = sample_train(train_rows, n=pilot_n, seed=pilot_seed)
    pilot_ids = {row["id"] for row in pilot}
    remaining = [row for row in train_rows if row["id"] not in pilot_ids]
    dev = sample_train(remaining, n=dev_n, seed=dev_seed)
    overlap = pilot_ids & {row["id"] for row in dev}
    if overlap:
        raise ValueError(f"pilot and train_dev overlap on {len(overlap)} QIDs")
    return pilot, dev


def split_record(rows: Sequence[Mapping]) -> dict:
    """Sorted QIDs, their canonical hash and the hop/type strata actually drawn."""
    qids = sorted(str(row["id"]) for row in rows)
    return {
        "n": len(qids),
        "qids": qids,
        "qids_sha256": sha256_json(qids),
        "strata": {
            "hop_level": {str(k): v for k, v in
                          sorted(Counter(int(row["hop_level"]) for row in rows).items())},
            "question_type": {str(k): v for k, v in
                              sorted(Counter(str(row["question_type"]) for row in rows).items())},
        },
    }


def build_manifest(*, revision: str, source_files: Mapping[str, Mapping],
                   raw_rows: Mapping[str, int], pilot_rows: Sequence[Mapping],
                   dev_rows: Sequence[Mapping], val_rows: Sequence[Mapping]) -> dict:
    """Assemble the manifest dict; raise if any split is empty or the three overlap."""
    pilot, dev, val = (split_record(rows) for rows in (pilot_rows, dev_rows, val_rows))
    if not pilot["n"] or not dev["n"] or not val["n"]:
        raise ValueError("every pilot split must be non-empty")
    overlaps = {
        "pilot_vs_train_dev": len(set(pilot["qids"]) & set(dev["qids"])),
        "pilot_vs_validation": len(set(pilot["qids"]) & set(val["qids"])),
        "train_dev_vs_validation": len(set(dev["qids"]) & set(val["qids"])),
    }
    if any(overlaps.values()):
        raise ValueError(f"pilot splits are not disjoint: {overlaps}")
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": 30,
        "run_id": RUN_ID,
        "purpose": "Freeze the VixSTORY pilot split (300 train + 200 disjoint train-dev, 246 "
                   "validation) before any GPU run. Selection only; no answers or passages.",
        "source": {
            "repo_id": REPO,
            "resolved_revision": revision,
            "files": dict(source_files),
            "raw_rows": {"train": raw_rows["train"], "validation": raw_rows["validation"]},
            "official_test_loaded": False,
            "license": "CC BY-NC-SA 4.0; research use only",
        },
        "sampling": {
            "sampler": "scripts.vixstory_suitability.sample_train",
            "version": SAMPLING_VERSION,
            "stratify_by": ["hop_level", "question_type"],
            "pilot": {"n": pilot["n"], "seed": PILOT_SEED},
            "train_dev": {"n": dev["n"], "seed": DEV_SEED,
                          "pool": "official train minus pilot QIDs"},
        },
        "splits": {"pilot_300": pilot, "train_dev_200": dev, "validation_246": val},
        "disjointness": {**overlaps, "all_disjoint": True},
        "selection_exploratory": True,
        "exploratory_reason": (
            "Earlier VixSTORY probing already scored 500 official-train QIDs (the seed-42 "
            "sample of scripts/vixstory_suitability.py); pilot_300 and train_dev_200 overlap "
            "that seen set, so these train questions are not held out. The 246 "
            "official-validation QIDs are scored once after selection is locked and must not "
            "be reported as an independent estimate if reader or hyperparameters are tuned on "
            "these train QIDs."
        ),
    }


def source_hashes(paths: Mapping[str, Path], *, expected: Mapping[str, str] | None = None) -> dict:
    """{name: {sha256, bytes}} for supplied files; refuse an unexpected digest."""
    expected = EXPECTED_SHA256 if expected is None else expected
    hashes = {}
    for name in paths:
        digest = sha256_file(paths[name])
        if name in expected and digest != expected[name]:
            raise ValueError(f"{name}: sha256 {digest} != pinned {expected[name]}")
        hashes[name] = {"sha256": digest, "bytes": Path(paths[name]).stat().st_size}
    return hashes


def run(*, out: Path = OUT, revision: str = REVISION,
        include_validation: bool = False) -> dict:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite pilot manifest {out}")
    files = FILES if include_validation else FILES[:1]
    paths = {name: Path(hf_hub_download(REPO, name, repo_type="dataset", revision=revision))
             for name in files}
    hashes = source_hashes(paths)
    train = pq.read_table(paths["data/train.parquet"],
                          columns=["id", "hop_level", "question_type"]).to_pylist()
    if len(train) != TRAIN_COUNT:
        raise ValueError(f"pinned train row count changed: {len(train)}/{TRAIN_COUNT}")
    pilot, dev = select_pilot(train)
    if include_validation:
        val = pq.read_table(paths["data/validation.parquet"],
                            columns=["id", "hop_level", "question_type"]).to_pylist()
        if len(val) != VALIDATION_COUNT:
            raise ValueError(f"pinned validation row count changed: {len(val)}/{VALIDATION_COUNT}")
        manifest = build_manifest(revision=revision, source_files=hashes,
                                  raw_rows={"train": len(train), "validation": len(val)},
                                  pilot_rows=pilot, dev_rows=dev, val_rows=val)
    else:
        p, d = split_record(pilot), split_record(dev)
        if set(p["qids"]) & set(d["qids"]):
            raise ValueError("pilot and train-dev QIDs overlap")
        manifest = {
            "schema_version": SCHEMA_VERSION, "run_id": RUN_ID,
            "source": {"repo_id": REPO, "resolved_revision": revision, "files": hashes,
                       "raw_rows": {"train": len(train)}, "official_test_loaded": False},
            "sampling": {"sampler": "scripts.vixstory_suitability.sample_train",
                         "version": SAMPLING_VERSION,
                         "stratify_by": ["hop_level", "question_type"],
                         "pilot": {"n": p["n"], "seed": PILOT_SEED},
                         "train_dev": {"n": d["n"], "seed": DEV_SEED,
                                       "pool": "official train minus pilot QIDs"}},
            "splits": {"pilot_300": p, "train_dev_200": d},
            "disjointness": {"pilot_vs_train_dev": 0, "all_disjoint": True},
        }
    atomic_write_json(out, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--include-validation", action="store_true",
                        help="explicitly include validation IDs and strata (never answers)")
    args = parser.parse_args()
    manifest = run(out=args.out, include_validation=args.include_validation)
    print(json.dumps({"run_id": manifest["run_id"],
                      "revision": manifest["source"]["resolved_revision"],
                      "disjointness": manifest["disjointness"],
                      "qids_sha256": {name: split["qids_sha256"]
                                      for name, split in manifest["splits"].items()}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
