"""Pinned download + raw persistence for ``taidng/UIT-ViQuAD2.0``.

The revision is resolved from the Hub (never hardcoded), the raw splits are written as
parquet under ``data/raw/<revision>/`` and everything written is checksummed into
``data/raw/source_manifest.json``. Row counts are recorded as measured — no expected
count is treated as an invariant.
"""

from __future__ import annotations

import datetime as _dt
import os
import tempfile
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import HfApi

from src.utils.hashing import sha256_file
from src.utils.io import atomic_write_json, read_json
from src.utils.reproducibility import capture_library_versions

MANIFEST_NAME = "source_manifest.json"
DEFAULT_REPO = "taidng/UIT-ViQuAD2.0"
SPLITS = ("train", "validation", "test")


def now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def resolve_revision(repo_id: str = DEFAULT_REPO) -> dict[str, Any]:
    """Resolve the dataset's current commit SHA and card metadata."""
    info = HfApi().dataset_info(repo_id=repo_id)
    card = info.card_data.to_dict() if info.card_data is not None else {}
    return {
        "repo_id": repo_id,
        "resolved_revision": info.sha,
        "last_modified": str(info.lastModified) if info.lastModified else None,
        "tags": list(info.tags or []),
        "card_license": card.get("license"),
        "card_pretty_name": card.get("pretty_name"),
        "files": sorted(s.rfilename for s in (info.siblings or [])),
        "license_verified": False,
        "license_note": (
            "The Hub dataset card exposes no license field; redistribution rights were "
            "NOT verified. Raw data stays local and is never committed or republished."
        ),
    }


def manifest_path(raw_dir: str | Path) -> Path:
    return Path(raw_dir) / MANIFEST_NAME


def load_source_manifest(raw_dir: str | Path) -> dict[str, Any]:
    path = manifest_path(raw_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python -m src.pipeline.download` first")
    return read_json(path)


def _is_reusable(manifest: dict[str, Any], repo_id: str, revision: str, raw_dir: Path) -> bool:
    if manifest.get("repo_id") != repo_id or manifest.get("resolved_revision") != revision:
        return False
    for entry in manifest.get("splits", {}).values():
        path = raw_dir / entry["parquet"]
        if not path.exists() or sha256_file(path) != entry["sha256"]:
            return False
    return True


def _split_stats(dataset: Any, name: str) -> dict[str, Any]:
    answers = dataset["answers"]
    flags = dataset["is_impossible"]
    return {
        "rows": dataset.num_rows,
        "impossible_rows": int(sum(1 for x in flags if x)),
        "answerable_rows": int(sum(1 for x in flags if not x)),
        "answers_field_none_rows": int(sum(1 for a in answers if a is None)),
        "features": {k: str(v) for k, v in dataset.features.items()},
    }


def persist_raw(
    raw_dir: str | Path,
    *,
    repo_id: str = DEFAULT_REPO,
    revision: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Download the pinned revision and persist it; idempotent on identical input.

    Re-running with the same resolved revision and intact files is a no-op; a new
    upstream commit produces a new ``<revision>`` directory and a manifest that records
    the previous revision it supersedes.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    info = resolve_revision(repo_id)
    resolved = revision if revision is not None else info["resolved_revision"]
    mpath = manifest_path(raw_dir)

    previous: dict[str, Any] = {}
    if mpath.exists():
        previous = read_json(mpath)
        if not force and _is_reusable(previous, repo_id, resolved, raw_dir):
            out = dict(previous)
            out["reused"] = True
            return out

    rev_dir = raw_dir / resolved
    rev_dir.mkdir(parents=True, exist_ok=True)
    splits: dict[str, Any] = {}
    for name in SPLITS:
        dataset = load_dataset(repo_id, revision=resolved, split=name)
        fd, tmp = tempfile.mkstemp(dir=str(rev_dir), suffix=".parquet.tmp")
        os.close(fd)
        try:
            dataset.to_parquet(tmp)
            target = rev_dir / f"{name}.parquet"
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        entry = _split_stats(dataset, name)
        entry["parquet"] = str(target.relative_to(raw_dir))
        entry["sha256"] = sha256_file(target)
        entry["bytes"] = target.stat().st_size
        splits[name] = entry
        del dataset

    history = list(previous.get("revision_history", []))
    if previous.get("resolved_revision") and previous["resolved_revision"] != resolved:
        history.append(previous["resolved_revision"])

    manifest: dict[str, Any] = {
        "repo_id": repo_id,
        **{k: info[k] for k in ("last_modified", "tags", "card_license", "card_pretty_name",
                                "files", "license_verified", "license_note")},
        "resolved_revision": resolved,
        "download_utc": now_utc(),
        "raw_dir": str(raw_dir),
        "splits": splits,
        "total_rows": sum(s["rows"] for s in splits.values()),
        "revision_history": history,
        "library_versions": capture_library_versions(),
        "reused": False,
    }
    atomic_write_json(mpath, manifest)
    return manifest


def raw_split_paths(raw_dir: str | Path, manifest: dict[str, Any] | None = None) -> dict[str, Path]:
    """Absolute paths of the persisted raw parquet files, per split."""
    raw_dir = Path(raw_dir)
    manifest = manifest if manifest is not None else load_source_manifest(raw_dir)
    return {name: raw_dir / entry["parquet"] for name, entry in manifest["splits"].items()
            if (raw_dir / entry["parquet"]).exists()}
