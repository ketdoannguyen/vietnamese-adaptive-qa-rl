"""Run registry: every stage/sweep entry point appends a status record here.

Statuses are exactly: PLANNED, RUNNING, SUCCEEDED, FAILED, SKIPPED, BLOCKED, NOT_RUN.
"""

from __future__ import annotations

import datetime as _dt
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.utils.hashing import sha256_file
from src.utils.io import append_jsonl, atomic_write_json

STATUSES = {"PLANNED", "RUNNING", "SUCCEEDED", "FAILED", "SKIPPED", "BLOCKED", "NOT_RUN"}


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def git_sha(repo: str | Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def env_fingerprint() -> dict:
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "hostname": socket.gethostname(),
        "user": os.environ.get("USER", "unknown"),
    }


class RunRegistry:
    """Append-only JSONL registry at ``runs/index.jsonl`` (project-relative)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, *, run_id: str, stage: str, status: str, **extra: Any) -> dict:
        if status not in STATUSES:
            raise ValueError(f"invalid status {status!r}; must be one of {sorted(STATUSES)}")
        row = {
            "run_id": run_id,
            "stage": stage,
            "status": status,
            "ts": now_iso(),
            "env": env_fingerprint(),
        }
        row.update(extra)
        append_jsonl(self.path, row)
        return row

    def record_artifact(self, run_id: str, stage: str, artifacts: list[str | Path], **extra: Any) -> dict:
        """Record artifacts WITH checksums; a directory existing is never proof of success."""
        manifest = {}
        for a in artifacts:
            p = Path(a)
            if p.is_file():
                manifest[str(p)] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
            elif p.is_dir():
                manifest[str(p)] = {
                    "dir": True,
                    "files": sorted(x.name for x in p.iterdir() if x.is_file())[:200],
                }
            else:
                manifest[str(p)] = {"missing": True}
        return self.append(run_id=run_id, stage=stage, status="SUCCEEDED", artifacts=manifest, **extra)


def write_run_meta(run_dir: str | Path, meta: dict) -> None:
    atomic_write_json(Path(run_dir) / "run_meta.json", meta)
