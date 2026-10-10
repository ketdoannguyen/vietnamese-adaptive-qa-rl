"""Run registry plus the P0 run-isolation primitives (output lock, GPU ledger).

``RunRegistry``: every stage/sweep entry point appends a status record here. Registry
statuses are: PLANNED, RUNNING, SUCCEEDED, FAILED, SKIPPED, BLOCKED, NOT_RUN,
INTERRUPTED, SUPERSEDED.

``OutputNamespaceLock``: one live ``run_all`` per output namespace, enforced with ``flock``.

``GpuLedger``: append-only ledger of stage allocation intervals; cumulative GPU seconds are
the union of the recorded intervals (overlaps are never double counted) and every malformed
row fails closed.
"""

from __future__ import annotations

import datetime as _dt
import fcntl
import json
import math
import os
import socket
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from src.utils.hashing import sha256_file
from src.utils.io import append_jsonl, atomic_write_json

STATUSES = {"PLANNED", "RUNNING", "SUCCEEDED", "FAILED", "SKIPPED", "BLOCKED",
            "NOT_RUN", "INTERRUPTED", "SUPERSEDED"}


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


# ------------------------------------------------------------------- output namespace lock


class LockHeld(RuntimeError):
    """Another live process owns the output namespace lock."""


class OutputNamespaceLock:
    """Exclusive cross-process lock over one output namespace, backed by ``flock``.

    ``acquire()`` raises :class:`LockHeld` while another live holder exists; the namespace
    stays locked until ``release()`` or until the holding process exits (the kernel drops
    the lock, even after a kill). Use as a context manager to release on any exit.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fd: int | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            raise LockHeld(f"lock already held for {self.path}: {exc}") from exc
        self._fd = fd

    def release(self) -> None:
        if self._fd is None:
            return
        fcntl.flock(self._fd, fcntl.LOCK_UN)
        os.close(self._fd)
        self._fd = None

    def __enter__(self) -> "OutputNamespaceLock":
        self.acquire()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.release()


# ----------------------------------------------------------------------------- GPU ledger


class GpuLedgerError(ValueError):
    """The GPU ledger contains a malformed row (accounting fails closed)."""


LEDGER_STATUSES = ("SUCCEEDED", "FAILED", "INTERRUPTED")


def _parse_ts(value: Any, where: str) -> _dt.datetime:
    if not isinstance(value, str):
        raise GpuLedgerError(f"{where}: timestamp must be an ISO-8601 string, "
                             f"got {type(value).__name__}")
    try:
        parsed = _dt.datetime.fromisoformat(value)
    except ValueError as exc:
        raise GpuLedgerError(f"{where}: invalid timestamp {value!r}: {exc}") from exc
    if parsed.tzinfo is None:
        raise GpuLedgerError(f"{where}: timestamp {value!r} has no UTC offset")
    return parsed


def _as_now(now: _dt.datetime | None) -> _dt.datetime:
    if now is None:
        return _dt.datetime.now(_dt.timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware (UTC offset required)")
    return now


def _finite_number(value: Any) -> float | None:
    """``float(value)`` for a finite real number; None for bools/NaN/inf/anything else."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _union_seconds(spans: list[tuple[_dt.datetime, _dt.datetime]]) -> float:
    """Wall-clock length of the union of ``spans`` (overlaps counted once)."""
    total = 0.0
    current_start: _dt.datetime | None = None
    current_end: _dt.datetime | None = None
    for span_start, span_end in sorted(spans):
        if current_end is None or span_start > current_end:
            if current_end is not None and current_start is not None:
                total += (current_end - current_start).total_seconds()
            current_start, current_end = span_start, span_end
        elif span_end > current_end:
            current_end = span_end
    if current_start is not None and current_end is not None:
        total += (current_end - current_start).total_seconds()
    return total


class GpuLedger:
    """Append-only JSONL ledger of stage allocation intervals.

    One ``open`` row per launch::

        {"event": "open", "interval_id": ..., "run_id": ..., "stage_key": ...,
         "stage": ..., "gpu": true, "started": <iso8601>}

    one ``close`` row per terminal status::

        {"event": "close", "interval_id": ..., "ended": <iso8601>,
         "status": "SUCCEEDED"|"FAILED"|"INTERRUPTED", "seconds": <float>}

    ``spent_seconds`` bills the union of GPU intervals: CPU-only intervals (``gpu=false``)
    spend nothing, failed/interrupted intervals still count, and an interval left open by a
    killed invocation is charged from ``started`` to ``now``. Malformed rows (bad JSON,
    close without open, duplicate open/close, invalid timestamps, negative/non-finite
    seconds, a close ``seconds`` that disagrees with its own ``started``/``ended``, unknown
    event/status) raise :class:`GpuLedgerError` from every read and make ``budget_ok``
    return False.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    # ----------------------------------------------------------------------- writes

    def open_interval(self, *, run_id: str, stage_key: str, stage: str | None = None,
                      gpu: bool = True, now: _dt.datetime | None = None) -> str:
        """Append one ``open`` row; returns the new ``interval_id``."""
        if not isinstance(gpu, bool):
            raise GpuLedgerError(f"gpu must be a bool, got {gpu!r}")
        started = _as_now(now)
        interval_id = f"{stage_key}-{run_id}-{uuid.uuid4().hex[:12]}"
        append_jsonl(self.path, {
            "event": "open", "interval_id": interval_id, "run_id": run_id,
            "stage_key": stage_key, "stage": stage, "gpu": gpu,
            "started": started.isoformat(),
        })
        return interval_id

    def close_interval(self, interval_id: str, *, status: str,
                       now: _dt.datetime | None = None) -> float:
        """Append one ``close`` row; returns the billed seconds (``now - started``)."""
        if status not in LEDGER_STATUSES:
            raise GpuLedgerError(f"invalid status {status!r}; "
                                 f"must be one of {list(LEDGER_STATUSES)}")
        ended = _as_now(now)
        opens, closes = self._fold()
        if interval_id not in opens:
            raise GpuLedgerError(f"close_interval: unknown interval_id {interval_id!r}")
        if interval_id in closes:
            raise GpuLedgerError(f"close_interval: interval {interval_id!r} already closed")
        where = f"{self.path}: open row of {interval_id!r}"
        started = _parse_ts(opens[interval_id]["started"], where)
        seconds = (ended - started).total_seconds()
        if seconds < 0:
            raise GpuLedgerError(f"close_interval: {interval_id!r} would end before it started")
        append_jsonl(self.path, {
            "event": "close", "interval_id": interval_id, "ended": ended.isoformat(),
            "status": status, "seconds": seconds,
        })
        return seconds

    # ------------------------------------------------------------------------ reads

    def _rows(self) -> list[dict]:
        if not self.path.is_file():
            return []
        try:
            text = self.path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise GpuLedgerError(f"{self.path}: not valid UTF-8: {exc}") from exc
        rows = []
        for line_no, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GpuLedgerError(f"{self.path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise GpuLedgerError(f"{self.path}:{line_no}: row is not a JSON object")
            rows.append(row)
        return rows

    def _fold(self) -> tuple[dict[str, dict], dict[str, dict]]:
        """``(opens, closes)`` keyed by interval_id; raises GpuLedgerError when malformed."""
        opens: dict[str, dict] = {}
        closes: dict[str, dict] = {}
        for line_no, row in enumerate(self._rows(), 1):
            where = f"{self.path}:{line_no}"
            event = row.get("event")
            interval_id = row.get("interval_id")
            if not isinstance(interval_id, str) or not interval_id:
                raise GpuLedgerError(f"{where}: missing interval_id")
            if event == "open":
                if interval_id in opens:
                    raise GpuLedgerError(f"{where}: duplicate open for {interval_id!r}")
                _parse_ts(row.get("started"), f"{where}: started")
                if "gpu" in row and not isinstance(row["gpu"], bool):
                    raise GpuLedgerError(f"{where}: gpu must be a bool")
                opens[interval_id] = row
            elif event == "close":
                if interval_id not in opens:
                    raise GpuLedgerError(f"{where}: close without open for {interval_id!r}")
                if interval_id in closes:
                    raise GpuLedgerError(f"{where}: duplicate close for {interval_id!r}")
                started = _parse_ts(opens[interval_id]["started"], f"{where}: started")
                ended = _parse_ts(row.get("ended"), f"{where}: ended")
                if ended < started:
                    raise GpuLedgerError(f"{where}: ended before started for {interval_id!r}")
                seconds = _finite_number(row.get("seconds"))
                if seconds is None or seconds < 0:
                    raise GpuLedgerError(
                        f"{where}: seconds must be a non-negative finite number")
                elapsed = (ended - started).total_seconds()
                if not math.isclose(seconds, elapsed, rel_tol=1e-9, abs_tol=1e-3):
                    raise GpuLedgerError(
                        f"{where}: seconds={seconds} does not match ended-started="
                        f"{elapsed} for {interval_id!r} (the close row is evidence; it "
                        f"must carry its own elapsed time)")
                if row.get("status") not in LEDGER_STATUSES:
                    raise GpuLedgerError(f"{where}: invalid status {row.get('status')!r}; "
                                         f"must be one of {list(LEDGER_STATUSES)}")
                closes[interval_id] = row
            else:
                raise GpuLedgerError(f"{where}: invalid event {event!r}")
        return opens, closes

    def open_intervals(self) -> list[dict]:
        """Open (never closed) interval rows, in append order."""
        opens, closes = self._fold()
        return [row for interval_id, row in opens.items() if interval_id not in closes]

    def spent_seconds(self, *, now: _dt.datetime | None = None) -> float:
        """Cumulative billed GPU seconds (union of GPU intervals) at ``now``."""
        current = _as_now(now)
        opens, closes = self._fold()
        spans: list[tuple[_dt.datetime, _dt.datetime]] = []
        for interval_id, open_row in opens.items():
            if not open_row.get("gpu", True):
                continue  # CPU-only interval: spends nothing
            started = _parse_ts(open_row["started"], f"{self.path}: started")
            close_row = closes.get(interval_id)
            if close_row is None:
                end = max(current, started)  # killed invocation: charge up to now
            else:
                end = _parse_ts(close_row["ended"], f"{self.path}: ended")
            spans.append((started, end))
        return _union_seconds(spans)

    def budget_ok(self, max_gpu_hours: float | None, *,
                  now: _dt.datetime | None = None) -> bool:
        """False once the cumulative spend reaches the limit; ``None``/0 disables the budget.

        A malformed ledger fails closed (False); so does an unusable limit (non-numeric,
        boolean, NaN, infinite, negative or too large to convert) — an unusable limit never
        silently disables the budget.
        """
        if max_gpu_hours is None:
            return True
        limit_hours = _finite_number(max_gpu_hours)
        if limit_hours is None or limit_hours < 0:
            return False
        if limit_hours == 0:
            return True  # budget disabled
        limit_seconds = limit_hours * 3600.0
        if not math.isfinite(limit_seconds):
            return False
        try:
            return self.spent_seconds(now=now) < limit_seconds
        except (GpuLedgerError, TypeError, ValueError):
            return False
