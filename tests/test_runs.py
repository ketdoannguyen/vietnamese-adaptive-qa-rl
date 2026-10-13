"""Contracts for run isolation and provenance stamps (``src/utils/runs.py``).

``OutputNamespaceLock`` must be exclusive per namespace — across threads of one process and
across live processes — and must release on any exit. ``now_iso``/``git_sha`` are the stamps
the VixStory run manifests carry, so their shape is pinned here.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.utils.runs import LockHeld, OutputNamespaceLock, git_sha, now_iso

REAL = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ output namespace lock

def test_lock_is_exclusive_then_released_for_the_next_owner(tmp_path):
    path = tmp_path / "outputs" / ".run_all.lock"
    first = OutputNamespaceLock(path)
    first.acquire()
    second = OutputNamespaceLock(path)
    with pytest.raises(LockHeld):
        second.acquire()
    first.release()
    second.acquire()
    second.release()


def test_lock_is_released_by_the_context_manager_on_exception(tmp_path):
    path = tmp_path / "outputs" / ".run_all.lock"
    with pytest.raises(RuntimeError):
        with OutputNamespaceLock(path):
            raise RuntimeError("stage blew up")
    other = OutputNamespaceLock(path)
    other.acquire()
    other.release()


def test_lock_refuses_a_second_live_process(tmp_path):
    path = tmp_path / "outputs" / ".run_all.lock"
    holder = OutputNamespaceLock(path)
    holder.acquire()
    try:
        code = (
            "import sys\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "from src.utils.runs import OutputNamespaceLock, LockHeld\n"
            "try:\n"
            "    OutputNamespaceLock(sys.argv[2]).acquire()\n"
            "except LockHeld:\n"
            "    raise SystemExit(42)\n"
            "raise SystemExit(0)\n"
        )
        proc = subprocess.run([sys.executable, "-c", code, str(REAL), str(path)],
                              capture_output=True, text=True, timeout=60)
        assert proc.returncode == 42, proc.stderr
    finally:
        holder.release()


# ----------------------------------------------------------------------- provenance stamps

def test_now_iso_is_a_utc_iso8601_timestamp_at_second_precision():
    stamp = now_iso()
    parsed = datetime.fromisoformat(stamp)
    assert parsed.tzinfo is not None and parsed.utcoffset() == timedelta(0)
    assert "." not in stamp  # timespec="seconds": no fractional part


def test_git_sha_reads_head_and_returns_none_outside_a_repository(tmp_path):
    expected = subprocess.run(["git", "-C", str(REAL), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    assert expected and git_sha(REAL) == expected
    assert git_sha(tmp_path) is None
