"""P0 TDD contracts for run isolation + cumulative GPU accounting (src/utils/runs.py).

The forthcoming implementation must provide:

``OutputNamespaceLock(path)``
    Exclusive cross-process lock for one output namespace. ``acquire()`` raises
    ``LockHeld`` while another live holder exists; ``release()`` frees the namespace; the
    object is a context manager that releases even when its body raises.

``GpuLedger(path)``
    Append-only JSONL ledger of stage allocations (run_all writes
    ``outputs/experiments/gpu_ledger.jsonl``). One ``open`` event per launch::

        {"event": "open", "interval_id": <id>, "run_id": <run_id>,
         "stage_key": <key>, "stage": <stage>, "gpu": true, "started": <iso8601>}

    and one ``close`` event per terminal status::

        {"event": "close", "interval_id": <id>, "ended": <iso8601>,
         "status": "SUCCEEDED"|"FAILED"|"INTERRUPTED", "seconds": <float>}

    ``spent_seconds(now=...)`` sums closed GPU intervals; a GPU interval left open (the
    invocation was killed) is charged from ``started`` to ``now`` — fail closed, never
    dropped. CPU-only intervals (``gpu=false``) spend nothing. Malformed ledgers (bad
    JSON, close without open, invalid timestamps, negative seconds) raise
    ``GpuLedgerError`` from ``spent_seconds`` and make ``budget_ok`` return False.
    ``budget_ok(max_gpu_hours, now=...)`` is False once the cumulative spend reaches the
    limit; ``None`` or 0 disables the budget.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.utils.runs import GpuLedger, GpuLedgerError, LockHeld, OutputNamespaceLock

REAL = Path(__file__).resolve().parents[1]

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def at(offset_seconds: float) -> datetime:
    """A deterministic timezone-aware instant ``offset_seconds`` after BASE."""
    return BASE + timedelta(seconds=offset_seconds)


# ------------------------------------------------------------------ output namespace lock

def test_registry_accepts_historical_terminal_statuses(tmp_path):
    from src.utils.runs import RunRegistry

    registry = RunRegistry(tmp_path / "runs.jsonl")
    for status in ("INTERRUPTED", "SUPERSEDED"):
        row = registry.append(run_id=status, stage="P6_rl", status=status)
        assert row["status"] == status


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


# ------------------------------------------------------------------------ GPU ledger

def test_spend_accumulates_across_intervals_and_invocations(tmp_path):
    path = tmp_path / "gpu_ledger.jsonl"
    first = GpuLedger(path)
    i1 = first.open_interval(run_id="run-a", stage_key="train_reader", stage="P3_reader",
                             now=at(0))
    assert first.close_interval(i1, status="SUCCEEDED", now=at(100)) == pytest.approx(
        100.0, abs=0.5)
    i2 = first.open_interval(run_id="run-a", stage_key="train_laya_ce",
                             stage="P5_laya_train", gpu=True, now=at(200))
    assert first.close_interval(i2, status="SUCCEEDED", now=at(350)) == pytest.approx(
        150.0, abs=0.5)
    # a fresh instance (the next invocation) sees the persisted allocation
    second = GpuLedger(path)
    assert second.spent_seconds(now=at(1000)) == pytest.approx(250.0, abs=0.5)
    assert second.budget_ok(1.0, now=at(1000)) is True   # 3600 s limit, 250 s spent
    assert second.budget_ok(0.05, now=at(1000)) is False  # 180 s limit, 250 s spent


def test_cpu_only_intervals_never_spend_gpu_time(tmp_path):
    path = tmp_path / "gpu_ledger.jsonl"
    ledger = GpuLedger(path)
    i = ledger.open_interval(run_id="run-a", stage_key="download", stage="P1_download",
                             gpu=False, now=at(0))
    ledger.close_interval(i, status="SUCCEEDED", now=at(999))
    assert ledger.spent_seconds(now=at(999)) == 0.0
    assert ledger.budget_ok(0.0001, now=at(999)) is True  # 0.36 s budget, nothing spent


def test_failed_and_interrupted_intervals_still_count(tmp_path):
    path = tmp_path / "gpu_ledger.jsonl"
    ledger = GpuLedger(path)
    failed = ledger.open_interval(run_id="run-a", stage_key="train_reader", now=at(0))
    ledger.close_interval(failed, status="FAILED", now=at(30))
    interrupted = ledger.open_interval(run_id="run-b", stage_key="train_laya_ce", now=at(100))
    ledger.close_interval(interrupted, status="INTERRUPTED", now=at(180))
    cpu_failed = ledger.open_interval(run_id="run-b", stage_key="report", gpu=False,
                                      now=at(100))
    ledger.close_interval(cpu_failed, status="FAILED", now=at(900))
    assert ledger.spent_seconds(now=at(1000)) == pytest.approx(110.0, abs=1.0)


def test_open_interval_counts_elapsed_time_fail_closed(tmp_path):
    path = tmp_path / "gpu_ledger.jsonl"
    ledger = GpuLedger(path)
    ledger.open_interval(run_id="crashed-run", stage_key="train_laya_ce",
                         stage="P5_laya_train", now=at(0))
    # the invocation was killed and never closed it; the elapsed time must be charged
    assert ledger.spent_seconds(now=at(120)) == pytest.approx(120.0, abs=1.0)
    assert ledger.budget_ok(60 / 3600, now=at(120)) is False  # 60 s limit, 120 s charged
    assert ledger.budget_ok(1.0, now=at(120)) is True


def test_budget_is_exhausted_at_the_limit_and_can_be_disabled(tmp_path):
    path = tmp_path / "gpu_ledger.jsonl"
    ledger = GpuLedger(path)
    i = ledger.open_interval(run_id="run-a", stage_key="train_reader", now=at(0))
    ledger.close_interval(i, status="SUCCEEDED", now=at(3600))
    assert ledger.spent_seconds(now=at(3600)) == pytest.approx(3600.0, abs=1.0)
    assert ledger.budget_ok(1.0, now=at(3600)) is False      # exactly at the limit
    assert ledger.budget_ok(3.0, now=at(3600)) is True
    assert ledger.budget_ok(None, now=at(3600)) is True      # budget disabled
    assert ledger.budget_ok(0, now=at(3600)) is True


@pytest.mark.parametrize("rows", [
    pytest.param(["{not json"], id="garbage-line"),
    pytest.param([json.dumps({"event": "close", "interval_id": "ghost",
                              "ended": "2026-01-01T00:01:40+00:00",
                              "status": "FAILED", "seconds": 100.0})],
                 id="close-without-open"),
    pytest.param([json.dumps({"event": "open", "interval_id": "i1", "run_id": "r1",
                              "stage_key": "train_reader", "gpu": True,
                              "started": "not-a-time"}),
                  json.dumps({"event": "close", "interval_id": "i1",
                              "ended": "2026-01-01T00:01:40+00:00",
                              "status": "SUCCEEDED", "seconds": 100.0})],
                 id="invalid-timestamp"),
    pytest.param([json.dumps({"event": "open", "interval_id": "i1", "run_id": "r1",
                              "stage_key": "train_reader", "gpu": True,
                              "started": "2026-01-01T00:00:00+00:00"}),
                  json.dumps({"event": "close", "interval_id": "i1",
                              "ended": "2026-01-01T00:01:40+00:00",
                              "status": "FAILED", "seconds": -1.0})],
                 id="negative-seconds"),
])
def test_malformed_ledger_fails_closed(tmp_path, rows):
    path = tmp_path / "gpu_ledger.jsonl"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    ledger = GpuLedger(path)
    with pytest.raises(GpuLedgerError):
        ledger.spent_seconds(now=at(200))
    assert ledger.budget_ok(12, now=at(200)) is False


@pytest.mark.parametrize("seconds", [
    pytest.param(1.0, id="understated"),
    pytest.param(1000.0, id="overstated"),
])
def test_close_seconds_must_equal_the_interval_length(tmp_path, seconds):
    """A close row's ``seconds`` is evidence: it must match ended-started, not be tampered."""
    path = tmp_path / "gpu_ledger.jsonl"
    path.write_text("\n".join([
        json.dumps({"event": "open", "interval_id": "i1", "run_id": "r1",
                    "stage_key": "train_reader", "stage": "P3_reader", "gpu": True,
                    "started": "2026-01-01T00:00:00+00:00"}),
        # 100 s really elapsed; the row claims otherwise.
        json.dumps({"event": "close", "interval_id": "i1",
                    "ended": "2026-01-01T00:01:40+00:00", "status": "FAILED",
                    "seconds": seconds}),
    ]) + "\n", encoding="utf-8")
    ledger = GpuLedger(path)
    with pytest.raises(GpuLedgerError):
        ledger.spent_seconds(now=at(200))
    assert ledger.budget_ok(12, now=at(200)) is False


@pytest.mark.parametrize("limit", [
    pytest.param(-1.0, id="negative"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="infinite"),
    pytest.param(10 ** 400, id="too-large-to-convert"),
    pytest.param("12h", id="non-numeric"),
])
def test_unusable_budget_limits_fail_closed(tmp_path, limit):
    """An unusable limit never silently disables the budget: budget_ok returns False."""
    ledger = GpuLedger(tmp_path / "gpu_ledger.jsonl")
    assert ledger.budget_ok(limit, now=at(0)) is False
