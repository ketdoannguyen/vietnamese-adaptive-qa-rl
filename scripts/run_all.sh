#!/usr/bin/env bash
# =============================================================================
# scripts/run_all.sh — P0..P9 orchestrator (spec 06 §1-§6).
#
#   bash scripts/run_all.sh --profile pilot|medium|full [--dry-run] [--resume] \
#        [--config <overlay.yaml>] [--seed N] [--stop-after <stage_key>] [--zip]
#
# Stage DAG, in execution order:
#   download -> audit_split -> build_retrieval -> train_reader -> cache_reader
#   -> make_teacher -> train_laya --loss soft-ce -> train_laya --loss rlcd
#   -> cache_laya -> train_rl -> train_correctness (B5) -> eval_baselines (val)
#   -> eval_baselines (test; only after outputs/manifests/best_selection.json exists)
#   -> report
#
# Gating (spec 06 §4, adapted — see "deviations"):
#   a stage is SKIPPED only when (a) its latest registry row for this profile is
#   SUCCEEDED, (b) every artifact recorded in that row still matches its sha256,
#   (c) a stage manifest written by a previous run_all carries the same resolved
#   config sha, the same registry run_id, and its recorded outputs still match.
#   No stage writes stage_manifest.json today, so run_all writes it
#   (outputs/manifests/run_all/<stage_key>.json) after each successful stage.
#   Any mismatch re-runs the stage AND every stage downstream of it.
#
# Resume (spec 06 §5): a previous RUNNING row for this profile whose process is no
#   longer alive aborts the run with exit 3; with --resume those rows are marked
#   INTERRUPTED (recovery event recorded) and the DAG continues from the last
#   complete stage. --resume also closes each GPU interval a killed invocation left open,
#   conservatively and fail-closed: only when pgrep definitively proves the stage module
#   no longer runs (exit 1), and the interval is closed at the recovery instant — an upper
#   bound on the real allocation end that also charges the idle downtime between the kill
#   and the recovery. That overbilling is deliberate (never underbill); a precise
#   device-time correction needs a new mechanism (e.g. device-side accounting), not a
#   hand-edited ledger. Intervals that cannot be verified (live process, pgrep missing or
#   exiting with a status other than 0/1, unknown owner/stage) stay open and keep failing
#   closed. INTERRUPTED is recorded through the standard RunRegistry validator.
#
# Retries (spec 06 §6): a stage that dies with an OOM signature is retried up to
#   limits.max_retries_per_stage times with a generated config overlay that shrinks
#   the stage's own resource knobs; every retry logs the old -> new config diff. A retry
#   is a fresh GPU launch, so the fail-closed GPU budget gate runs again for it over the
#   ledger that already includes the attempt that just failed.
#
# Overlay forwarding: every stage child is launched with --config carrying the CLI base
#   overlay merged under the stage's own overlay (a previous retry's reduced-resource
#   knobs win on conflicts), so children resolve the same experiment namespace
#   (paths.runs/models/...) the orchestrator gated against; the stage manifest records
#   that merged overlay and its resolved config sha. With no --config the merged overlay
#   is empty and the launch is byte-identical to a plain profile launch.
#
# Isolation & GPU budget (P0): a real invocation takes an exclusive flock over
#   outputs/experiments/.run_all.lock before writing anything. It is fixed outside all
#   overlay-controlled paths: partial overlays cannot split mutual exclusion or GPU
#   accounting on this one L4. The lock is held by main for the whole invocation, and
#   stage children never inherit its fd. A second live invocation exits 3 and writes
#   nothing, while `--dry-run` needs no lock and stays strictly read-only. Every stage
#   launch is billed in outputs/experiments/gpu_ledger.jsonl (CPU-only stages record
#   gpu=false and spend nothing); before a GPU stage run_all fails closed when
#   the ledger is malformed, when an interval from a killed invocation is still open, or
#   when the cumulative spend has reached limits.max_gpu_hours.
#
# `--dry-run` prints the plan, the measured RAM/disk/GPU facts and activates nothing.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
  echo "run_all.sh: python interpreter not found: $PYTHON (set PYTHON=/path/to/python)" >&2
  exit 3
fi
if [ ! -f "$REPO_ROOT/src/utils/config.py" ]; then
  echo "run_all.sh: $REPO_ROOT does not look like the repository root" >&2
  exit 3
fi

export RUN_ALL_REPO="$REPO_ROOT"
export RUN_ALL_PYTHON="$PYTHON"

"$PYTHON" - "$@" <<'PY'
"""Orchestrator core. Runs with the repository's interpreter; imports only public helpers."""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(os.environ["RUN_ALL_REPO"])
PYTHON = os.environ["RUN_ALL_PYTHON"]
sys.path.insert(0, str(REPO))

from src.utils.config import _deep_merge, load_config, resolve, snapshot_config  # noqa: E402
from src.utils.hashing import sha256_file, sha256_json, sha256_text  # noqa: E402
from src.utils.io import atomic_write_json, read_yaml  # noqa: E402
from src.utils.runs import (  # noqa: E402
    GpuLedger, GpuLedgerError, LockHeld, OutputNamespaceLock, RunRegistry, git_sha, now_iso,
)

ORCH_STAGE = "run_all"
TERMINAL = ("SUCCEEDED", "FAILED", "SKIPPED", "BLOCKED", "NOT_RUN", "INTERRUPTED")
OOM_SIGNATURES = ("OutOfMemoryError", "out of memory", "OutOfMemory", "MemoryError",
                  "Cannot allocate memory", "std::bad_alloc")
FREEZE_FILE = "best_selection.json"
PROFILES = ("smoke", "small", "pilot", "medium", "full")


# ------------------------------------------------------------------------------- DAG

def stage_specs(cfg: dict) -> list[dict]:
    """The DAG for this config. ``row`` = exact registry-key filters for the gate.

    ``gpu`` explicitly classifies each stage for the ledger/budget (P0): True when the
    stage loads or trains neural models on the GPU (dense E5 encoder, reader, Laya) or
    re-runs that training path; False when it only moves files, works over recorded
    caches/registries (state cache, Laya prob cache) or writes reports.
    """
    frozen = ((cfg.get("laya") or {}).get("frozen") or "rlcd")
    regime = None if frozen in (None, "none") else str(frozen)
    return [
        dict(key="download", stage="P1_download", module="src.pipeline.download",
             args=[], deps=[], gpu=False),
        dict(key="audit_split", stage="P1_audit_split", module="src.pipeline.audit_split",
             args=[], deps=["download"], gpu=False),
        dict(key="build_retrieval", stage="P2_retrieval",
             module="src.pipeline.build_retrieval", args=[], deps=["audit_split"],
             gpu=True),
        dict(key="train_reader", stage="P3_reader", module="src.pipeline.train_reader",
             args=[], deps=["audit_split"], gpu=True),
        dict(key="cache_reader", stage="P3_reader_cache", module="src.pipeline.cache_reader",
             args=[], deps=["train_reader", "build_retrieval"], gpu=True,
             row={"splits": ["train_strict", "val_strict"]}),
        dict(key="make_teacher", stage="P5_teacher", module="src.pipeline.make_teacher",
             args=[], deps=["cache_reader", "build_retrieval"], gpu=False),
        dict(key="train_laya_ce", stage="P5_laya_train", module="src.pipeline.train_laya",
             args=["--loss", "soft-ce"], deps=["make_teacher"], row={"loss": "soft-ce"},
             gpu=True),
        dict(key="train_laya_rlcd", stage="P5_laya_train", module="src.pipeline.train_laya",
             args=["--loss", "rlcd"], deps=["make_teacher"], row={"loss": "rlcd"},
             gpu=True),
        dict(key="cache_laya", stage="P5_cache_laya", module="src.pipeline.cache_laya",
             args=[], deps=["train_laya_rlcd"], row={"regime": regime}, gpu=True),
        # train_rl pre-warms Laya; a probability-cache miss calls the model on the GPU,
        # so the launch is billed and budget-guarded like any other GPU stage.
        dict(key="train_rl", stage="P6_rl", module="src.pipeline.train_rl",
             args=[], deps=["cache_laya", "cache_reader"], gpu=True),
        dict(key="select_rl", stage="P6_select", module="src.pipeline.select_rl",
             args=[], deps=["train_rl"], gpu=False),
        dict(key="train_correctness", stage="B5_correctness",
             module="src.pipeline.train_correctness", args=[],
             deps=["cache_reader", "build_retrieval"], optional=True, gpu=True),
        dict(key="eval_baselines_val", stage="P7_baselines",
             module="src.pipeline.eval_baselines", args=["--split", "val_strict"],
             deps=["select_rl", "train_correctness", "cache_laya", "build_retrieval",
                   "train_laya_ce"],
             row={"split": "val_strict"}, gpu=True),
        dict(key="cache_reader_test", stage="P3_reader_cache",
             module="src.pipeline.cache_reader", args=["--splits", "test_strict"],
             deps=["select_rl"], row={"splits": ["test_strict"]}, gpu=True),
        dict(key="eval_baselines_test", stage="P7_baselines",
             module="src.pipeline.eval_baselines", args=["--split", "test_strict"],
             deps=["eval_baselines_val", "cache_reader_test"], row={"split": "test_strict"},
             needs_freeze=True, gpu=True),
        dict(key="report", stage="P9_report", module="src.pipeline.report", args=[],
             deps=["download", "audit_split", "build_retrieval", "train_reader",
                   "cache_reader", "make_teacher", "train_laya_ce", "train_laya_rlcd",
                   "cache_laya", "train_rl", "select_rl", "train_correctness",
                   "cache_reader_test", "eval_baselines_val", "eval_baselines_test"],
             gpu=False),
    ]


# ------------------------------------------------------------------------- utilities

def read_registry(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def short(sha: str | None) -> str:
    return (sha or "-")[:8]


def resolved_config_sha(profile: str, overlay: dict) -> tuple[dict, str]:
    """Resolved config and its sha, computed exactly as src.utils.config.snapshot_config."""
    cfg = load_config(profile, extra=overlay or None)
    text = json.dumps(cfg, ensure_ascii=False, sort_keys=True, indent=2)
    return cfg, sha256_text(text)


def row_matches(row: dict, spec: dict, profile: str) -> bool:
    if row.get("stage") != spec["stage"]:
        return False
    if "profile" in row and row["profile"] != profile:
        return False
    for key, value in (spec.get("row") or {}).items():
        if row.get(key) != value:
            return False
    return True


def latest_rows(rows: list[dict], spec: dict, profile: str) -> list[dict]:
    """Rows of the newest run_id for this stage (append order preserved)."""
    matched = [r for r in rows if row_matches(r, spec, profile)]
    if not matched:
        return []
    newest = matched[-1]["run_id"]
    return [r for r in matched if r.get("run_id") == newest]


def stage_status(rows: list[dict], spec: dict, profile: str) -> tuple[str | None, dict | None]:
    group = latest_rows(rows, spec, profile)
    if not group:
        return None, None
    last = group[-1]
    if last.get("status") == "RUNNING":
        return "RUNNING", last
    return last.get("status"), last


def artifacts_ok(artifacts: dict) -> tuple[bool, str]:
    """Every recorded artifact must still match: files by sha256+size, dirs by presence."""
    for path, meta in (artifacts or {}).items():
        p = Path(path)
        if not isinstance(meta, dict):
            return False, f"{path}: mục artifact không hợp lệ"
        if meta.get("missing"):
            return False, f"{path}: đã bị ghi nhận missing khi chạy"
        if meta.get("dir"):
            if not p.is_dir():
                return False, f"{path}: thư mục đã mất"
            missing = [name for name in meta.get("files") or [] if not (p / name).is_file()]
            if missing:
                return False, f"{path}: thiếu {missing[:3]}"
            continue
        if not p.is_file():
            return False, f"{path}: tệp đã mất"
        if sha256_file(p) != meta.get("sha256"):
            return False, f"{path}: sha256 đã đổi"
        if "bytes" in meta and p.stat().st_size != meta["bytes"]:
            return False, f"{path}: kích thước đã đổi"
    return True, ""


def manifest_outputs_ok(manifest: dict) -> tuple[bool, str]:
    outputs = manifest.get("outputs_sha") or {}
    for rel, meta in outputs.items():
        p = REPO / rel
        if meta.get("dir"):
            if not p.is_dir():
                return False, f"{rel}: thư mục đã mất"
            missing = [n for n in meta.get("files") or [] if not (p / n).is_file()]
            if missing:
                return False, f"{rel}: thiếu {missing[:3]}"
        elif not p.is_file():
            return False, f"{rel}: tệp đã mất"
        elif sha256_file(p) != meta.get("sha256"):
            return False, f"{rel}: sha256 đã đổi"
    return True, ""


def gate(spec: dict, profile: str, rows: list[dict], manifests: dict,
         base_sha: str) -> tuple[str, str, dict]:
    """Verdict RUN|SKIP for one stage plus reason and the overlay it should reuse."""
    status, row = stage_status(rows, spec, profile)
    if status is None:
        return "RUN", "chưa từng chạy cho profile này", {}
    if status == "RUNNING":
        return "RUN", f"lần chạy trước còn RUNNING ({row.get('run_id')})", {}
    if status != "SUCCEEDED":
        extra = f" ({row.get('error') or row.get('reason')})" if (row.get("error") or row.get("reason")) else ""
        return "RUN", f"lần chạy trước: {status}{extra}", {}
    ok, why = artifacts_ok(row.get("artifacts") or {})
    if not ok:
        return "RUN", f"artifact không khớp: {why}", {}
    manifest = manifests.get(spec["key"])
    if manifest is None:
        return "RUN", "chưa có stage_manifest (outputs/manifests/run_all/%s.json)" % spec["key"], {}
    if manifest.get("base_config_sha") != base_sha:
        return "RUN", "config --config hoặc --seed đã đổi kể từ stage_manifest", {}
    overlay = manifest.get("overlay") or {}
    _, cfg_sha = resolved_config_sha(profile, overlay)
    if manifest.get("config_sha") != cfg_sha:
        return "RUN", f"config đã đổi {short(manifest.get('config_sha'))}→{short(cfg_sha)}", {}
    if manifest.get("registry_run_id") != row.get("run_id"):
        return "RUN", "có registry row mới hơn stage_manifest", {}
    mok, mwhy = manifest_outputs_ok(manifest)
    if not mok:
        return "RUN", f"output của stage_manifest không khớp: {mwhy}", {}
    note = f"đã xong {row.get('run_id')}"
    if cfg_sha != base_sha:
        note += " (đang dùng overlay giảm tài nguyên của lần retry trước)"
    return "SKIP", note, overlay


def dependents(specs: list[dict]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {spec["key"]: [] for spec in specs}
    for spec in specs:
        for dep in spec["deps"]:
            if dep in out:
                out[dep].append(spec["key"])
    return out


def transitive(start: str, edges: dict[str, list[str]]) -> set[str]:
    seen: set[str] = set()
    stack = list(edges.get(start, []))
    while stack:
        key = stack.pop()
        if key in seen:
            continue
        seen.add(key)
        stack.extend(edges.get(key, []))
    return seen


def compute_plan(specs: list[dict], profile: str, rows: list[dict], manifests: dict,
                 base_sha: str) -> dict[str, tuple[str, str, dict]]:
    plan: dict[str, tuple[str, str, dict]] = {}
    for spec in specs:
        plan[spec["key"]] = gate(spec, profile, rows, manifests, base_sha)
    edges = dependents(specs)
    by_key = {spec["key"]: spec for spec in specs}
    forced: dict[str, str] = {}
    for spec in specs:
        verdict, _, _ = plan[spec["key"]]
        if verdict == "RUN":
            for down in transitive(spec["key"], edges):
                forced.setdefault(down, spec["key"])
    for key, dep in forced.items():
        verdict, reason, overlay = plan[key]
        if verdict == "SKIP":
            plan[key] = ("RUN", f"phụ thuộc {dep} sẽ chạy (invalidate downstream)", overlay)
    return plan


# -------------------------------------------------------------------------- resources

def resource_report() -> list[str]:
    lines = []
    usage = shutil.disk_usage(REPO)
    gib = 1024 ** 3
    lines.append(f"  đĩa   : tổng {usage.total / gib:.1f} GiB, còn trống {usage.free / gib:.1f} GiB "
                 f"({REPO})")
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        values = {}
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] in ("MemTotal:", "MemAvailable:"):
                values[parts[0]] = int(parts[1]) / 1024 ** 2  # kB -> GiB
        lines.append(f"  RAM   : tổng {values.get('MemTotal:', float('nan')):.1f} GiB, "
                     f"khả dụng {values.get('MemAvailable:', float('nan')):.1f} GiB")
    else:
        lines.append("  RAM   : không đo được (/proc/meminfo không có)")
    if shutil.which("nvidia-smi"):
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=False)
        text = (proc.stdout or proc.stderr or "").strip()
        gpu_line = f"  GPU   : {text}" if proc.returncode == 0 and text else \
            f"  GPU   : nvidia-smi lỗi rc={proc.returncode} — không đo được VRAM"
    else:
        gpu_line = "  GPU   : không có nvidia-smi — không đo được VRAM (chưa đo, không ước lượng)"
    lines.append(gpu_line)
    return lines


# -------------------------------------------------------------------------- execution

def module_process_state(module: str) -> str:
    """Three-state oracle for "is a stage module running?": alive | dead | unknown.

    pgrep exits 0 when it matched at least one process and 1 when it matched none. Any
    other exit status (usage error, signal, environment failure) is evidence of neither
    state, and a missing pgrep makes liveness unknowable — both report "unknown" so the
    caller can fail closed instead of guessing.
    """
    if not shutil.which("pgrep"):
        return "unknown"
    proc = subprocess.run(["pgrep", "-f", module], capture_output=True, text=True,
                          check=False)
    if proc.returncode == 0 and proc.stdout.strip():
        return "alive"
    if proc.returncode == 1:
        return "dead"
    return "unknown"


def find_running(specs: list[dict], profile: str, rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Only a provably dead RUNNING stage may be resumed; uncertainty blocks the run."""
    stale: list[dict] = []
    busy: list[str] = []
    for spec in specs:
        status, row = stage_status(rows, spec, profile)
        if status != "RUNNING":
            continue
        state = module_process_state(spec["module"])
        if state == "dead":
            stale.append(row)
        else:
            detail = "still running" if state == "alive" else "pgrep status unknown"
            busy.append(f"{spec['key']} ({spec['module']}, {row.get('run_id')}; {detail})")
    return stale, busy


def mark_interrupted(registry_path: Path, stale: list[dict], run_id: str) -> list[str]:
    marked = []
    registry = RunRegistry(registry_path)
    for row in stale:
        registry.append(
            run_id=row.get("run_id"), stage=row.get("stage"), status="INTERRUPTED",
            profile=row.get("profile"),
            **{key: row[key] for key in ("loss", "regime", "split", "splits") if key in row},
            event="interrupted", marked_by=run_id,
            reason="RUNNING nhưng process không còn sống (pgrep không thấy); đánh dấu "
                   "INTERRUPTED theo spec 06 §5",
        )
        marked.append(f"{row.get('stage')}:{row.get('run_id')}")
    return marked


def recover_killed_gpu_intervals(ledger: GpuLedger, specs: list[dict], rows: list[dict],
                                 registry_path: Path, run_id: str, profile: str
                                 ) -> tuple[list[str], list[str]]:
    """--resume: close intervals a killed invocation left open, conservatively.

    Fail-closed and never underbill: an interval is closed only when (a) ownership is
    recorded — its ``run_id`` belongs to a run_all ``start`` row — (b) its ``stage_key``
    resolves to a DAG module, and (c) pgrep definitively proves that module's process is
    gone (exit 1). The close instant is then the recovery NOW: an upper bound on the true
    allocation end that also charges the idle downtime between the kill and the recovery.
    That overbilling is deliberate — the alternative, closing at the stage log's mtime as a
    guessed "stop instant" (the previous rule), underbills every silent job, whose log
    freezes at launch while the GPU stays held. A precise device-time correction needs a
    new mechanism (e.g. device-side accounting); recovery never pretends to know it.
    Anything uncertain — a live process, a pgrep that is missing or exits with a status
    other than 0/1, an unknown owner or stage — leaves the interval open and reported, and
    the pre-launch GPU gate keeps failing closed on it.
    """
    try:
        opened = ledger.open_intervals()
    except GpuLedgerError as exc:
        return [], [f"không đọc được ledger: {exc}"]
    if not opened:
        return [], []

    by_key = {spec["key"]: spec for spec in specs}
    starts = {row.get("run_id") for row in rows
              if row.get("stage") == ORCH_STAGE and row.get("event") == "start"}
    recovered: list[str] = []
    unresolved: list[str] = []
    for row in opened:
        interval_id = row.get("interval_id")
        stage_key = row.get("stage_key")
        if not row.get("gpu", True) and row.get("run_id") not in starts:
            continue  # Direct CPU intervals belong to gpu_stage; do not recover them here.
        if row.get("run_id") not in starts:
            unresolved.append(f"{interval_id}: không có registry row start cho run "
                              f"{row.get('run_id')!r} — không xác minh được chủ sở hữu")
            continue
        spec = by_key.get(stage_key)
        if spec is None:
            unresolved.append(f"{interval_id}: stage_key {stage_key!r} không có trong DAG")
            continue
        state = module_process_state(spec["module"])
        if state == "alive":
            unresolved.append(f"{interval_id}: {spec['module']} vẫn còn tiến trình sống — "
                              f"không đóng interval")
            continue
        if state == "unknown":
            unresolved.append(f"{interval_id}: pgrep không kết luận được trạng thái của "
                              f"{spec['module']} (thiếu pgrep hoặc rc khác 0/1) — không "
                              f"đóng interval (fail closed)")
            continue
        ended = _dt.datetime.now(_dt.timezone.utc)
        try:
            seconds = ledger.close_interval(interval_id, status="INTERRUPTED", now=ended)
        except GpuLedgerError as exc:
            unresolved.append(f"{interval_id}: không đóng được tại {ended.isoformat()}: {exc}")
            continue
        RunRegistry(registry_path).append(
            run_id=run_id, stage=ORCH_STAGE, status="SUCCEEDED",
            profile=profile, event="recovered", interval_id=interval_id,
            stage_key=stage_key, owner_run_id=row.get("run_id"),
            ended=ended.isoformat(), seconds=round(seconds, 3), gpu=row.get("gpu", True),
            evidence=f"pgrep -f {spec['module']} rc=1 (không còn tiến trình) lúc "
                     f"{ended.isoformat()}",
            reason=("interval GPU bị bỏ ngỏ; đóng tại lúc khôi phục, tính cả thời gian chết"
                    if row.get("gpu", True) else
                    "interval CPU bị bỏ ngỏ; đóng tại lúc khôi phục, không tính GPU"),
        )
        kind = "GPU" if row.get("gpu", True) else "CPU"
        recovered.append(f"{stage_key} [{interval_id}] → INTERRUPTED lúc "
                         f"{ended.isoformat()} ({seconds:.1f}s {kind}, upper bound tại lúc khôi "
                         f"phục; pgrep xác nhận process đã mất)")
    return recovered, unresolved


def gpu_budget_block(ledger: GpuLedger, max_gpu_hours) -> str | None:
    """Fail-closed check before one GPU launch (first attempt or OOM retry); returns a
    reason when it must not run.

    No per-stage GPU forecast exists in this repository (nothing measures or estimates a
    stage's duration), so only measured facts are enforced: the cumulative ledger spend
    against limits.max_gpu_hours, and any interval a killed invocation left open (charged
    to now by the ledger and refused here). The projected-overshoot check is therefore not
    performed — inventing a forecast would be worse than not projecting.
    """
    try:
        open_gpu = [row for row in ledger.open_intervals() if row.get("gpu", True)]
    except GpuLedgerError as exc:
        return (f"ledger GPU hỏng — fail closed, không mở stage GPU nào: {exc}")
    if open_gpu:
        ids = ", ".join(str(row.get("interval_id")) for row in open_gpu)
        return (f"còn interval GPU chưa đóng từ invocation bị kill ({ids}) — không mở "
                f"stage GPU mới (fail closed)")
    if not ledger.budget_ok(max_gpu_hours):
        spent_hours = ledger.spent_seconds() / 3600
        return (f"vượt ngân sách GPU tích luỹ (budget): đã dùng {spent_hours:.2f}h >= hạn mức "
                f"limits.max_gpu_hours={max_gpu_hours}")
    return None


def oom_retry_overlay(spec: dict, profile: str, overlay: dict, attempt: int
                      ) -> tuple[dict, list[str]]:
    """Next smaller-resource overlay for one more attempt (spec 06 §6)."""
    cfg, _ = resolved_config_sha(profile, overlay)
    new = json.loads(json.dumps(overlay))
    diffs: list[str] = []

    def put(path: list[str], value) -> None:
        node = new
        for part in path[:-1]:
            node = node.setdefault(part, {})
        cur = cfg
        for part in path:
            cur = (cur or {}).get(part) if isinstance(cur, dict) else None
        node[path[-1]] = value
        diffs.append(f"    {'.'.join(path)}: {cur} -> {value}")

    def cur_at(path: list[str]):
        node = cfg
        for part in path:
            node = (node or {}).get(part) if isinstance(node, dict) else None
        return node

    key = spec["key"]
    if key in ("train_reader",):
        if attempt == 1:
            put(["reader", "batch_size"], max(1, int(cur_at(["reader", "batch_size"]) or 1) // 2))
        else:
            put(["reader", "max_length"], 256)
            put(["reader", "grad_accum"],
                max(1, int(cur_at(["reader", "grad_accum"]) or 1) * 2))
    elif key in ("cache_reader", "train_correctness"):
        current = int(cur_at(["reader", "inference_batch_size"]) or 16)
        put(["reader", "inference_batch_size"], max(1, current // (2 ** attempt)))
    elif key in ("build_retrieval",):
        current = int(cur_at(["retrieval", "dense", "batch_size"]) or 16)
        put(["retrieval", "dense", "batch_size"], max(1, current // (2 ** attempt)))
    elif key.startswith("train_laya"):
        if attempt == 1:
            put(["laya", "max_len"], 512)
            put(["laya", "head_max_len"], 128)
        else:
            put(["laya", "max_len"], 384)
            put(["laya", "head_max_len"], 96)
            put(["laya", "freeze_encoder"], True)
    elif key == "train_rl":
        current = int(cur_at(["rl", "batch_episodes"]) or 16)
        put(["rl", "batch_episodes"], max(8, current // (2 ** attempt)))
    else:
        return overlay, []  # no known knob: caller refuses the retry honestly
    return new, diffs


def run_stage(spec: dict, profile: str, seed: int | None, overlay: dict,
              log_path: Path, extra_args: list[str]) -> tuple[int, str]:
    cmd = [PYTHON, "-m", spec["module"], "--profile", profile, *spec["args"]]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    overlay_path = None
    if overlay:
        overlay_path = log_path.with_suffix(".overlay.yaml")
        from src.utils.io import write_yaml
        write_yaml(overlay_path, overlay)
        cmd += ["--config", str(overlay_path)]
    cmd += extra_args
    print(f"    $ {shlex.join(cmd)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Line-buffered: every stage-log line hits disk immediately, so the log is a faithful
    # live trace. Recovery deliberately does NOT read its mtime as a "stop instant": a
    # silent job's log freezes at launch while the GPU stays held, which would underbill.
    with open(log_path, "w", encoding="utf-8", buffering=1) as log:
        log.write(f"# {' '.join(cmd)}\n")
        # close_fds: a child must never inherit the run_all lock fd — flock follows the open
        # file description, so an orphaned child holding it would keep the namespace locked
        # after this process dies.
        proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, close_fds=True)
        captured: list[str] = []
        for line in proc.stdout:
            sys.stdout.write("      " + line)
            log.write(line)
            captured.append(line)
        proc.wait()
    return proc.returncode, "".join(captured)


def read_manifests(path: Path) -> dict:
    out = {}
    if path.is_dir():
        for file in sorted(path.glob("*.json")):
            try:
                out[file.stem] = json.loads(file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
    return out


def write_stage_manifest(path: Path, spec: dict, profile: str, registry_row: dict,
                         registry_run_id: str, cfg_sha: str, base_sha: str, overlay: dict,
                         command: list[str], started: str, ended: str, seconds: float,
                         log_rel: str, input_sha: dict) -> None:
    outputs = {}
    for artifact, meta in (registry_row.get("artifacts") or {}).items():
        try:
            rel = str(Path(artifact).relative_to(REPO))
        except ValueError:
            rel = artifact
        outputs[rel] = meta
    atomic_write_json(path, {
        "stage_key": spec["key"], "registry_stage": spec["stage"],
        "registry_run_id": registry_run_id, "module": spec["module"], "profile": profile,
        "regime": (spec.get("row") or {}).get("regime")
        or (spec.get("row") or {}).get("loss") or (spec.get("row") or {}).get("split"),
        "command": shlex.join(command), "config_sha": cfg_sha,
        "base_config_sha": base_sha, "overlay": overlay,
        "code_git_sha": git_sha(REPO), "input_sha": input_sha, "outputs_sha": outputs,
        "started_utc": started, "ended_utc": ended, "seconds": round(seconds, 3),
        "exit_code": 0, "log": log_rel,
    })


# --------------------------------------------------------------------------- the plan

def print_plan(specs: list[dict], plan: dict, profile: str) -> None:
    print(f"KẾ HOẠCH — profile={profile}")
    print(f"  {'#':>2}  {'stage_key':<22} {'quyết định':<6} lý do")
    for index, spec in enumerate(specs, start=1):
        verdict, reason, _ = plan[spec["key"]]
        mark = {"RUN": "CHẠY", "SKIP": "BỎ QUA"}[verdict]
        print(f"  {index:>2}  {spec['key']:<22} {mark:<6} {reason}")
        if spec.get("needs_freeze"):
            print(f"      ↳ cần {FREEZE_FILE} trước khi mở test (spec 03 §5)")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="bash scripts/run_all.sh",
        description="P0..P9 orchestrator (spec 06): gated stage DAG, resume, dry-run")
    parser.add_argument("--profile", default=None, choices=list(PROFILES),
                        help="profile chạy toàn pipeline (mặc định: data.profile trong base.yaml)")
    parser.add_argument("--config", default=None, help="overlay YAML phụ, chuyển cho mọi stage")
    parser.add_argument("--seed", type=int, default=None, help="ghi đè seed chung")
    parser.add_argument("--dry-run", action="store_true",
                        help="in kế hoạch + tài nguyên đo được; không chạy stage, không ghi gì")
    parser.add_argument("--resume", action="store_true",
                        help="đánh dấu INTERRUPTED các RUNNING cũ đã chết và đóng interval "
                             "GPU bỏ ngỏ khi đã có bằng chứng process stage không còn "
                             "(pgrep rc=1); interval đóng tại lúc khôi phục — upper bound, "
                             "có thể tính dư thời gian chết — rồi chạy tiếp")
    parser.add_argument("--stop-after", default=None, metavar="STAGE_KEY",
                        help="dừng ngay sau khi xử lý xong stage này (dùng cho kiểm thử)")
    parser.add_argument("--zip", action="store_true",
                        help="chuyển --zip cho stage report (tuỳ chọn, mặc định tắt)")
    args = parser.parse_args(argv)

    base_overlay = (read_yaml(args.config) or {}) if args.config else {}
    if not isinstance(base_overlay, dict):
        parser.error("--config must contain a YAML mapping")
    if args.seed is not None:
        base_overlay = _deep_merge(base_overlay, {"seed": args.seed})
    profile = args.profile or load_config()["data"]["profile"]
    cfg, base_sha = resolved_config_sha(profile, base_overlay)
    specs = stage_specs(cfg)
    experiment_root = (REPO / "outputs/experiments").resolve()
    if (not args.dry_run and any(
            (REPO / value).resolve().is_relative_to(experiment_root)
            for value in (cfg.get("paths") or {}).values() if isinstance(value, str))):
        print("DỪNG: run_all không có forecast/watchdog cho GPU stage và report chưa "
              "được cô lập; experiment phải dùng scripts/gpu_stage.py cho các GPU "
              "stage có giới hạn, không chạy DAG run_all này.", file=sys.stderr)
        return 3
    keys = [spec["key"] for spec in specs]
    if args.stop_after is not None and args.stop_after not in keys:
        print(f"--stop-after: không có stage {args.stop_after!r}; hợp lệ: {', '.join(keys)}",
              file=sys.stderr)
        return 3

    registry_path = resolve(cfg, "runs") / "index.jsonl"
    manifests_dir = resolve(cfg, "manifests") / "run_all"
    rows = read_registry(registry_path)
    manifests = read_manifests(manifests_dir)
    plan = compute_plan(specs, profile, rows, manifests, base_sha)

    print("=" * 79)
    print(f"run_all.sh — profile={profile} config_sha={short(base_sha)} dry_run={args.dry_run} "
          f"resume={args.resume}")
    print("=" * 79)

    if args.dry_run:
        print_plan(specs, plan, profile)
        print("\nTÀI NGUYÊN (đo tại chỗ; không ước lượng thời gian/VRAM theo stage vì chưa đo):")
        for line in resource_report():
            print(line)
        gpu_hours = ((cfg.get("limits") or {}).get("max_gpu_hours"))
        print(f"  ngân sách: limits.max_gpu_hours={gpu_hours} (đối chiếu tích luỹ từ "
              f"outputs/experiments/gpu_ledger.jsonl trước mỗi stage GPU)")
        print(f"  gợi ý: chạy `bash scripts/preflight.sh` trước khi chạy thật.")
        print("DRY-RUN: không chạy stage nào, không ghi registry/stage_manifest.")
        return 0

    # Khoá cố định ở project root (ngoài mọi path mà --config có thể dịch chuyển):
    # một máy/L4 chỉ có một run_all đang ghi, kể cả các namespace riêng. Tránh ghi đè
    # artifact khi chỉ một phần paths được overlay và tránh chia nhỏ ngân sách GPU.
    # close_fds=True bên dưới ngăn stage con thừa hưởng fd; dry-run không lấy khoá.
    lock_path = REPO / "outputs/experiments/.run_all.lock"
    lock = OutputNamespaceLock(lock_path)
    try:
        lock.acquire()
    except LockHeld as exc:
        print(f"DỪNG: namespace registry/outputs đang bị khoá độc quyền bởi một run_all "
              f"khác (lock): {exc}", file=sys.stderr)
        print("Không ghi gì; chờ invocation kia kết thúc rồi chạy lại.", file=sys.stderr)
        return 3

    ledger = GpuLedger(REPO / "outputs/experiments/gpu_ledger.jsonl")
    try:
        ledger.open_intervals()  # validate even before CPU-only stages touch the registry
    except GpuLedgerError as exc:
        print(f"DỪNG: ledger GPU hỏng — không khởi chạy stage: {exc}", file=sys.stderr)
        return 3
    # A prior invocation may have finished between the lock-free preview and lock acquire.
    # Gate only against registry and manifests observed while holding the writer lock.
    rows = read_registry(registry_path)
    manifests = read_manifests(manifests_dir)
    plan = compute_plan(specs, profile, rows, manifests, base_sha)

    stale, busy = find_running(specs, profile, rows)
    if busy:
        print("DỪNG: stage còn RUNNING hoặc pgrep không xác minh được tiến trình "
              "(không chạy chồng):", file=sys.stderr)
        for item in busy:
            print(f"  - {item}", file=sys.stderr)
        print("Chờ tiến trình kết thúc và xác minh pgrep rồi chạy lại run_all.",
              file=sys.stderr)
        return 3
    if stale and not args.resume:
        print("DỪNG: có stage còn RUNNING mà process đã chết:", file=sys.stderr)
        for row in stale:
            print(f"  - {row.get('stage')} ({row.get('run_id')})", file=sys.stderr)
        print("Chạy lại với --resume để đánh dấu INTERRUPTED và tiếp tục (spec 06 §5).",
              file=sys.stderr)
        return 3

    stamp = now_iso().replace(":", "")
    run_id = f"run-all-{profile}-{stamp}"
    log_dir = resolve(cfg, "diagnostics") / "run_all" / stamp
    log_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = log_dir / f"config_{profile}.json"
    snapshot_sha = snapshot_config(cfg, snapshot_path)
    assert snapshot_sha == base_sha, "config snapshot sha mismatch"

    registry = RunRegistry(registry_path)
    registry.append(run_id=run_id, stage=ORCH_STAGE, status="RUNNING", profile=profile,
                    event="start", config=args.config, config_sha=base_sha,
                    config_snapshot=str(snapshot_path.relative_to(REPO)),
                    seed=args.seed, log_dir=str(log_dir.relative_to(REPO)))

    if args.resume:
        recovered, unresolved = recover_killed_gpu_intervals(
            ledger, specs, rows, registry_path, run_id, profile)
        for line in recovered:
            print(f"RECOVERY LEDGER: {line}")
        for line in unresolved:
            print(f"CẢNH BÁO LEDGER: {line}", file=sys.stderr)

    if stale:
        marked = mark_interrupted(registry_path, stale, run_id)
        registry.append(run_id=run_id, stage=ORCH_STAGE, status="SUCCEEDED",
                        event="recovery", marked=marked, profile=profile,
                        note="các RUNNING cũ đã được đánh dấu INTERRUPTED trước khi tiếp tục")
        rows = read_registry(registry_path)
        plan = compute_plan(specs, profile, rows, manifests, base_sha)
        print(f"RECOVERY: đã đánh dấu INTERRUPTED {len(marked)} dòng: {', '.join(marked)}")

    print_plan(specs, plan, profile)
    executed: list[str] = []
    skipped: list[dict] = []
    failures: list[str] = []
    started_all = time.time()
    max_retries = int((cfg.get("limits") or {}).get("max_retries_per_stage", 2))
    max_gpu_hours = (cfg.get("limits") or {}).get("max_gpu_hours")

    def event(**extra) -> None:
        registry.append(run_id=run_id, stage=ORCH_STAGE, **extra)

    exit_code = 0
    for index, spec in enumerate(specs, start=1):
        verdict, reason, overlay = plan[spec["key"]]
        print("-" * 79)
        print(f"[{index}/{len(specs)}] {spec['key']} ({spec['stage']}) — "
              f"{'CHẠY' if verdict == 'RUN' else 'BỎ QUA'}: {reason}")
        if verdict == "SKIP":
            skipped.append({"stage_key": spec["key"], "reason": reason})
            event(status="SKIPPED", event="stage", stage_key=spec["key"],
                  registry_stage=spec["stage"], reason=reason)
            if args.stop_after == spec["key"]:
                print(f"  --stop-after {spec['key']}: dừng theo yêu cầu.")
                break
            continue

        if spec.get("optional") and not (REPO / (spec["module"].replace(".", "/") + ".py")).is_file():
            reason = f"chưa có module {spec['module']} — bỏ qua (các stage khác chạy tiếp)"
            print(f"  BỎ QUA: {reason}")
            skipped.append({"stage_key": spec["key"], "reason": reason})
            event(status="SKIPPED", event="stage", stage_key=spec["key"],
                  registry_stage=spec["stage"], reason=reason)
            continue

        if spec.get("needs_freeze"):
            freeze = resolve(cfg, "manifests") / FREEZE_FILE
            if not freeze.is_file():
                reason = (f"chưa có {freeze.relative_to(REPO)} — test chỉ được mở sau khi khoá "
                          f"lựa chọn (spec 03 §5); chạy lại run_all sau khi freeze")
                print(f"  BỎ QUA: {reason}")
                skipped.append({"stage_key": spec["key"], "reason": reason})
                event(status="SKIPPED", event="stage", stage_key=spec["key"],
                      registry_stage=spec["stage"], reason=reason)
                continue

        try:
            open_same_stage = [row for row in ledger.open_intervals()
                               if row.get("stage_key") == spec["key"]]
        except GpuLedgerError as exc:
            open_same_stage = []
            blocked = f"ledger hỏng — không mở stage: {exc}"
        else:
            blocked = (f"còn interval {spec['key']} chưa đóng — không mở trùng stage"
                       if open_same_stage else None)
        if not blocked and spec["gpu"]:
            blocked = gpu_budget_block(ledger, max_gpu_hours)
        if blocked:
            exit_code = 1
            failures.append(f"{spec['key']}: {blocked}")
            print(f"  DỪNG: {blocked}", file=sys.stderr)
            break

        stage_started = now_iso()
        stage_clock = time.time()
        attempts = 0
        extra_args = ["--zip"] if (spec["key"] == "report" and args.zip) else []
        # The overlay every launch receives is the CLI base overlay merged under the stage's
        # own overlay (a previous retry's reduced-resource knobs win on conflicts). Children
        # must resolve the same config the orchestrator planned against, or an overlay moving
        # the experiment namespace (e.g. paths.runs/paths.models) would leave them writing
        # historical artifacts while the orchestrator believes it is isolated. On a
        # no-overlay baseline this is {} and the launch is byte-identical to before.
        current_overlay = _deep_merge(base_overlay, overlay)
        outcome = None
        failure_detail = None
        while True:
            attempts += 1
            log_path = log_dir / f"{index:02d}_{spec['key']}_r{attempts}.log"
            # One open per launch, closed below with the launch's terminal status; the
            # ledger bills per interval, so retries are separate disjoint allocations
            # (no double counting) and a killed invocation stays charged via its open row.
            interval_id = ledger.open_interval(run_id=run_id, stage_key=spec["key"],
                                               stage=spec["stage"], gpu=spec["gpu"])
            rc, output = run_stage(spec, profile, args.seed, current_overlay, log_path,
                                   extra_args)
            try:
                ledger.close_interval(interval_id, status="SUCCEEDED" if rc == 0 else "FAILED")
            except (GpuLedgerError, OSError) as exc:
                outcome = "FAILED"
                failure_detail = (f"{spec['key']}: không đóng được ledger interval "
                                  f"{interval_id}; dừng fail-closed: {exc}")
                print(f"  DỪNG: {failure_detail}", file=sys.stderr)
                break
            if rc == 0:
                outcome = "SUCCEEDED"
                break
            if spec["key"] == "eval_baselines_test" and rc == 2 and "REFUSED" in output:
                outcome = "SKIPPED"
                reason = "eval_baselines từ chối: " + (
                    [ln for ln in output.splitlines() if "REFUSED" in ln] or ["refused"])[-1]
                break
            oom = rc == 137 or any(sig in output for sig in OOM_SIGNATURES)
            if oom and attempts <= max_retries:
                # A retry is a fresh GPU launch: the fail-closed budget gate must run again
                # for it, over the ledger that already includes the attempt that just
                # failed — otherwise the retry relaunches a stage whose spend has already
                # reached limits.max_gpu_hours.
                if spec["gpu"]:
                    blocked = gpu_budget_block(ledger, max_gpu_hours)
                    if blocked:
                        outcome = "FAILED"
                        failure_detail = f"{spec['key']}: {blocked}"
                        print(f"  DỪNG retry: {blocked}", file=sys.stderr)
                        break
                new_overlay, diffs = oom_retry_overlay(spec, profile, current_overlay, attempts)
                if not diffs:
                    print(f"  OOM nhưng không có knob giảm tài nguyên cho {spec['key']}; "
                          f"dừng retry.", file=sys.stderr)
                    outcome = "FAILED"
                    break
                print(f"  OOM (rc={rc}); retry {attempts}/{max_retries} với overlay nhỏ hơn:")
                for line in diffs:
                    print(line)
                with open(log_path, "a", encoding="utf-8") as log:
                    log.write("\n# OOM retry — config diff:\n" + "\n".join(diffs) + "\n")
                event(status="RUNNING", event="retry", stage_key=spec["key"],
                      registry_stage=spec["stage"], attempt=attempts, config_diff=diffs)
                current_overlay = new_overlay
                continue
            outcome = "FAILED"
            break
        seconds = time.time() - stage_clock
        stage_ended = now_iso()

        if outcome == "FAILED":
            exit_code = 1
            failures.append(failure_detail
                            or f"{spec['key']} thất bại (xem {log_path.name})")
            event(status="FAILED", event="stage", stage_key=spec["key"],
                  registry_stage=spec["stage"], attempts=attempts,
                  seconds=round(seconds, 3), reason=f"rc!=0, log={log_path.name}")
            print(f"  THẤT BẠI sau {attempts} lần thử; log: {log_path}", file=sys.stderr)
            break

        if outcome == "SKIPPED":
            skipped.append({"stage_key": spec["key"], "reason": reason})
            event(status="SKIPPED", event="stage", stage_key=spec["key"],
                  registry_stage=spec["stage"], attempts=attempts,
                  seconds=round(seconds, 3), reason=reason)
            print(f"  BỎ QUA: {reason}")
            if args.stop_after == spec["key"]:
                print(f"  --stop-after {spec['key']}: dừng theo yêu cầu.")
                break
            continue

        # SUCCEEDED: the stage wrote its own registry row; mirror it into a stage manifest.
        rows = read_registry(registry_path)
        group = latest_rows(rows, spec, profile)
        row = group[-1] if group else None
        if row is None or row.get("status") != "SUCCEEDED":
            exit_code = 1
            failures.append(f"{spec['key']} trả rc=0 nhưng không có registry row SUCCEEDED")
            event(status="FAILED", event="stage", stage_key=spec["key"],
                  registry_stage=spec["stage"], attempts=attempts,
                  reason="rc=0 nhưng thiếu registry row SUCCEEDED")
            print(f"  THẤT BẠI: rc=0 nhưng không thấy registry row SUCCEEDED cho "
                  f"{spec['stage']}", file=sys.stderr)
            break
        input_sha = {}
        for dep in spec["deps"]:
            dep_manifest = read_manifests(manifests_dir).get(dep)
            if dep_manifest:
                input_sha[dep] = sha256_json({"config_sha": dep_manifest.get("config_sha"),
                                              "outputs_sha": dep_manifest.get("outputs_sha")})
        _, stage_cfg_sha = resolved_config_sha(profile, current_overlay)
        manifest_path = manifests_dir / f"{spec['key']}.json"
        write_stage_manifest(manifest_path, spec, profile, row, row.get("run_id"),
                             stage_cfg_sha, base_sha, current_overlay,
                             [PYTHON, "-m", spec["module"], "--profile", profile,
                              *spec["args"]], stage_started, stage_ended, seconds,
                             str(log_path.relative_to(REPO)), input_sha)
        manifests = read_manifests(manifests_dir)
        executed.append(spec["key"])
        event(status="SUCCEEDED", event="stage", stage_key=spec["key"],
              registry_stage=spec["stage"], registry_run_id=row.get("run_id"),
              attempts=attempts, seconds=round(seconds, 3),
              manifest=str(manifest_path.relative_to(REPO)))
        print(f"  XONG sau {attempts} lần thử, {seconds:.1f}s; stage_manifest đã ghi.")
        if args.stop_after == spec["key"]:
            print(f"  --stop-after {spec['key']}: dừng theo yêu cầu.")
            break

    total_seconds = time.time() - started_all
    if exit_code == 0:
        event(status="SUCCEEDED", event="end", executed=executed,
              skipped=skipped, seconds=round(total_seconds, 3),
              log_dir=str(log_dir.relative_to(REPO)))
        print("=" * 79)
        print(f"run_all HOÀN TẤT ({total_seconds:.1f}s) — đã chạy {len(executed)} stage: "
              f"{', '.join(executed) or '(không)'}")
        if skipped:
            print("BỎ QUA:")
            for item in skipped:
                print(f"  - {item['stage_key']}: {item['reason']}")
        print(f"log: {log_dir}")
        return 0
    event(status="FAILED", event="end", executed=executed, failures=failures,
          skipped=skipped, seconds=round(total_seconds, 3),
          log_dir=str(log_dir.relative_to(REPO)))
    print("=" * 79, file=sys.stderr)
    print(f"run_all THẤT BẠI: {'; '.join(failures)}", file=sys.stderr)
    print(f"log: {log_dir}", file=sys.stderr)
    return exit_code or 1


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        print("run_all: bị ngắt (KeyboardInterrupt)", file=sys.stderr)
        raise SystemExit(130)
PY
