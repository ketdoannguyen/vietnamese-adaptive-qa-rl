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
#   complete stage. INTERRUPTED is not in src.utils.runs.STATUSES, so those rows are
#   appended with src.utils.io.append_jsonl (same file, same schema) — reported.
#
# Retries (spec 06 §6): a stage that dies with an OOM signature is retried up to
#   limits.max_retries_per_stage times with a generated config overlay that shrinks
#   the stage's own resource knobs; every retry logs the old -> new config diff.
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

from src.utils.config import load_config, resolve, snapshot_config  # noqa: E402
from src.utils.hashing import sha256_file, sha256_json, sha256_text  # noqa: E402
from src.utils.io import append_jsonl, atomic_write_json, read_yaml  # noqa: E402
from src.utils.runs import RunRegistry, git_sha, now_iso  # noqa: E402

ORCH_STAGE = "run_all"
TERMINAL = ("SUCCEEDED", "FAILED", "SKIPPED", "BLOCKED", "NOT_RUN", "INTERRUPTED")
OOM_SIGNATURES = ("OutOfMemoryError", "out of memory", "OutOfMemory", "MemoryError",
                  "Cannot allocate memory", "std::bad_alloc")
FREEZE_FILE = "best_selection.json"
PROFILES = ("smoke", "small", "pilot", "medium", "full")


# ------------------------------------------------------------------------------- DAG

def stage_specs(cfg: dict) -> list[dict]:
    """The DAG for this config. ``row`` = exact registry-key filters for the gate."""
    frozen = ((cfg.get("laya") or {}).get("frozen") or "rlcd")
    regime = None if frozen in (None, "none") else str(frozen)
    return [
        dict(key="download", stage="P1_download", module="src.pipeline.download",
             args=[], deps=[]),
        dict(key="audit_split", stage="P1_audit_split", module="src.pipeline.audit_split",
             args=[], deps=["download"]),
        dict(key="build_retrieval", stage="P2_retrieval",
             module="src.pipeline.build_retrieval", args=[], deps=["audit_split"]),
        dict(key="train_reader", stage="P3_reader", module="src.pipeline.train_reader",
             args=[], deps=["audit_split"]),
        dict(key="cache_reader", stage="P3_reader_cache", module="src.pipeline.cache_reader",
             args=[], deps=["train_reader", "build_retrieval"],
             row={"splits": ["train_strict", "val_strict"]}),
        dict(key="make_teacher", stage="P5_teacher", module="src.pipeline.make_teacher",
             args=[], deps=["cache_reader", "build_retrieval"]),
        dict(key="train_laya_ce", stage="P5_laya_train", module="src.pipeline.train_laya",
             args=["--loss", "soft-ce"], deps=["make_teacher"], row={"loss": "soft-ce"}),
        dict(key="train_laya_rlcd", stage="P5_laya_train", module="src.pipeline.train_laya",
             args=["--loss", "rlcd"], deps=["make_teacher"], row={"loss": "rlcd"}),
        dict(key="cache_laya", stage="P5_cache_laya", module="src.pipeline.cache_laya",
             args=[], deps=["train_laya_rlcd"], row={"regime": regime}),
        dict(key="train_rl", stage="P6_rl", module="src.pipeline.train_rl",
             args=[], deps=["cache_laya", "cache_reader"]),
        dict(key="select_rl", stage="P6_select", module="src.pipeline.select_rl",
             args=[], deps=["train_rl"]),
        dict(key="train_correctness", stage="B5_correctness",
             module="src.pipeline.train_correctness", args=[],
             deps=["cache_reader", "build_retrieval"], optional=True),
        dict(key="eval_baselines_val", stage="P7_baselines",
             module="src.pipeline.eval_baselines", args=["--split", "val_strict"],
             deps=["select_rl", "train_correctness", "cache_laya", "build_retrieval",
                   "train_laya_ce"],
             row={"split": "val_strict"}),
        dict(key="cache_reader_test", stage="P3_reader_cache",
             module="src.pipeline.cache_reader", args=["--splits", "test_strict"],
             deps=["select_rl"], row={"splits": ["test_strict"]}),
        dict(key="eval_baselines_test", stage="P7_baselines",
             module="src.pipeline.eval_baselines", args=["--split", "test_strict"],
             deps=["eval_baselines_val", "cache_reader_test"], row={"split": "test_strict"},
             needs_freeze=True),
        dict(key="report", stage="P9_report", module="src.pipeline.report", args=[],
             deps=["download", "audit_split", "build_retrieval", "train_reader",
                   "cache_reader", "make_teacher", "train_laya_ce", "train_laya_rlcd",
                   "cache_laya", "train_rl", "select_rl", "train_correctness",
                   "cache_reader_test", "eval_baselines_val", "eval_baselines_test"]),
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


def gate(spec: dict, profile: str, rows: list[dict], manifests: dict) -> tuple[str, str, dict]:
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
    if overlay:
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


def compute_plan(specs: list[dict], profile: str, rows: list[dict], manifests: dict
                 ) -> dict[str, tuple[str, str, dict]]:
    plan: dict[str, tuple[str, str, dict]] = {}
    for spec in specs:
        plan[spec["key"]] = gate(spec, profile, rows, manifests)
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

def module_alive(module: str) -> bool:
    if not shutil.which("pgrep"):
        return False
    proc = subprocess.run(["pgrep", "-f", module], capture_output=True, text=True,
                          check=False)
    return proc.returncode == 0 and proc.stdout.strip() != ""


def find_running(specs: list[dict], profile: str, rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Split RUNNING stage rows: dead process -> stale (spec 06 §5), live -> busy (abort)."""
    stale: list[dict] = []
    busy: list[str] = []
    for spec in specs:
        status, row = stage_status(rows, spec, profile)
        if status != "RUNNING":
            continue
        if module_alive(spec["module"]):
            busy.append(f"{spec['key']} ({spec['module']}, {row.get('run_id')})")
        else:
            stale.append(row)
    return stale, busy


def mark_interrupted(registry_path: Path, stale: list[dict], run_id: str) -> list[str]:
    marked = []
    for row in stale:
        append_jsonl(registry_path, {
            "run_id": row.get("run_id"), "stage": row.get("stage"), "status": "INTERRUPTED",
            "ts": now_iso(), "profile": row.get("profile"),
            "event": "interrupted", "marked_by": run_id,
            "reason": "RUNNING nhưng process không còn sống (pgrep không thấy); đánh dấu "
                      "INTERRUPTED theo spec 06 §5 (INTERRUPTED không có trong STATUSES)",
        })
        marked.append(f"{row.get('stage')}:{row.get('run_id')}")
    return marked


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
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"# {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
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
                         registry_run_id: str, cfg_sha: str, overlay: dict, command: list[str],
                         started: str, ended: str, seconds: float, log_rel: str,
                         input_sha: dict) -> None:
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
        "command": shlex.join(command), "config_sha": cfg_sha, "overlay": overlay,
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
                        help="đánh dấu INTERRUPTED các RUNNING cũ đã chết rồi chạy tiếp")
    parser.add_argument("--stop-after", default=None, metavar="STAGE_KEY",
                        help="dừng ngay sau khi xử lý xong stage này (dùng cho kiểm thử)")
    parser.add_argument("--zip", action="store_true",
                        help="chuyển --zip cho stage report (tuỳ chọn, mặc định tắt)")
    args = parser.parse_args(argv)

    base_overlay = read_yaml(args.config) if args.config else {}
    profile = args.profile or load_config()["data"]["profile"]
    cfg, base_sha = resolved_config_sha(profile, base_overlay)
    specs = stage_specs(cfg)
    keys = [spec["key"] for spec in specs]
    if args.stop_after is not None and args.stop_after not in keys:
        print(f"--stop-after: không có stage {args.stop_after!r}; hợp lệ: {', '.join(keys)}",
              file=sys.stderr)
        return 3

    registry_path = resolve(cfg, "runs") / "index.jsonl"
    manifests_dir = resolve(cfg, "manifests") / "run_all"
    rows = read_registry(registry_path)
    manifests = read_manifests(manifests_dir)
    plan = compute_plan(specs, profile, rows, manifests)

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
        print(f"  ngân sách: limits.max_gpu_hours={gpu_hours} "
              f"(đối chiếu theo từng invocation run_all, chưa có đồng hồ tích luỹ)")
        print(f"  gợi ý: chạy `bash scripts/preflight.sh` trước khi chạy thật.")
        print("DRY-RUN: không chạy stage nào, không ghi registry/stage_manifest.")
        return 0

    stale, busy = find_running(specs, profile, rows)
    if busy:
        print("DỪNG: stage đang chạy ở tiến trình khác (không chạy chồng):", file=sys.stderr)
        for item in busy:
            print(f"  - {item}", file=sys.stderr)
        print("Chờ tiến trình đó kết thúc rồi chạy lại run_all.", file=sys.stderr)
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

    if stale:
        marked = mark_interrupted(registry_path, stale, run_id)
        registry.append(run_id=run_id, stage=ORCH_STAGE, status="SUCCEEDED",
                        event="recovery", marked=marked, profile=profile,
                        note="các RUNNING cũ đã được đánh dấu INTERRUPTED trước khi tiếp tục")
        rows = read_registry(registry_path)
        plan = compute_plan(specs, profile, rows, manifests)
        print(f"RECOVERY: đã đánh dấu INTERRUPTED {len(marked)} dòng: {', '.join(marked)}")

    print_plan(specs, plan, profile)
    executed: list[str] = []
    skipped: list[dict] = []
    failures: list[str] = []
    started_all = time.time()
    max_retries = int((cfg.get("limits") or {}).get("max_retries_per_stage", 2))
    budget_seconds = float((cfg.get("limits") or {}).get("max_gpu_hours") or 0) * 3600

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

        if budget_seconds and (time.time() - started_all) > budget_seconds:
            exit_code = 1
            failures.append(f"vượt ngân sách GPU trước stage {spec['key']}")
            print(f"  DỪNG: vượt limits.max_gpu_hours; chưa chạy {spec['key']} và các stage sau.",
                  file=sys.stderr)
            break

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

        stage_started = now_iso()
        stage_clock = time.time()
        attempts = 0
        extra_args = ["--zip"] if (spec["key"] == "report" and args.zip) else []
        current_overlay = dict(overlay)
        outcome = None
        while True:
            attempts += 1
            log_path = log_dir / f"{index:02d}_{spec['key']}_r{attempts}.log"
            rc, output = run_stage(spec, profile, args.seed, current_overlay, log_path,
                                   extra_args)
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
            failures.append(f"{spec['key']} thất bại (xem {log_path.name})")
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
                             stage_cfg_sha, current_overlay,
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
