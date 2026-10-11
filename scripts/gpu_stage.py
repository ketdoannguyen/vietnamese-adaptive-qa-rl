#!/usr/bin/env python
"""Guard experiment pipeline launches with a shared lock and GPU/CPU ledger.

GPU launches require a measured forecast and a watchdog. Audited CPU-only stages
use --cpu-stage, share the namespace lock, and record unbilled CPU time. Bare
``python -m`` bypasses these cooperative guards. An open GPU interval bills through
recovery time; recovery refuses intervals owned by run_all or a live stage.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config  # noqa: E402
from src.utils.io import read_yaml  # noqa: E402
from src.utils.runs import GpuLedger, GpuLedgerError, LockHeld, OutputNamespaceLock  # noqa: E402

EXPERIMENTS = ROOT / "outputs/experiments"
LEDGER = EXPERIMENTS / "gpu_ledger.jsonl"
LOCK = EXPERIMENTS / ".run_all.lock"
PATH_KEYS = ("data_raw", "data_processed", "data_scenarios", "data_indexes", "data_cache",
             "data_diagnostics", "outputs", "models", "runs", "predictions", "manifests",
             "diagnostics", "reports", "plots", "tables")
GPU_MODULES = {"src.pipeline.build_retrieval", "src.pipeline.train_reader",
               "src.pipeline.cache_reader", "src.pipeline.train_laya",
               "src.pipeline.cache_laya", "src.pipeline.train_rl",
               "src.pipeline.train_correctness", "src.pipeline.eval_baselines",
               "src.pipeline.train_gliner", "src.pipeline.cache_decision"}
CPU_MODULES = {"src.pipeline.audit_split", "src.pipeline.make_teacher"}


def _module_running(module: str) -> bool:
    """Fail closed if any live command's '-m <module>' cannot be ruled out."""
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return True
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return True
    for entry in entries:
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except FileNotFoundError:
            continue
        except OSError:
            return True
        if any(arg == b"-m" and index + 1 < len(argv)
               and argv[index + 1] == module.encode()
               for index, arg in enumerate(argv)):
            return True
    return False


def _is_experiment_config(cfg: dict) -> bool:
    root = EXPERIMENTS.resolve()
    paths = []
    for key in PATH_KEYS:
        value = cfg.get("paths", {}).get(key)
        if not isinstance(value, str):
            return False
        path = (ROOT / value).resolve()
        if not path.is_relative_to(root) or path == root:
            return False
        paths.append(path)
    # Do not mix histories from different experiments in a single launch.
    return len({path.relative_to(root).parts[0] for path in paths}) == 1


def _recover(ledger: GpuLedger) -> int:
    pending = ledger.open_intervals()
    errors = []
    for row in pending:
        interval_id = row["interval_id"]
        module = row.get("stage")
        if not str(row.get("run_id", "")).startswith("direct-") or not isinstance(module, str):
            if row.get("gpu", True):
                errors.append(f"{interval_id}: not owned by direct runner; use run_all --resume")
            else:
                print(f"{interval_id}: non-direct CPU interval left to its owner (unbilled)")
        elif _module_running(module):
            errors.append(f"{interval_id}: module still running or process status unknown")
        else:
            seconds = ledger.close_interval(interval_id, status="INTERRUPTED")
            kind = "GPU" if row.get("gpu", True) else "CPU"
            print(f"recovered {interval_id}: {seconds:.3f}s {kind}, upper bound through recovery")
    for error in errors:
        print(error, file=sys.stderr)
    return 3 if errors else 0


def _number(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a finite nonnegative number")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--module", required=True, help="pipeline module passed to python -m")
    parser.add_argument("--venv", choices=("base", "gliner"), default="base")
    parser.add_argument("--recover", action="store_true", help="close dead direct intervals at recovery time")
    parser.add_argument("--cpu-stage", action="store_true", help="only run an audited CPU-only stage")
    parser.add_argument("--forecast-seconds", type=_number)
    parser.add_argument("--reserve-seconds", type=_number, default=0)
    parser.add_argument("--timeout-seconds", type=_number, required=True)
    parser.add_argument("stage_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0:
        parser.error("timeout must be positive")
    if args.cpu_stage:
        if args.module not in CPU_MODULES or args.venv != "base":
            parser.error("--cpu-stage requires an audited CPU module and the base interpreter")
        if args.forecast_seconds is not None or args.reserve_seconds:
            parser.error("CPU stages cannot specify GPU forecast or reserve")
    elif args.module not in GPU_MODULES or args.forecast_seconds is None or args.forecast_seconds <= 0:
        parser.error("GPU stages require an audited module and positive --forecast-seconds")
    if not (ROOT / Path(*args.module.split(".")).with_suffix(".py")).is_file() and not args.recover:
        parser.error(f"stage module does not exist: {args.module}")
    if args.recover and args.stage_args:
        parser.error("--recover takes no stage arguments")
    if not args.config.is_file():
        parser.error(f"--config not found: {args.config}")
    overlay = read_yaml(args.config)
    if not isinstance(overlay, dict):
        parser.error("--config must be a YAML mapping")
    cfg = load_config(args.profile, extra=overlay)
    if not _is_experiment_config(cfg):
        parser.error("every output/data path must belong to the same outputs/experiments namespace")
    cap = (cfg.get("limits") or {}).get("max_gpu_hours")
    python = ROOT / (".venv-gliner" if args.venv == "gliner" else ".venv") / "bin/python"
    if not python.is_file():
        parser.error(f"interpreter not found: {python}")
    stage_args = args.stage_args[1:] if args.stage_args[:1] == ["--"] else args.stage_args
    forbidden = ("--config", "--profile", "--data", "--base", "--out",
                 "--force", "--dev", "--reselect", "--limit")
    if args.cpu_stage and stage_args:
        parser.error("CPU stages accept no extra arguments until split protocol is frozen")
    if any(arg.startswith("--") and any(name.startswith(arg.split("=", 1)[0])
           for name in forbidden) for arg in stage_args):
        parser.error("stage arguments must not override paths, profile, or artifact safety")
    ledger = GpuLedger(LEDGER)
    try:
        with OutputNamespaceLock(LOCK) as lock:
            if args.recover:
                return _recover(ledger)
            if not args.cpu_stage and (
                isinstance(cap, bool) or not isinstance(cap, (float, int))
                or not math.isfinite(cap) or cap <= 0
            ):
                raise ValueError("limits.max_gpu_hours must be a positive finite number")
            if isinstance(cap, (float, int)) and not isinstance(cap, bool) and cap > 48:
                raise ValueError("limits.max_gpu_hours exceeds the authorized 48 GPU-hour ceiling")
            opened = ledger.open_intervals()
            if any(row.get("gpu", True) or str(row.get("run_id", "")).startswith("direct-")
                   for row in opened):
                raise ValueError("open stage interval (fail closed): recover with --recover or run_all --resume")
            spent = ledger.spent_seconds()
            if not args.cpu_stage and (spent + max(args.forecast_seconds, args.timeout_seconds + 30)
                                       + args.reserve_seconds >= cap * 3600):
                raise ValueError(f"forecast/watchdog plus reserve exceeds {cap} GPU-hours; spent={spent:.3f}s")
            if _module_running(args.module):
                raise ValueError("stage module already running outside the guarded lock")
            # Freeze the validated overlay: the child must not re-read a mutable config path.
            namespace = (ROOT / cfg["paths"]["diagnostics"]).resolve()
            namespace.mkdir(parents=True, exist_ok=True)
            run_id = f"direct-{args.module.rsplit('.', 1)[-1]}-{uuid.uuid4().hex}"
            from src.utils.io import write_yaml
            stage_dir = "cpu_stage" if args.cpu_stage else "gpu_stage"
            config_snapshot = namespace / stage_dir / f"{run_id}.yaml"
            config_snapshot.parent.mkdir(parents=True, exist_ok=True)
            write_yaml(config_snapshot, overlay)
            command = [str(python), "-m", args.module, "--profile", args.profile,
                       "--config", str(config_snapshot), *stage_args]
            interval_id = ledger.open_interval(run_id=run_id, stage_key=args.module,
                                               stage=args.module, gpu=not args.cpu_stage)
            log_dir = namespace / stage_dir
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{run_id}.log"
            result = None
            interrupted = False
            child = None
            def on_signal(signum, _frame):
                nonlocal interrupted
                interrupted = True
                if child is not None and child.poll() is None:
                    child.send_signal(signum)

            previous = {sig: signal.signal(sig, on_signal) for sig in (signal.SIGTERM, signal.SIGINT)}
            try:
                with log_path.open("x", encoding="utf-8") as log:
                    log.write(json.dumps({"run_id": run_id, "interval_id": interval_id,
                                          "cap_hours": cap, "spent_at_admission": spent,
                                          "gpu": not args.cpu_stage,
                                          "forecast_seconds": args.forecast_seconds,
                                          "reserve_seconds": args.reserve_seconds,
                                          "timeout_seconds": args.timeout_seconds,
                                          "command": command}, ensure_ascii=False) + "\n")
                    log.flush()
                    # Child keeps the lock if this supervisor is SIGKILLed while it is alive.
                    if interrupted:
                        raise InterruptedError("signal received before GPU child launched")
                    child = subprocess.Popen(command, cwd=ROOT, stdout=log,
                                             stderr=subprocess.STDOUT,
                                             pass_fds=(lock._fd,))
                    if interrupted and child.poll() is None:
                        child.terminate()
                    try:
                        result = child.wait(timeout=args.timeout_seconds)
                    except subprocess.TimeoutExpired:
                        child.terminate()
                        try:
                            child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                        result = 124
                status = "INTERRUPTED" if interrupted else "SUCCEEDED" if result == 0 else "FAILED"
            except InterruptedError:
                status = "INTERRUPTED"
                result = 130
            except BaseException:
                status = "INTERRUPTED"
                if child is not None and child.poll() is None:
                    child.terminate()
                    child.wait()
                raise
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
                kind = "CPU" if args.cpu_stage else "GPU"
                try:
                    seconds = ledger.close_interval(interval_id, status=status)
                    print(f"{status}: {seconds:.3f} {kind} seconds; interval={interval_id}; log={log_path}")
                except (GpuLedgerError, OSError) as exc:
                    raise RuntimeError(f"cannot close {kind} interval {interval_id}; remains open (GPU intervals bill to now): {exc}") from exc
            return 0 if status == "SUCCEEDED" else (130 if interrupted else 1)
    except (LockHeld, GpuLedgerError, ValueError, OSError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
