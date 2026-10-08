"""P0 preflight: observe the real machine and write machine-readable diagnostics.

Never assumes hardware. Everything reported here is measured or explicitly null.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from src.utils.config import REPO_ROOT
from src.utils.io import atomic_write_json


def _run(cmd: list[str], timeout: int = 30) -> dict:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return {"cmd": cmd, "returncode": out.returncode,
                "stdout": out.stdout.strip(), "stderr": out.stderr.strip()}
    except FileNotFoundError:
        return {"cmd": cmd, "returncode": None, "stdout": "", "stderr": "not found"}
    except subprocess.TimeoutExpired:
        return {"cmd": cmd, "returncode": None, "stdout": "", "stderr": "timeout"}


def gpu_info() -> dict:
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return {"nvidia_smi": None, "gpus": [], "torch": None}
    query = _run([smi, "--query-gpu=index,name,memory.total,memory.free,driver_version,compute_cap",
                  "--format=csv,noheader,nounits"])
    gpus = []
    for line in query["stdout"].splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            gpus.append({
                "index": int(parts[0]), "name": parts[1],
                "memory_total_mib": int(parts[2]), "memory_free_mib": int(parts[3]),
                "driver_version": parts[4], "compute_cap": parts[5] if len(parts) > 5 else None,
            })
    torch_info = None
    try:
        import torch
        torch_info = {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
        }
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            torch_info.update({
                "device_name": props.name,
                "total_memory_gib": round(props.total_memory / 2**30, 2),
                "capability": f"{props.major}.{props.minor}",
                "usable_bf16": props.major >= 8,
            })
    except Exception as exc:  # torch may not be installed yet
        torch_info = {"error": f"{type(exc).__name__}: {exc}"}
    return {"nvidia_smi": smi, "gpus": gpus, "torch": torch_info}


def system_info() -> dict:
    mem = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                k, _, rest = line.partition(":")
                if k in ("MemTotal", "MemAvailable", "SwapTotal"):
                    mem[k] = int(rest.strip().split()[0]) // 1024  # MiB
    except OSError:
        pass
    disks = {}
    for label, path in (("root", "/"), ("home", str(Path.home())), ("tmp", "/tmp")):
        if not os.path.exists(path):
            continue
        usage = shutil.disk_usage(path)
        disks[label] = {"path": path, "total_gb": round(usage.total / 2**30, 1),
                        "free_gb": round(usage.free / 2**30, 1),
                        "used_pct": round(100 * usage.used / usage.total, 1)}
    return {
        "hostname": platform.node(),
        "user": os.environ.get("USER", ""),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "cpu_count": os.cpu_count(),
        "cpu_model": _cpu_model(),
        "memory_mib": mem,
        "disks": disks,
        "python": sys.version.split()[0],
        "venv": os.environ.get("VIRTUAL_ENV"),
        "cwd": str(Path.cwd()),
    }


def _cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def network_info() -> dict:
    checks = {}
    for name, url in (("huggingface", "https://huggingface.co"),
                      ("pypi", "https://pypi.org"),
                      ("github", "https://github.com")):
        res = _run(["curl", "-sI", "-m", "10", "-o", "/dev/null", "-w", "%{http_code}", url], timeout=15)
        checks[name] = {"url": url, "http_code": res["stdout"] or None, "ok": res["stdout"].startswith("2")}
    return checks


def disk_guard(cfg: dict, min_free_gb: float = 20.0) -> dict:
    usage = shutil.disk_usage(str(REPO_ROOT))
    free_gb = usage.free / 2**30
    return {"free_gb": round(free_gb, 1), "min_free_gb": min_free_gb, "ok": free_gb >= min_free_gb}


def evaluate(gpu: dict, sysinfo: dict, net: dict, disk: dict) -> dict:
    gpus = gpu.get("gpus") or []
    torch_info = gpu.get("torch") or {}
    vram = gpus[0]["memory_total_mib"] if gpus else 0
    ram_mib = sysinfo.get("memory_mib", {}).get("MemTotal", 0)
    checks = {
        "gpu_present": bool(gpus),
        "gpu_vram_ge_20gb": vram >= 20 * 1024,
        "torch_cuda_available": bool(torch_info.get("cuda_available")),
        "ram_ge_12gb": ram_mib >= 12 * 1024,
        "disk_ok": disk["ok"],
        "hf_reachable": net.get("huggingface", {}).get("ok", False),
        "pypi_reachable": net.get("pypi", {}).get("ok", False),
    }
    blockers = [k for k, v in checks.items() if not v]
    return {"checks": checks, "blockers": blockers,
            "status": "READY_FOR_TRAIN" if not blockers else "DEGRADED_RUN_CPU_OR_REPAIR"}


def write_report(out_dir: Path, gpu: dict, sysinfo: dict, net: dict, disk: dict, verdict: dict) -> None:
    lines = ["# Preflight report", "", f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}", "",
             f"**Status:** `{verdict['status']}`", ""]
    if verdict["blockers"]:
        lines += [f"**Blockers:** {', '.join(verdict['blockers'])}", ""]
    lines += ["## Checks", ""]
    for k, v in verdict["checks"].items():
        lines.append(f"- {'PASS' if v else 'FAIL'} — `{k}`")
    lines += ["", "## GPUs", ""]
    if gpu["gpus"]:
        lines += ["| idx | name | total MiB | free MiB | driver | cc |",
                  "|---|---|---|---|---|---|"]
        for g in gpu["gpus"]:
            lines.append(f"| {g['index']} | {g['name']} | {g['memory_total_mib']} | "
                         f"{g['memory_free_mib']} | {g['driver_version']} | {g['compute_cap']} |")
    else:
        lines.append("No GPU visible via `nvidia-smi`.")
    lines += ["", "## torch", "", "```json", json.dumps(gpu.get("torch"), indent=2), "```", "",
              "## System", "", "```json", json.dumps(sysinfo, indent=2, ensure_ascii=False), "```", "",
              "## Network", "", "```json", json.dumps(net, indent=2), "```", "",
              "## Disk guard", "", "```json", json.dumps(disk, indent=2), "```", ""]
    (out_dir / "preflight.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="P0 preflight")
    parser.add_argument("--out", default=str(REPO_ROOT / "outputs" / "diagnostics"))
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu = gpu_info()
    sysinfo = system_info()
    net = network_info()
    disk = disk_guard({})
    verdict = evaluate(gpu, sysinfo, net, disk)

    atomic_write_json(out_dir / "gpu_info.json", gpu)
    atomic_write_json(out_dir / "system_info.json", sysinfo)
    atomic_write_json(out_dir / "network_info.json", net)
    atomic_write_json(out_dir / "preflight_verdict.json", verdict)
    write_report(out_dir, gpu, sysinfo, net, disk, verdict)

    if shutil.which("pip") or (Path(sys.executable).parent / "pip").exists():
        freeze = _run([sys.executable, "-m", "pip", "freeze"])
        (REPO_ROOT / "requirements.lock").write_text(freeze["stdout"] + "\n", encoding="utf-8")
    (out_dir / "preflight_exit_code.txt").write_text(
        f"{0 if verdict['status'] == 'READY_FOR_TRAIN' else 2}\n", encoding="utf-8")

    print(json.dumps({"status": verdict["status"], "blockers": verdict["blockers"],
                      "gpus": [g["name"] for g in gpu["gpus"]]}, ensure_ascii=False))
    return 0 if verdict["status"] == "READY_FOR_TRAIN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
