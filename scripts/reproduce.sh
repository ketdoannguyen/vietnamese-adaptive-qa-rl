#!/usr/bin/env bash
# =============================================================================
# scripts/reproduce.sh — single-command reproduce (spec 06 §8).
#
#   bash scripts/reproduce.sh --run-id <recorded-run-id> --mode eval \
#        [--config <overlay.yaml>] [--subset N] [--json]
#
# What it does, in order:
#   1. looks the run up in outputs/runs/index.jsonl and re-verifies EVERY sha256 /
#      size recorded in that run's rows (source, config and artifact hashes);
#   2. for any run_all stage_manifest that recorded this run, re-resolves
#      base+profile (+ the manifest's overlay) and compares the config sha;
#   3. eval mode: regenerates a predictions subset by re-running eval_baselines with
#      the run's own split/seed/limit (loading the saved models + indexes — this
#      script NEVER re-trains fine-tunes) and compares the overlapping
#      (exp_id, seed, qid) rows: exact for categorical fields, 1e-9 for floats.
#
# Exit codes: 0 = không có FAIL (mọi NOT_RUN được nêu rõ), 1 = có FAIL,
#             3 = lỗi tham số / không tìm thấy run-id trong registry.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
  echo "reproduce.sh: python interpreter not found: $PYTHON (set PYTHON=/path/to/python)" >&2
  exit 3
fi

export RUN_REPRO_REPO="$REPO_ROOT"
export RUN_REPRO_PYTHON="$PYTHON"

"$PYTHON" - "$@" <<'PY'
"""Reproduce core. Imports only public helpers; never trains a fine-tune."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(os.environ["RUN_REPRO_REPO"])
PYTHON = os.environ["RUN_REPRO_PYTHON"]
sys.path.insert(0, str(REPO))

from src.utils.config import load_config  # noqa: E402
from src.utils.hashing import sha256_file, sha256_text  # noqa: E402
from src.utils.runs import now_iso  # noqa: E402

EXACT_FIELDS = ("terminal_action", "final_candidate", "final_candidate_doc_id",
                "eval_correct", "eval_is_impossible", "action_name_trace",
                "optional_used", "bm25_calls", "dense_calls", "total_retrievals",
                "forced_terminal")
FLOAT_FIELDS = ("eval_correctness_f1", "eval_correctness_em", "retrieval_cost",
                "eval_episode_return")
TOLERANCE = 1e-9
KEY = ("exp_id", "seed", "qid")

CHECKS: list[tuple[str, str, str]] = []  # (name, verdict, detail)


def record(name: str, verdict: str, detail: str = "") -> None:
    CHECKS.append((name, verdict, detail))
    print(f"  [{verdict}] {name}" + (f" — {detail}" if detail else ""))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="bash scripts/reproduce.sh",
        description="Tái lập một run đã ghi: kiểm tra hash + tái tạo subset predictions")
    parser.add_argument("--run-id", required=True, help="run_id trong outputs/runs/index.jsonl")
    parser.add_argument("--mode", default="eval", choices=["eval"],
                        help="chế độ tái lập (hiện chỉ hỗ trợ eval)")
    parser.add_argument("--config", default=None, help="overlay YAML phụ cho lần eval lại")
    parser.add_argument("--subset", type=int, default=32,
                        help="số dòng (exp_id, seed, qid) so sánh (mặc định 32)")
    parser.add_argument("--json", action="store_true", help="in kết quả dạng JSON")
    args = parser.parse_args(argv)

    registry_path = REPO / "outputs" / "runs" / "index.jsonl"
    rows = []
    if registry_path.is_file():
        for line in registry_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("run_id") == args.run_id:
                    rows.append(row)
    if not rows:
        print(f"reproduce: run_id {args.run_id!r} không có trong {registry_path}; "
              f"các run gần nhất:", file=sys.stderr)
        known = []
        if registry_path.is_file():
            for line in registry_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    known.append(json.loads(line).get("run_id"))
        for run_id in list(dict.fromkeys(known))[-8:]:
            print(f"  - {run_id}", file=sys.stderr)
        return 3

    stamp = now_iso().replace(":", "")
    log_dir = REPO / "outputs" / "diagnostics" / "reproduce" / args.run_id

    print("=" * 79)
    print(f"reproduce.sh — run_id={args.run_id} mode={args.mode} "
          f"({len(rows)} dòng registry)")
    print("=" * 79)

    # ---------------------------------------------------------------- hash check
    artifacts: dict[str, dict] = {}
    for row in rows:
        for path, meta in (row.get("artifacts") or {}).items():
            artifacts[path] = meta
    if not artifacts:
        record("hash nguồn & artifact", "NOT_RUN", "run này không ghi artifact nào")
    else:
        failures = []
        for path, meta in sorted(artifacts.items()):
            p = Path(path)
            if meta.get("missing"):
                failures.append(f"{path}: đã missing khi ghi")
            elif meta.get("dir"):
                if not p.is_dir():
                    failures.append(f"{path}: thư mục mất")
            elif not p.is_file():
                failures.append(f"{path}: tệp mất")
            elif sha256_file(p) != meta.get("sha256"):
                failures.append(f"{path}: sha256 khác bản ghi")
            elif "bytes" in meta and p.stat().st_size != meta["bytes"]:
                failures.append(f"{path}: kích thước khác bản ghi")
        if failures:
            record("hash nguồn & artifact", "FAIL", f"{len(failures)}/{len(artifacts)} lệch")
            for item in failures:
                print(f"        - {item}")
        else:
            record("hash nguồn & artifact", "PASS",
                   f"{len(artifacts)} artifact khớp sha256/kích thước")

    # -------------------------------------------------------------- config check
    profile = next((row.get("profile") for row in reversed(rows) if row.get("profile")),
                   None)
    manifests_dir = REPO / "outputs" / "manifests" / "run_all"
    manifest_hits = []
    if manifests_dir.is_dir():
        for file in sorted(manifests_dir.glob("*.json")):
            try:
                manifest = json.loads(file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if manifest.get("registry_run_id") == args.run_id:
                manifest_hits.append((file, manifest))
    if not manifest_hits:
        record("cấu hình (config_sha)", "NOT_RUN",
               "không có stage_manifest run_all nào trỏ tới run-id này")
    for file, manifest in manifest_hits:
        overlay = manifest.get("overlay") or {}
        man_profile = manifest.get("profile") or profile
        if not man_profile:
            record(f"cấu hình {file.stem}", "NOT_RUN", "manifest không ghi profile")
            continue
        cfg = load_config(man_profile, extra=overlay or None)
        sha = sha256_text(json.dumps(cfg, ensure_ascii=False, sort_keys=True, indent=2))
        if sha == manifest.get("config_sha"):
            record(f"cấu hình {file.stem}", "PASS",
                   f"config_sha khớp ({sha[:8]}, profile={man_profile})")
        else:
            record(f"cấu hình {file.stem}", "FAIL",
                   f"config_sha lệch: bản ghi {str(manifest.get('config_sha'))[:8]} "
                   f"vs hiện tại {sha[:8]}")

    # -------------------------------------------------------- predictions subset
    p7_rows = [row for row in rows if row.get("stage") == "P7_baselines"]
    parquet = next((path for path in artifacts if path.endswith(".parquet")
                    and "/predictions/" in path.replace("\\", "/")), None)
    if args.mode != "eval":
        record("tái tạo subset predictions", "NOT_RUN", f"mode={args.mode} không hỗ trợ")
    elif not p7_rows:
        record("tái tạo subset predictions", "NOT_RUN",
               f"run không phải P7_baselines (stage={rows[-1].get('stage')}) — "
               f"chỉ kiểm tra hash/cấu hình")
    elif parquet is None or p7_rows[-1].get("seed") is None:
        record("tái tạo subset predictions", "NOT_RUN",
               "P7 row thiếu artifact predictions hoặc thiếu seed")
    else:
        row = p7_rows[-1]
        split = row.get("split") or "val_strict"
        seed = int(row["seed"])
        limit = row.get("limit")
        dev = bool(row.get("dev"))
        if not profile:
            record("tái tạo subset predictions", "FAIL",
                   "không suy ra được profile từ registry row")
        else:
            cmd = [PYTHON, "-m", "src.pipeline.eval_baselines", "--profile", profile,
                   "--split", split, "--seed", str(seed), "--json"]
            if limit:
                cmd += ["--limit", str(limit)]
            if dev:
                cmd += ["--dev"]
            if args.config:
                cmd += ["--config", args.config]
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"eval_{stamp}.log"
            print(f"  $ {shlex.join(cmd)}")
            print(f"    (log: {log_path.relative_to(REPO)}; "
                  f"nạp checkpoint/index đã lưu — không train lại)")
            with open(log_path, "w", encoding="utf-8") as log:
                proc = subprocess.run(cmd, cwd=str(REPO), stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True)
                log.write(proc.stdout or "")
            if proc.returncode != 0:
                record("tái tạo subset predictions", "FAIL",
                       f"eval_baselines rc={proc.returncode}; xem {log_path.name}")
            else:
                try:
                    summary = json.loads(proc.stdout)
                except json.JSONDecodeError:
                    summary = None
                new_path = ((summary or {}).get("predictions") or {}).get("path")
                if not new_path or not Path(new_path).is_file():
                    record("tái tạo subset predictions", "FAIL",
                           "eval_baselines không ghi parquet mới")
                else:
                    verdict, detail = compare_subset(parquet, new_path, args.subset)
                    record("tái tạo subset predictions", verdict, detail)

    # ------------------------------------------------------------------ verdict
    fails = [item for item in CHECKS if item[1] == "FAIL"]
    not_runs = [item for item in CHECKS if item[1] == "NOT_RUN"]
    print("-" * 79)
    overall = "FAIL" if fails else "PASS"
    print(f"KẾT QUẢ: {overall} ({len(CHECKS) - len(fails) - len(not_runs)} PASS, "
          f"{len(fails)} FAIL, {len(not_runs)} NOT_RUN)")
    if args.json:
        print(json.dumps({"run_id": args.run_id, "mode": args.mode, "overall": overall,
                          "checks": [{"name": n, "verdict": v, "detail": d}
                                     for n, v, d in CHECKS]}, ensure_ascii=False, indent=2))
    return 1 if fails else 0


def compare_subset(stored_path: str, new_path: str, subset: int) -> tuple[str, str]:
    """Compare the overlapping rows of two predictions parquets (subset of qids)."""
    import pyarrow.parquet as pq

    stored = {(r[KEY[0]], int(r[KEY[1]]), r[KEY[2]]): r
              for r in pq.read_table(stored_path).to_pylist()}
    fresh = {(r[KEY[0]], int(r[KEY[1]]), r[KEY[2]]): r
             for r in pq.read_table(new_path).to_pylist()}
    shared = sorted(set(stored) & set(fresh))
    if not shared:
        return "FAIL", (f"không có dòng (exp_id, seed, qid) chung: "
                        f"stored={len(stored)}, mới={len(fresh)}")
    selected = shared[:max(1, int(subset))]
    mismatches = []
    for key in selected:
        old, new = stored[key], fresh[key]
        for field in EXACT_FIELDS + FLOAT_FIELDS:
            if field not in old or field not in new:
                continue
            a, b = old[field], new[field]
            if field in FLOAT_FIELDS:
                if a is None or b is None:
                    same = a is b
                else:
                    same = abs(float(a) - float(b)) <= TOLERANCE
            else:
                same = a == b
            if not same:
                mismatches.append(f"{key} {field}: {a!r} != {b!r}")
    if mismatches:
        detail = (f"{len(mismatches)} lệch trong {len(selected)} dòng chung "
                  f"(stored {Path(stored_path).name} vs {Path(new_path).name})")
        for item in mismatches[:10]:
            print(f"        - {item}")
        return "FAIL", detail
    return "PASS", (f"{len(selected)}/{len(shared)} dòng chung khớp "
                    f"(chính xác phân loại, sai số <= {TOLERANCE} cho số thực)")


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        print("reproduce: bị ngắt (KeyboardInterrupt)", file=sys.stderr)
        raise SystemExit(130)
PY
