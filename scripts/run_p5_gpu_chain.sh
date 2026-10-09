#!/usr/bin/env bash
# Recovery chain (2026-10-09): the P5 Laya CE medium run was killed mid-epoch-1 with no
# resume support (registry run p5-laya-ce-medium-2026-10-09T010137+0000, marked INTERRUPTED),
# so the two Laya losses and the frozen-probability cache must be rebuilt from scratch.
#
# Runs the three remaining GPU stages of the medium chain serially — one GPU process at a
# time (D001) — each stage writing its own RUNNING/terminal registry rows as usual:
#   1. train_laya --loss soft-ce  (2 epochs, configs/laya_budget_2ep.yaml per D015)
#   2. train_laya --loss rlcd     (identical conditions: same data/seed/epochs, spec 04 §5)
#   3. cache_laya                 (regime defaults to laya.frozen = rlcd)
# `set -e` stops the chain on the first failure; earlier stages' checkpoints stay untouched.
#
# Intended to be started detached (setsid nohup) from the repo root:
#   setsid nohup bash scripts/run_p5_gpu_chain.sh < /dev/null >/dev/null 2>&1 &
# Progress: tail -f outputs/runs/logs/p5_laya_ce_medium_r2.log (then _rlcd_, then cache_)
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
LOGS=outputs/runs/logs
OVERLAY=configs/laya_budget_2ep.yaml

echo "chain start $(date -u +%FT%TZ) pid $$"
$PY -m src.pipeline.train_laya --profile medium --loss soft-ce --config "$OVERLAY" \
  >"$LOGS/p5_laya_ce_medium_r2.log" 2>&1
echo "soft-ce done $(date -u +%FT%TZ)"
$PY -m src.pipeline.train_laya --profile medium --loss rlcd --config "$OVERLAY" \
  >"$LOGS/p5_laya_rlcd_medium_r2.log" 2>&1
echo "rlcd done $(date -u +%FT%TZ)"
$PY -m src.pipeline.cache_laya --profile medium \
  >"$LOGS/p5_cache_laya_rlcd_medium_r2.log" 2>&1
echo "chain end $(date -u +%FT%TZ)"
