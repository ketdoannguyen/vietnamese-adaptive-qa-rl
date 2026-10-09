# Resume plan — D012 rebuild chain → P9 (2026-10-09)

## State on entry (verified from filesystem)

- P0/P1(v2)/P2(v2)/P3(reader medium + reselect)/P5(teacher smoke, laya GPU smoke) DONE.
- `p2-retrieval-medium-2026-10-08T170359+0000` was RUNNING with a dead process → marked
  `INTERRUPTED` (spec 06 §5), then re-run: `p2-retrieval-medium-2026-10-09T002621+0000`
  SUCCEEDED (train/val 0 re-ranked = cached v2 rows from the interrupted run; test 4260
  re-ranked). Recall@1 bm25 0.8539 / dense 0.9330 (was 0.9789 under v1) → hard negatives
  landed (D012 acceptance signal).
- `p3-reader-medium-reselect-2026-10-09T002942+0000` SUCCEEDED: threshold re-confirmed 0.5,
  feasible 25/33, weights untouched.

## Remaining chain (D012 + D013 order)

1. **P3 cache_reader** (`--profile medium`) — state cache over v2 rankings, baking the
   re-confirmed 0.5 threshold. *In flight*: `p3-reader-cache-medium-2026-10-09T002x`.
2. **P5 make_teacher** (`--profile medium`) — rebuild teacher JSONL + diagnostics.
   Acceptance: SEARCH share of train labels rises vs D012 baseline (SEARCH_BM25 19 /
   SEARCH_DENSE 10 of 34,993; 0.08%); questions with deeper-only correct candidate > 20/4999.
3. **Laya CE + RLCD, medium** — same config for both (spec 04 §5). Before committing to
   3 epochs × 35k states, run a measured profile slice (500 rows, 1 epoch) to log step time /
   peak VRAM; if projected CE+RLCD walltime > `limits.max_gpu_hours` (12 h) for the whole
   remaining chain, reduce epochs with the reduction recorded in DECISIONS.md as D-new.
4. **P5 cache_laya ×2** (`--regime ce`, `--regime rlcd`; per-regime dirs per D019) — each as
   soon as its checkpoint lands (CE ∥ RLCD now run in parallel, D018); the rlcd cache is the
   mainline frozen prior, the ce cache backs the E04/E07 baseline rows.
5. **P6 train_rl × seeds 42/43/44** — masked REINFORCE, best-by-val checkpoint; every run
   archives to `outputs/models/rl_runs/{run_id}/` (D016). CPU-only; can overlap B5.
   Then **P6 select_rl** (`python -m src.pipeline.select_rl`) — freezes the pre-registered
   rule winner (highest val_reward, tie-break lowest seed) into `best_selection.json`,
   swaps the canonical `models/rl_policy` to the winner, writes the freeze LAST.
6. **B5 train_correctness** — `--oof-folds 3` group-disjoint reader refit + correctness
   estimator (spec 08, D015). GPU heavy; overlaps the Laya block only within the D018
   headroom rule.
7. **P7 eval_baselines val** → **cache_reader --splits test_strict** (data-only state cache,
   strictly after the step-5 freeze, D019) → **eval_baselines test** (single access, logged).
8. **P9 report** — REPORT.md/txt, results.csv/json, model card, REPRODUCE, handoff, zip;
   includes the full training history (Bảng 1b/1c per `run_id` + loss figures) per D021.

## Hard rules carried

- GPU stages: one process at a time was D001/D017; **superseded by D018 (user-authorized,
  2026-10-09)**: CE ∥ RLCD run in parallel on the 24 GiB L4 with a ≥4 GiB headroom rule;
  models loaded/unloaded phase by phase; two runs never write the same output directory.
- Nothing hand-typed into reports; unrun stages stay `NOT_RUN`.
- Test split opened once, after `best_selection.json`; `docs/TEST_ACCESS_LOG.md` written by P9.
- Every deviation goes to `docs/DECISIONS.md` with measured before/after.