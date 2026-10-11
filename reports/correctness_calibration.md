# B5 correctness calibration — B5

- run: `b5-correctness-medium-42-2026-10-09T080314+0000` (profile `medium`, seed 42)
- variant: **B5** (calibrated probabilities: yes)
- deployed reader: `f05b24ee68f2e6bb…` (P3 full-train reader; frozen for every baseline)
- OOF folds: 3 (3 fold reader(s) trained for this run)
- report generated: 2026-10-09T08:32:12+00:00

## Pre-registered threshold criterion

> tau_answer = argmax over the candidate grid of mean_episode_reward on val_model_selection, replaying the deterministic B5 rule (ANSWER iff the action mask allows it and score >= tau; else SEARCH_BM25, then SEARCH_DENSE while budget remains; else ABSTAIN) through the shared environment/reward of src.pipeline.worlds; ties are broken toward the higher tau; the never-answer sentinel is flagged if it wins

Selected `tau_answer = 1.0000` on `val_model_selection` (997 episodes, coverage 0.3190, mean episode reward -0.8024).

## Data

- training rows (out-of-fold, `train_strict` only): 25691 states with a non-empty candidate (6100 correct / 19591 incorrect)
- excluded ambiguous negatives: 0 (no defensible gold label; excluded from the fit only)
- `val_calibration`: 19912 states (6342 positive), `val_model_selection`: 5111 states
- `test_strict` was not read at any point of this stage.

## Calibration decision

`correctness.calibration_plan` on the val-calibration rows chose **isotonic** (rows=19912, positives=6342, negatives=13570).

Calibrator fitted on `val_calibration` only: mode `isotonic`. Reported below on the held-out `val_model_selection` rows.

## Reliability (Brier / ECE) — val_model_selection

- Brier: 0.3680
- ECE (10 bins, L1): 0.3395
- rows: 5111

| bin | count | confidence | accuracy |
| --- | --- | --- | --- |
| [0.00, 0.10) | 1856 | 0.0039 | 0.2074 |
| [0.10, 0.20) | 555 | 0.1811 | 0.1892 |
| [0.20, 0.30) | 248 | 0.2617 | 0.2903 |
| [0.30, 0.40) | 201 | 0.3685 | 0.2189 |
| [0.40, 0.50) | 180 | 0.4582 | 0.2611 |
| [0.50, 0.60) | 135 | 0.5354 | 0.2593 |
| [0.60, 0.70) | 6 | 0.6296 | 0.1667 |
| [0.70, 0.80) | 0 | N/A | N/A |
| [0.80, 0.90) | 41 | 0.8148 | 0.4878 |
| [0.90, 1.00) | 1889 | 1.0000 | 0.3504 |

## Selective prediction

- risk at 80% coverage (calibrated ranking, spec 03 §1): 0.7155 over 4088/5111 states
- B5 at `tau=1.0000`: coverage 0.3190, selective risk 0.6635 (answered 318/997 episodes)
- mean episode reward -0.8024 [-0.9498; -0.5711] (cluster bootstrap over 6 title groups)

| coverage | risk |
| --- | --- |
| 0.00 | 0.0000 |
| 0.05 | 0.5148 |
| 0.11 | 0.6716 |
| 0.16 | 0.6943 |
| 0.21 | 0.6797 |
| 0.26 | 0.6679 |
| 0.32 | 0.6625 |
| 0.37 | 0.6513 |
| 0.42 | 0.6558 |
| 0.47 | 0.6689 |
| 0.53 | 0.6732 |
| 0.58 | 0.6854 |
| 0.63 | 0.6970 |
| 0.68 | 0.6972 |
| 0.74 | 0.7124 |
| 0.79 | 0.7135 |
| 0.84 | 0.7203 |
| 0.89 | 0.7280 |
| 0.95 | 0.7317 |
| 1.00 | 0.7318 |

## Subgroups (val_model_selection states)

| subgroup | n | positives | positive rate | mean score |
| --- | --- | --- | --- | --- |
| all | 5111 | 1371 | 0.2682 | 0.4554 |
| answerable | 3831 | 1371 | 0.3579 | 0.4563 |
| unanswerable | 1280 | 0 | 0.0000 | 0.4529 |
| provenance_ok | 5111 | 1371 | 0.2682 | 0.4554 |
| provenance_invalid | 0 | 0 | N/A | N/A |

## Threshold selection log

| tau | mean episode reward | coverage | answered | selective risk |
| --- | --- | --- | --- | --- |
| 0.0000 | -0.9784 | 0.7492 | 747 | 0.7309 |
| 0.1730 | -0.8854 | 0.5095 | 508 | 0.7047 |
| 0.1994 | -0.8688 | 0.4644 | 463 | 0.6955 |
| 0.3267 | -0.8162 | 0.3942 | 393 | 0.6692 |
| 0.4918 | -0.8210 | 0.3551 | 354 | 0.6723 |
| 1.0000 | -0.8024 | 0.3190 | 318 | 0.6635 |
| 2.0000 | -0.8324 | 0.0000 | 0 | N/A |

## Caveats

- Reader optimism: fitting rows come from readers trained without their feature questions (3 OOF fold(s)); the deployed reader is the P3 full-train reader, so its predictions on those questions are never used as training labels.
- The deployed reader's null threshold was tuned by P3 on all of `val_strict`; `val_model_selection` therefore inherits that single scalar. Calibration and tau selection below are still fitted/evaluated only on their own val subsets.
- 4 train question(s) are flagged ambiguous negatives; their states are excluded from the fit because their gold is not defensible (same rule as the P6 training reward), but they stay in validation.
- Subgroup rows use gold only for measurement; no gold enters any feature.
- 4 val question(s) are ambiguous negatives and are kept in validation by design.

- results row: `/home/nkd/training-code/vietnamese-adaptive-qa-rl/reports/results_b5_row.json` (results_b5_row_json)
