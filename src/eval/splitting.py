"""Title-group splitting and cluster bootstrap for the P7 evaluation stage.

The strict splits are partitioned by ``title_group`` (P1, `02_IMPLEMENTATION/02_...`), so
every internal split this stage creates must be group-disjoint too: near-duplicate questions
from one article never straddle a fold boundary. Confidence intervals resample **groups**,
not questions, for the same reason (spec `03_EXPERIMENTS/01_...` §4).

Everything here is pure — it takes ``qid -> title_group`` maps from the label store's public
``title_group`` column and returns index lists or intervals.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.data import artifacts
from src.utils.hashing import stable_seed


def group_map(groups: Mapping[str, str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for qid, group in groups.items():
        grouped.setdefault(str(group), []).append(str(qid))
    return {group: sorted(qids) for group, qids in grouped.items()}


def title_groups(cfg: Mapping[str, Any], split: str) -> dict[str, str]:
    """``qid -> title_group`` from the questions parquet (the label store's public column)."""
    from src.utils.config import resolve

    rows = artifacts.questions_by_qid(
        artifacts.read_questions(resolve(cfg, "data_processed"), split))
    groups = {qid: str(row.get("title_group")) for qid, row in rows.items()}
    missing = sorted(qid for qid, group in groups.items() if not group or group == "None")
    if missing:
        raise ValueError(f"{split}: {len(missing)} question(s) without a title_group, "
                         f"e.g. {missing[:3]}; group-disjoint splitting is not possible")
    return groups


def group_folds(groups: Mapping[str, str], n_folds: int, *, seed: int) -> list[list[str]]:
    """Partition qids into ``n_folds`` title-disjoint folds of near-equal size.

    Group order is a stable hash of ``(seed, group)``; each group goes to the currently
    smallest fold. Both choices are reproducible across processes and machines.
    """
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")
    grouped = group_map(groups)
    if len(grouped) < n_folds:
        raise ValueError(f"{len(grouped)} title group(s) cannot be split into {n_folds} folds")
    order = sorted(grouped, key=lambda group: stable_seed(seed, group))
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    sizes = [0] * n_folds
    for group in order:
        qids = grouped[group]
        target = min(range(n_folds), key=lambda i: (sizes[i], i))
        folds[target].extend(qids)
        sizes[target] += len(qids)
    return [sorted(fold) for fold in folds]


def group_holdout(groups: Mapping[str, str], *, fraction: float, seed: int
                  ) -> tuple[list[str], list[str]]:
    """``(train_qids, holdout_qids)`` with holdout ~``fraction`` of the qids, title-disjoint.

    At least one group always stays in train, so the holdout can never swallow the split.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    grouped = group_map(groups)
    if len(grouped) < 2:
        raise ValueError("holding out a group requires at least 2 title groups")
    total = sum(len(qids) for qids in grouped.values())
    target = max(1, int(round(fraction * total)))
    order = sorted(grouped, key=lambda group: stable_seed(seed, group))
    holdout: list[str] = []
    for group in order[:-1]:  # the last group is reserved for train
        if len(holdout) >= target:
            break
        holdout.extend(grouped[group])
    holdout_set = set(holdout)
    train = sorted(qid for qids in grouped.values() for qid in qids if qid not in holdout_set)
    if not holdout or not train:
        raise ValueError(f"group holdout left an empty side (holdout={len(holdout)}, "
                         f"train={len(train)})")
    return train, sorted(holdout)


def holdout_specs(groups: Mapping[str, str], oof_folds: int, *, seed: int,
                  fraction: float = 0.25) -> list[tuple[list[str], list[str]]]:
    """``(reader_train_qids, feature_qids)`` per fold, for the B5 out-of-fold procedure.

    ``oof_folds == 1`` is the budget option of spec 08 §3: one group-disjoint holdout whose
    reader is retrained on the reduced train set. ``oof_folds >= 2`` is full cross-fitting:
    fold ``i``'s features come from a reader trained on the other folds.
    """
    if oof_folds < 1:
        raise ValueError(f"oof_folds must be >= 1, got {oof_folds}")
    if oof_folds == 1:
        train, holdout = group_holdout(groups, fraction=fraction, seed=seed)
        return [(train, holdout)]
    folds = group_folds(groups, oof_folds, seed=seed)
    specs = []
    for i, fold in enumerate(folds):
        rest = sorted(qid for j, other in enumerate(folds) if j != i for qid in other)
        specs.append((rest, fold))
    return specs


def _cluster_means(values: Mapping[str, float], groups: Mapping[str, str],
                   n_resamples: int, seed: int) -> np.ndarray:
    by_group: dict[str, list[float]] = {}
    for qid, value in values.items():
        by_group.setdefault(groups[qid], []).append(float(value))
    names = sorted(by_group)
    stats = np.array([by_group[group] for group in names], dtype=object)
    rng = np.random.default_rng(stable_seed(seed, "cluster_bootstrap", n_resamples))
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        picked = rng.integers(0, len(names), size=len(names))
        pooled: list[float] = []
        for j in picked:
            pooled.extend(stats[j])
        means[i] = float(np.mean(pooled))
    return means


def cluster_bootstrap_ci(values: Mapping[str, float], groups: Mapping[str, str], *,
                         n_resamples: int = 1000, seed: int = 42,
                         alpha: float = 0.05) -> dict[str, Any]:
    """CI for a mean of per-qid values, resampling title groups with replacement.

    A single group cannot support an interval claim: the spread is then a within-group
    statistic, so the result is flagged ``single_group_warning`` instead of being reported
    as a robust interval (spec 01 §8).
    """
    if not values:
        raise ValueError("cluster bootstrap needs at least one observation")
    unknown = sorted(set(values) - set(groups))
    if unknown:
        raise ValueError(f"{len(unknown)} qid(s) have no title group, e.g. {unknown[:3]}")
    point = float(np.mean(list(values.values())))
    n_groups = len({groups[qid] for qid in values})
    single = n_groups < 2
    means = _cluster_means(values, groups, n_resamples, seed)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": point, "ci_low": float(lo), "ci_high": float(hi),
            "n_resamples": int(n_resamples), "n_groups": n_groups, "alpha": float(alpha),
            "basis": "single_group_warning" if single else "cluster_title_group"}


def paired_cluster_bootstrap_ci(diff_by_qid: Mapping[str, float],
                                groups: Mapping[str, str], *, n_resamples: int = 1000,
                                seed: int = 42, alpha: float = 0.05) -> dict[str, Any]:
    """CI for a **paired** per-qid difference (same qids on both sides), grouped resampling."""
    return cluster_bootstrap_ci(diff_by_qid, groups, n_resamples=n_resamples, seed=seed,
                                alpha=alpha)
