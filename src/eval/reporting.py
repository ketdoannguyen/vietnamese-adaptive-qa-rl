"""Baseline aggregation, the flat results CSV and the three P7 plots.

Metric definitions come from `03_EXPERIMENTS/01_METRICS_CALIBRATION_AND_STATISTICS.md`:
overall EM/F1 count an abstention as 0 (``answerable_*`` and the negative metrics are the
subgroup views); ``coverage`` is ANSWER over all episodes; ``selective_risk`` has the
answered subset as its denominator and is ``N/A`` when coverage is 0 (never a fake 0);
``risk_at_80pct_coverage`` ranks by the *calibrated* confidence only and is ``N/A`` for any
policy without one. Missing values stay ``None`` and are written as ``N/A`` — never as 0.

Confidence intervals resample title groups (:func:`src.eval.splitting.cluster_bootstrap_ci`)
because duplicated articles would otherwise be counted as independent evidence.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.eval import correctness, splitting
from src.eval.rollout import EpisodeRecord
from src.utils.config import resolve

#: Flat columns of ``reports/tables/baseline_results_{val,test}.csv``, in write order.
RESULTS_COLUMNS: tuple[str, ...] = (
    "run_id", "exp_id", "policy", "policy_variant", "prior_regime", "split", "dev", "seed",
    "status", "reason", "tau",
    "n_episodes", "n_answerable", "n_unanswerable",
    "exact_match", "answered_em", "token_f1", "answerable_answer_f1", "answerable_em",
    "unanswerable_abstain_accuracy", "false_answer_rate_on_unanswerables",
    "coverage", "selective_risk", "risk_at_80pct_coverage",
    "mean_episode_reward", "std_episode_reward",
    "avg_optional_retrievals", "avg_total_retrievals", "avg_bm25_calls", "avg_dense_calls",
    "avg_retrieval_cost", "forced_terminal_rate", "success_per_cost",
    "action_count_answer", "action_count_bm25", "action_count_dense", "action_count_abstain",
    "brier", "ece", "calibration",
    "clean_n", "clean_token_f1", "clean_mean_episode_reward",
    "ambiguous_n", "ambiguous_abstain_rate", "ambiguous_false_answer_rate",
    "f1_ci_low", "f1_ci_high", "reward_ci_low", "reward_ci_high",
    "cost_ci_low", "cost_ci_high", "ci_basis", "ci_groups", "ci_n_resamples",
)

#: Row labels for the ``status`` column.
STATUS_SUCCEEDED, STATUS_FAILED, STATUS_SKIPPED = "SUCCEEDED", "FAILED", "SKIPPED"

_NA_METRICS: dict[str, Any] = {column: None for column in RESULTS_COLUMNS
                               if column not in ("run_id", "exp_id", "policy", "policy_variant",
                                                 "prior_regime", "split", "dev", "seed", "status",
                                                 "reason", "tau")}


def empty_metrics() -> dict[str, Any]:
    """All metric columns ``N/A`` — the shape of a FAILED/SKIPPED row."""
    return dict(_NA_METRICS)


def _finite_mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if len(values) else None


def aggregate(episodes: Sequence[EpisodeRecord], *, groups: Mapping[str, str] | None = None,
              ambiguous_qids: frozenset[str] = frozenset(), calibrated: bool = False,
              ci_seed: int = 42, n_resamples: int = 1000) -> dict[str, Any]:
    """Every metric of one result row, from that row's episodes. ``None`` means ``N/A``."""
    metrics = empty_metrics()
    if not episodes:
        return metrics
    answerable = [episode for episode in episodes if not episode.eval_is_impossible]
    unanswerable = [episode for episode in episodes if episode.eval_is_impossible]
    answered = [episode for episode in episodes if episode.answered]
    returns = [episode.episode_return for episode in episodes]
    action_counts = Counter(step.action for episode in episodes for step in episode.steps)

    metrics.update({
        "n_episodes": len(episodes),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "exact_match": _finite_mean([e.eval_correctness_em for e in episodes]),
        "token_f1": _finite_mean([e.eval_correctness_f1 for e in episodes]),
        "answered_em": _finite_mean([e.eval_correctness_em for e in answered]),
        "answerable_answer_f1": _finite_mean([e.eval_correctness_f1 for e in answerable]),
        "answerable_em": _finite_mean([e.eval_correctness_em for e in answerable]),
        "unanswerable_abstain_accuracy":
            _finite_mean([float(e.abstained) for e in unanswerable]),
        "false_answer_rate_on_unanswerables":
            _finite_mean([float(e.answered) for e in unanswerable]),
        "coverage": len(answered) / len(episodes),
        "selective_risk": 1.0 - _finite_mean([e.eval_correct for e in answered])
        if answered else None,
        "mean_episode_reward": _finite_mean(returns),
        "std_episode_reward": float(np.std(returns)),
        "avg_optional_retrievals": _finite_mean([e.optional_used for e in episodes]),
        "avg_total_retrievals": _finite_mean([e.bm25_calls + e.dense_calls for e in episodes]),
        "avg_bm25_calls": _finite_mean([e.bm25_calls for e in episodes]),
        "avg_dense_calls": _finite_mean([e.dense_calls for e in episodes]),
        "avg_retrieval_cost": _finite_mean([e.retrieval_cost for e in episodes]),
        "forced_terminal_rate": _finite_mean([float(e.forced_terminal) for e in episodes]),
        "action_count_answer": int(action_counts.get(0, 0)),
        "action_count_bm25": int(action_counts.get(1, 0)),
        "action_count_dense": int(action_counts.get(2, 0)),
        "action_count_abstain": int(action_counts.get(3, 0)),
    })
    mean_reward = metrics["mean_episode_reward"]
    avg_cost = metrics["avg_retrieval_cost"]
    metrics["success_per_cost"] = (mean_reward / avg_cost) if avg_cost else None

    _add_calibration(metrics, episodes, calibrated=calibrated)
    _add_subgroups(metrics, episodes, ambiguous_qids)
    _add_intervals(metrics, episodes, groups, seed=ci_seed, n_resamples=n_resamples)
    return metrics


def _add_calibration(metrics: dict[str, Any], episodes: Sequence[EpisodeRecord], *,
                     calibrated: bool) -> None:
    """Brier/ECE + risk-at-80%-coverage. Only a genuinely calibrated p_correct qualifies.

    The target label is "the episode answered correctly" with an abstention counted
    incorrect (the same convention as overall EM/F1, spec 01 §1); ``p_correct`` is the
    estimator's value at the final decision state, so unanswered episodes enter with their
    (usually low) confidence rather than a fabricated one.
    """
    scores = [episode.confidence_p_correct for episode in episodes]
    if not calibrated or any(score is None for score in scores):
        metrics["calibration"] = "none"
        return
    values = [float(score) for score in scores]
    labels = [int(episode.eval_correct) for episode in episodes]
    reliability = correctness.reliability_metrics(values, labels)
    selection = correctness.selection_metrics(values, labels, coverage_fraction=0.8)
    metrics.update({"brier": reliability["brier"], "ece": reliability["ece"],
                    "risk_at_80pct_coverage": selection["risk_at_coverage"],
                    "calibration": "calibrated_p_correct"})


def _add_subgroups(metrics: dict[str, Any], episodes: Sequence[EpisodeRecord],
                   ambiguous_qids: frozenset[str]) -> None:
    """Clean vs ambiguous reporting (spec 08 §5, 00 §2): ambiguous negatives stay in val
    and are reported on their own so their noisier labels cannot flatter a policy."""
    clean = [episode for episode in episodes if episode.qid not in ambiguous_qids]
    ambiguous = [episode for episode in episodes if episode.qid in ambiguous_qids]
    metrics.update({
        "clean_n": len(clean),
        "clean_token_f1": _finite_mean([e.eval_correctness_f1 for e in clean]),
        "clean_mean_episode_reward": _finite_mean([e.episode_return for e in clean]),
        "ambiguous_n": len(ambiguous),
        "ambiguous_abstain_rate": _finite_mean([float(e.abstained) for e in ambiguous]),
        "ambiguous_false_answer_rate": _finite_mean([float(e.answered) for e in ambiguous]),
    })


def _add_intervals(metrics: dict[str, Any], episodes: Sequence[EpisodeRecord],
                   groups: Mapping[str, str] | None, *, seed: int, n_resamples: int) -> None:
    """Per-row 95% cluster-bootstrap CIs; a limitation is recorded when they are impossible."""
    fields = ("f1_ci_low", "f1_ci_high", "reward_ci_low", "reward_ci_high",
              "cost_ci_low", "cost_ci_high")
    basis, n_groups = None, None
    if groups is None:
        basis = "N/A: title groups unavailable for this split"
    elif len(episodes) < 2:
        basis = "N/A: fewer than two episodes"
    else:
        columns = {
            ("f1_ci_low", "f1_ci_high"): {e.qid: e.eval_correctness_f1 for e in episodes},
            ("reward_ci_low", "reward_ci_high"): {e.qid: e.episode_return for e in episodes},
            ("cost_ci_low", "cost_ci_high"): {e.qid: e.retrieval_cost for e in episodes},
        }
        for (low_field, high_field), values in columns.items():
            try:
                interval = splitting.cluster_bootstrap_ci(values, groups,
                                                          n_resamples=n_resamples, seed=seed)
            except (KeyError, ValueError) as exc:
                basis = f"N/A: {type(exc).__name__}: {exc}"
                break
            metrics[low_field], metrics[high_field] = interval["ci_low"], interval["ci_high"]
            basis, n_groups = interval["basis"], interval["n_groups"]
    metrics["ci_basis"] = basis or "cluster_title_group"
    metrics["ci_groups"] = n_groups
    metrics["ci_n_resamples"] = n_resamples
    for field in fields:
        metrics.setdefault(field, None)
    return


def format_cell(value: Any) -> str:
    """CSV cell: ``None``/non-finite -> ``N/A``; never a silent zero (spec 01 §1)."""
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return "N/A" if not math.isfinite(value) else f"{value:.6f}"
    if isinstance(value, (int, str)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def write_results_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write every row of the run — SUCCEEDED, FAILED and SKIPPED alike — as flat columns."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = sorted({column for row in rows for column in RESULTS_COLUMNS
                      if column not in row})
    if missing:
        raise ValueError(f"result row(s) are missing column(s) {missing}")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(RESULTS_COLUMNS)
        for row in rows:
            writer.writerow([format_cell(row[column]) for column in RESULTS_COLUMNS])
    return path


def retrieval_recall(worlds: Sequence[Any], *, k_max: int = 5) -> dict[str, Any]:
    """Gold-document recall@k per tool (positive-only) over one split's cached rankings.

    ``union`` counts a hit when either tool surfaces the gold document within k. A gold
    document outside the cached top-n counts as a miss — this measures the ranking the
    agent can actually reach, not an ideal retriever.
    """
    ks = list(range(1, k_max + 1))
    ranks: dict[str, list[int | None]] = {"bm25": [], "dense": [], "union": []}
    for world in worlds:
        gold = world.private.gold_doc_id
        if gold is None:
            continue  # positive-only, spec 01 §1
        per_tool = {}
        for tool in ("bm25", "dense"):
            ranking = list(world.rankings.get(tool, ()))
            per_tool[tool] = ranking.index(gold) + 1 if gold in ranking else None
        ranks["bm25"].append(per_tool["bm25"])
        ranks["dense"].append(per_tool["dense"])
        hits = [rank for rank in per_tool.values() if rank is not None]
        ranks["union"].append(min(hits) if hits else None)
    n = len(ranks["bm25"])
    recall: dict[str, Any] = {"n_answerable": n, "k": ks}
    for tool in ("bm25", "dense", "union"):
        recall[tool] = [float(np.mean([r is not None and r <= k for r in ranks[tool]]))
                        if n else None for k in ks]
    return recall


# ------------------------------------------------------------------------------- plots

def _plot_target(cfg: Mapping[str, Any]) -> Path:
    plots = resolve(cfg, "plots")
    plots.mkdir(parents=True, exist_ok=True)
    return plots


def write_plots(cfg: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
                episodes_by_row: Mapping[tuple[str, int], Sequence[EpisodeRecord]], *,
                split: str, recall: Mapping[str, Any]) -> dict[str, Path]:
    """The three P7 plots (spec 00 §7). The RL learning curve and action histogram belong
    to ``train_rl.py`` and are deliberately not written here."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = _plot_target(cfg)
    completed = [row for row in rows if row["status"] == STATUS_SUCCEEDED]
    scatter_path = plots / "answer_f1_vs_retrieval_cost.png"
    risk_path = plots / "risk_coverage.png"
    recall_path = plots / "retrieval_recall.png"

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for row in completed:
        x, y = row["avg_retrieval_cost"], row["token_f1"]
        if x is None or y is None:
            continue
        ax.scatter(x, y, s=28)
        label = row["exp_id"] if row["exp_id"] != "B0" else f"B0@{row['seed']}"
        ax.annotate(label, (x, y), fontsize=7, xytext=(4, 3), textcoords="offset points")
    if not any(row["avg_retrieval_cost"] is not None for row in completed):
        ax.text(0.5, 0.5, "N/A — no completed row", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlabel("mean retrieval cost per episode (reward units; includes forced bootstrap)")
    ax.set_ylabel("mean token F1 over all episodes (abstained = 0)")
    ax.set_title(f"P7 cost–quality — {split}")
    ax.grid(alpha=0.25)
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    drew_curve = False
    for row in completed:
        episodes = episodes_by_row.get((row["exp_id"], int(row["seed"])), ())
        scores = [episode.confidence_p_correct for episode in episodes]
        if row.get("calibration") != "calibrated_p_correct" or not episodes \
                or any(score is None for score in scores):
            continue
        order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
        labels = np.asarray([episode.eval_correct for episode in episodes], dtype=np.float64)
        coverage, risk = [], []
        for k in range(1, len(order) + 1):
            coverage.append(k / len(order))
            risk.append(1.0 - float(labels[order[:k]].mean()))
        ax.plot(coverage, risk, linewidth=1.4,
                label=f"{row['exp_id']} (calibrated p_correct)")
        drew_curve = True
    for row in completed:
        coverage, risk = row["coverage"], row["selective_risk"]
        if coverage is None or risk is None:
            continue
        ax.scatter([coverage], [risk], marker="x", s=40,
                   label=None if drew_curve else None)
        ax.annotate(row["exp_id"], (coverage, risk), fontsize=7, xytext=(4, -8),
                    textcoords="offset points")
    if not drew_curve:
        ax.text(0.5, 0.5, "N/A — no row carries a calibrated p_correct\n"
                          "(x marks each row's operating point)",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
    ax.set_xlabel("coverage (fraction of episodes answered)")
    ax.set_ylabel("selective risk (1 − accuracy on answered)")
    ax.set_title(f"P7 risk–coverage — {split}")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.25)
    if drew_curve:
        ax.legend(fontsize=8)
    fig.savefig(risk_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    if recall["n_answerable"]:
        for tool in ("bm25", "dense", "union"):
            ax.plot(recall["k"], recall[tool], marker="o", linewidth=1.4, label=tool)
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "N/A — no answerable world in this split", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlabel("k (passages retrieved by the tool)")
    ax.set_ylabel("gold-doc recall@k (answerable only)")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"P7 retrieval recall — {split} (n={recall['n_answerable']} answerable)")
    ax.grid(alpha=0.25)
    fig.savefig(recall_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"answer_f1_vs_retrieval_cost": scatter_path, "risk_coverage": risk_path,
            "retrieval_recall": recall_path}


# ------------------------------------------------------------------- training history

#: The Laya trainer's per-step log line, e.g. ``epoch 1/2 step 13000 loss 0.0009``.
LAYA_LOSS_RE = re.compile(r"^epoch (\d+)/(\d+) step (\d+) loss (-?[\d.eE+]+)\s*$")
#: HF Trainer's per-log-step dict from the reader stage, e.g.
#: ``{'loss': '5.846', 'grad_norm': '5.815', 'learning_rate': '5e-05', 'epoch': '0.125'}``.
READER_LOSS_RE = re.compile(r"\{'loss': '([\d.]+)'[^}]*'epoch': '([\d.]+)'\}")

TRAINING_LOSS_PLOTS = {"laya": "laya_loss_curves.png", "reader": "reader_loss_curves.png"}


def parse_laya_losses(text: str) -> list[tuple[int, float]]:
    """``(step, loss)`` pairs from a captured Laya training log, in log order."""
    out: list[tuple[int, float]] = []
    for line in text.splitlines():
        match = LAYA_LOSS_RE.match(line.strip())
        if match:
            out.append((int(match.group(3)), float(match.group(4))))
    return out


def parse_reader_losses(text: str) -> list[tuple[float, float]]:
    """``(epoch, loss)`` pairs from a captured reader (HF Trainer) log, in log order."""
    return [(float(m.group(2)), float(m.group(1))) for m in READER_LOSS_RE.finditer(text)]


def write_training_loss_plots(cfg: Mapping[str, Any], *,
                              logs: Mapping[str, Path]) -> dict[str, Path]:
    """Loss curves for every captured training log; one line per log file.

    Each log is parsed with both patterns and a log that matches neither is skipped, so the
    figure shows exactly the runs whose loss history still exists on disk — a run whose
    stdout was never captured stays absent instead of being drawn from anything else. Logs
    with fewer than two points are skipped too: one point is not a curve.
    """
    laya: dict[str, list[tuple[int, float]]] = {}
    reader: dict[str, list[tuple[float, float]]] = {}
    for label, path in sorted(logs.items()):
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        series = parse_laya_losses(text)
        if len(series) >= 2:
            laya[label] = series
        series = parse_reader_losses(text)
        if len(series) >= 2:
            reader[label] = series
    if not laya and not reader:
        return {}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = _plot_target(cfg)
    written: dict[str, Path] = {}
    if laya:
        path = plots / TRAINING_LOSS_PLOTS["laya"]
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        for label, series in laya.items():
            ax.plot([step for step, _ in series], [loss for _, loss in series],
                    linewidth=1.2, label=label)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("training loss (log line, not smoothed)")
        ax.set_title("P5 Laya — captured training loss trên toàn bộ step đã ghi")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="best")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written["laya"] = path
    if reader:
        path = plots / TRAINING_LOSS_PLOTS["reader"]
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        for label, series in reader.items():
            ax.plot([epoch for epoch, _ in series], [loss for _, loss in series],
                    linewidth=1.2, label=label)
        ax.set_xlabel("epoch (HF Trainer log step)")
        ax.set_ylabel("training loss (log line, not smoothed)")
        ax.set_title("P3 Reader — captured training loss trên toàn bộ log còn lưu")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="best")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written["reader"] = path
    return written
