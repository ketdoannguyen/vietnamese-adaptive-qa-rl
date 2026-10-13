"""Assemble the seven-arm V2 validation report: metric table, paired comparisons, curves.

Plan §7 (``.claude/plans/2026-10-12-vixstory-evidence-first-v2.md``): after E00/E01/E02/
E04/E05/E06/E07 are evaluated on the full 246-QID validation split, report the retrieval
metric table, the paired E06−E05 and E07−E04 comparisons with QID-bootstrap CIs, and the
learning curves — negative results included. This script only reads the private validation
JSONs and selections written by the training/evaluation runners; it writes aggregate
artifacts (no gold IDs, no per-QID records) under ``run_dir/report/``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.pipeline.vixstory_eval_v2 import paired_difference
from src.utils.io import atomic_write_json

EXPECTED_QIDS = 246
PAIRED = (("E06", "E05"), ("E07", "E04"))
SUMMARY_FIELDS = ("mean_retrieval_return", "full_gold_success_n", "mean_full_gold_success",
                  "mean_gold_recall", "mean_gold_precision", "mean_n_search", "mean_seen_count")


def _read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _private(path: Path) -> dict:
    payload = _read(path)
    if (payload.get("summary", {}).get("n") != EXPECTED_QIDS
            or len(payload.get("per_qid", {})) != EXPECTED_QIDS):
        raise ValueError(f"{path} does not hold the full {EXPECTED_QIDS}-QID validation")
    return payload


def _sources(run_dir: Path) -> list[tuple[str, str, Path]]:
    """Arm -> (selection label, private validation file), in report order."""
    baselines = _read(run_dir / "baselines/selection.json")
    sources = [("E00", "stop", run_dir / "baselines/E00-validation-private.json")]
    for arm in ("E01", "E02"):
        index = baselines["selection"][arm]["index"]
        if index not in (1, 2):
            raise ValueError(f"{arm} selection index is not K in 1..2: {index!r}")
        sources.append((arm, f"K={index}",
                        run_dir / f"baselines/{arm}-K{index}-validation-private.json"))
    for arm in ("E04", "E05"):
        epoch = _read(run_dir / "laya" / arm / "selection.json")["selected_epoch"]
        sources.append((arm, f"epoch {epoch}",
                        run_dir / "laya" / arm / f"epoch-{epoch}-validation-private.json"))
    for arm in ("E06", "E07"):
        update = _read(run_dir / "rl" / arm / "selection.json")["selected_update"]
        sources.append((arm, f"update {update}",
                        run_dir / "rl" / arm / f"update_{update:04d}-validation-private.json"))
    sources.append(("random", "seed 42", run_dir / "baselines/random-seed42-validation-private.json"))
    return sources


def _table_and_per_qid(run_dir: Path) -> tuple[list[dict], dict[str, dict], dict]:
    rows, per_qid, digests = [], {}, set()
    for arm, label, path in _sources(run_dir):
        payload = _private(path)
        summary = payload["summary"]
        digests.add(summary["qids_sha256"])
        per_qid[arm] = payload["per_qid"]
        rows.append({"arm": arm, "selection": label,
                     **{field: summary[field] for field in SUMMARY_FIELDS}})
    if len(digests) != 1:
        raise ValueError("arms were not evaluated on one identical validation split")
    return rows, per_qid, _read(run_dir / "baselines/selection.json")


def _curves(run_dir: Path) -> dict:
    laya = {}
    for arm in ("E04", "E05"):
        selection = _read(run_dir / "laya" / arm / "selection.json")
        candidates = selection["candidates"]
        if len(candidates) != 3 or [c["index"] for c in candidates] != [1, 2, 3]:
            raise ValueError(f"{arm} selection candidates are not the three epochs")
        laya[arm] = [[c["index"], c["mean_retrieval_return"]] for c in candidates]
    rl = {}
    for arm in ("E06", "E07"):
        selection = _read(run_dir / "rl" / arm / "selection.json")
        candidates = selection["candidates"]
        updates = [c["update"] for c in candidates]
        if len(candidates) != 10 or updates != [*range(25, 241, 25), 240]:
            raise ValueError(f"{arm} selection candidates are not the ten-update cadence")
        rl[arm] = [[c["update"], c["mean_retrieval_return"]] for c in candidates]
    return {"laya_epochs": laya, "rl_updates": rl}


def _write_plots(out: Path, rows: list[dict], curves: dict, ceiling: float) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = {}
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))
    for arm, points in curves["laya_epochs"].items():
        left.plot([x for x, _ in points], [y for _, y in points], marker="o", label=arm)
    left.set(xlabel="epoch", ylabel="mean retrieval return", xticks=[1, 2, 3],
             title="Laya imitation (E04/E05) — validation by epoch")
    for arm, points in curves["rl_updates"].items():
        right.plot([x for x, _ in points], [y for _, y in points], marker="o", label=arm)
    right.axhline(ceiling, ls="--", color="gray", label=f"oracle {ceiling:.3f}")
    right.set(xlabel="REINFORCE update", ylabel="mean retrieval return",
              title="REINFORCE (E06/E07) — greedy validation by update")
    for axis in (left, right):
        axis.legend()
        axis.grid(alpha=0.25)
    paths["learning_curves"] = out / "learning-curves.png"
    fig.savefig(paths["learning_curves"], dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    colors = ["0.6" if row["arm"] == "random" else "tab:blue" for row in rows]
    bars = axis.bar([row["arm"] for row in rows],
                    [row["mean_retrieval_return"] for row in rows], color=colors)
    axis.bar_label(bars, fmt="%.3f", fontsize=8)
    axis.axhline(ceiling, ls="--", color="gray", label=f"oracle {ceiling:.3f}")
    axis.set(ylabel="mean retrieval return",
             title=f"Seven-arm validation comparison ({EXPECTED_QIDS} QIDs)")
    axis.legend()
    axis.grid(alpha=0.25, axis="y")
    paths["arm_comparison"] = out / "arm-comparison.png"
    fig.savefig(paths["arm_comparison"], dpi=150, bbox_inches="tight")
    plt.close(fig)
    return paths


def run(run_dir: Path) -> dict:
    out = run_dir / "report"
    if (out / "summary.json").exists():
        raise FileExistsError(f"refusing to overwrite {out / 'summary.json'}")
    rows, per_qid, baselines = _table_and_per_qid(run_dir)
    curves = _curves(run_dir)
    paired = {f"{arm}_minus_{base}": paired_difference(per_qid[arm], per_qid[base],
                                                       metric="retrieval_return")
              for arm, base in PAIRED}
    ceiling = float(baselines["oracle"]["mean_oracle_return"])
    out.mkdir(parents=True, exist_ok=True)
    plots = _write_plots(out, rows, curves, ceiling)
    report = {
        "scope": "official_validation_only",
        "n_qids": EXPECTED_QIDS,
        "evaluation_label": paired["E06_minus_E05"]["evaluation_label"],
        "oracle": baselines["oracle"],
        "arms": rows,
        "paired": paired,
        "learning_curves": curves,
        "plots": {key: str(path) for key, path in plots.items()},
        "note": "aggregate metrics only; per-QID gold stays in the private validation files",
    }
    atomic_write_json(out / "summary.json", report)
    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["arm", "selection", *SUMMARY_FIELDS])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"arms": rows,
                      "paired": {key: {"mean_difference": value["mean_difference"],
                                       "bootstrap": value["bootstrap"]}
                                 for key, value in paired.items()},
                      "oracle": baselines["oracle"]}), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.run_dir)


if __name__ == "__main__":
    main()
