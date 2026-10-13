"""The seven-arm report reads the private selections and refuses inconsistent inputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.vixstory_v2_report import run

QIDS = [f"qa_{index:06d}" for index in range(246)]
UPDATES = [*range(25, 241, 25), 240]


def private(arm: str, *, split: str = "split-sha", return_: float = 0.3) -> dict:
    return {"schema": "vixstory-eval-v2", "arm": arm,
            "summary": {"arm": arm, "n": 246, "qids_sha256": split,
                        "mean_retrieval_return": return_, "full_gold_success_n": 50,
                        "mean_full_gold_success": 50 / 246, "mean_gold_recall": 0.4,
                        "mean_gold_precision": 0.2, "mean_n_search": 1.0,
                        "mean_seen_count": 2.0},
            "per_qid": {qid: {"qid": qid, "arm": arm, "retrieval_return": return_}
                        for qid in QIDS}}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def make_run_dir(tmp_path: Path) -> Path:
    baselines = tmp_path / "baselines"
    for name, return_ in (("E00", 0.2), ("random-seed42", 0.25), ("E01-K1", 0.28),
                          ("E01-K2", 0.33), ("E02-K1", 0.3), ("E02-K2", 0.335)):
        write_json(baselines / f"{name}-validation-private.json", private(name, return_=return_))
    write_json(baselines / "selection.json", {
        "oracle": {"mean_oracle_return": 0.52, "full_gold_reachable_rate": 0.34,
                   "full_gold_reachable_n": 84},
        "selection": {"E01": {"index": 2}, "E02": {"index": 2}}})
    for arm, epoch, candidates in (("E04", 3, [0.27, 0.275, 0.28]),
                                   ("E05", 1, [0.24, 0.23, 0.22])):
        location = tmp_path / "laya" / arm
        write_json(location / "selection.json", {
            "arm": arm, "selected_epoch": epoch,
            "candidates": [{"index": index, "mean_retrieval_return": value}
                           for index, value in enumerate(candidates, start=1)]})
        write_json(location / f"epoch-{epoch}-validation-private.json",
                   private(arm, return_=candidates[epoch - 1]))
    for arm, return_ in (("E06", 0.31), ("E07", 0.35)):
        location = tmp_path / "rl" / arm
        write_json(location / "selection.json", {
            "arm": arm, "selected_update": 240,
            "candidates": [{"update": update, "mean_retrieval_return": return_}
                           for update in UPDATES]})
        write_json(location / "update_0240-validation-private.json",
                   private(arm, return_=return_))
    return tmp_path


def test_report_writes_table_paired_blocks_csv_and_plots(tmp_path):
    run_dir = make_run_dir(tmp_path)
    report = run(run_dir)
    assert [row["arm"] for row in report["arms"]] == ["E00", "E01", "E02", "E04", "E05",
                                                      "E06", "E07", "random"]
    assert report["arms"][2]["selection"] == "K=2"
    assert report["arms"][3]["selection"] == "epoch 3"
    assert report["arms"][5]["selection"] == "update 240"
    paired = report["paired"]["E07_minus_E04"]
    assert paired["n"] == 246
    assert abs(paired["mean_difference"] - (0.35 - 0.28)) < 1e-9
    assert paired["bootstrap"]["ci_low"] <= paired["bootstrap"]["ci_high"]
    assert report["oracle"]["full_gold_reachable_n"] == 84
    out = run_dir / "report"
    with (out / "summary.csv").open(encoding="utf-8") as handle:
        lines = list(csv.reader(handle))
    assert lines[0][:2] == ["arm", "selection"] and len(lines) == 9
    for name in ("learning-curves.png", "arm-comparison.png"):
        assert (out / name).stat().st_size > 0


def test_report_refuses_to_overwrite(tmp_path):
    run_dir = make_run_dir(tmp_path)
    run(run_dir)
    with pytest.raises(FileExistsError):
        run(run_dir)


def test_report_rejects_split_mismatch(tmp_path):
    run_dir = make_run_dir(tmp_path)
    path = run_dir / "baselines/E00-validation-private.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["summary"]["qids_sha256"] = "other-split"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="identical validation split"):
        run(run_dir)
