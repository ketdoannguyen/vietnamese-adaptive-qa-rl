"""Tests for the P9 report stage (``src/pipeline/report.py``) — synthetic artifacts only.

Every fixture lives under ``tmp_path``; ``report.REPO_ROOT`` is monkeypatched so the manifest
and the test-access log land there too. The assertions cover the stage's two hard rules: a
missing artifact renders as ``NOT_RUN``/``N/A`` (never 0, never an invented number), and any
experiment row that IS measured must carry the path of the file it came from.
"""

from __future__ import annotations

import collections
import csv
import hashlib
import json
from pathlib import Path

import pytest

from src.eval import reporting
from src.pipeline import report
from src.utils.config import load_config

NOW = "2026-10-08T00:00:00+00:00"
REL = {
    "data_raw": "data/raw",
    "data_processed": "data/processed",
    "data_scenarios": "data/scenarios",
    "data_indexes": "data/indexes",
    "data_cache": "data/cache",
    "data_diagnostics": "data/diagnostics",
    "outputs": "outputs",
    "models": "outputs/models",
    "runs": "outputs/runs",
    "predictions": "outputs/predictions",
    "manifests": "outputs/manifests",
    "diagnostics": "outputs/diagnostics",
    "reports": "reports",
    "plots": "reports/plots",
    "tables": "reports/tables",
}


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A synthetic repo root with every configured path, isolated from the real repo."""
    root = tmp_path / "repo"
    cfg = load_config()
    cfg["paths"] = {key: str(root / rel) for key, rel in REL.items()}
    for rel in set(REL.values()):
        (root / rel).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(report, "REPO_ROOT", root)
    return cfg, root


def run_report(cfg, **kwargs):
    return report.run(cfg=cfg, profile="pilot", now=NOW, **kwargs)


def write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def write_baseline_csv(root: Path, short: str, rows) -> Path:
    path = root / "reports" / "tables" / f"baseline_results_{short}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=reporting.RESULTS_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: reporting.format_cell(row.get(column))
                             for column in reporting.RESULTS_COLUMNS})
    return path


def measured_row(exp_id: str, split: str, **overrides):
    row = {"run_id": f"p7-baselines-pilot-{exp_id}-7-20261007", "exp_id": exp_id,
           "policy": "B8", "prior_regime": "rlcd", "split": split, "dev": False, "seed": 7,
           "status": "SUCCEEDED", "n_episodes": 100, "mean_episode_reward": 1.5,
           "token_f1": 0.4, "coverage": 0.75, "avg_retrieval_cost": 0.3}
    row.update(overrides)
    return row


def write_registry(root: Path, rows) -> Path:
    path = root / "outputs" / "runs" / "index.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")
    return path


# --------------------------------------------------------------------------- the sketch

def test_every_experiment_gets_a_not_run_row_and_never_a_zero(world):
    cfg, root = world
    summary = run_report(cfg)
    assert summary["n_rows"] == 16 and summary["n_succeeded"] == 0
    with open(root / "reports" / "results.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["exp_id"] for row in rows] == [exp for exp in report.EXP_IDS
                                               for _ in report.SPLITS]
    for row in rows:
        assert row["status"] == "NOT_RUN"
        assert row["mean_episode_reward"] == "N/A" and row["token_f1"] == "N/A"
        assert row["source"] == "N/A"  # nothing measured, so nothing to cite
        assert "eval_baselines" in row["reason"]  # the command that would fill it
    data = json.loads((root / "reports" / "results.json").read_text(encoding="utf-8"))
    assert data["rows"][0]["mean_episode_reward"] is None  # null, not 0
    assert data["generated_at"] == NOW and data["profile"] == "pilot"


def test_measured_rows_flow_into_results_and_the_report_with_their_source(world):
    cfg, root = world
    write_baseline_csv(root, "val", [measured_row("E06", "val_strict")])
    summary = run_report(cfg)
    assert summary["n_succeeded"] == 1
    data = json.loads((root / "reports" / "results.json").read_text(encoding="utf-8"))
    row = next(item for item in data["rows"]
               if item["exp_id"] == "E06" and item["split"] == "val_strict")
    assert row["mean_episode_reward"] == pytest.approx(1.5)
    assert row["status"] == "SUCCEEDED" and row["split"] == "val_strict"
    assert row["source"].endswith("baseline_results_val.csv")
    # The measured row counts only for the split that actually recorded it.
    other = next(item for item in data["rows"]
                 if item["exp_id"] == "E06" and item["split"] == "test_strict")
    assert other["status"] == "NOT_RUN" and other["token_f1"] is None
    text = (root / "reports" / "REPORT.md").read_text(encoding="utf-8")
    assert "1.500000" in text and "0.400000" in text
    assert "baseline_results_val.csv" in text


def test_foreign_rows_already_in_results_csv_survive(world):
    cfg, root = world
    path = root / "reports" / "results.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(report.RESULTS_OUT_COLUMNS)
        writer.writerow(["b5-run-1", "", "B5", "raw", "", "val_model_selection", "false", "7",
                         "SUCCEEDED", "train_correctness", "", "42", "40", "2"])
    run_report(cfg)
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 17  # 16 E-rows + the foreign B5 row, preserved not dropped
    foreign = next(row for row in rows if row["run_id"] == "b5-run-1")
    assert foreign["policy"] == "B5" and foreign["mean_episode_reward"] == "N/A"


# ------------------------------------------------------------------ sources and manifest

def test_manifest_lists_hashes_of_every_artifact_and_excludes_itself(world):
    cfg, root = world
    weights = root / "outputs" / "models" / "laya_ce" / "model.safetensors"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"weights")
    run_report(cfg)
    text = (root / "reports" / "artifact_manifest.sha256").read_text(encoding="utf-8")
    digest = hashlib.sha256(b"weights").hexdigest()
    assert f"{digest}  outputs/models/laya_ce/model.safetensors" in text
    assert "artifact_manifest.sha256" not in text
    assert "reports/results.csv" in text and "reports/REPORT.md" in text
    assert "docs/TEST_ACCESS_LOG.md" in text


def test_missing_stage_rows_render_not_run_and_name_the_code_path(world):
    cfg, root = world
    write_registry(root, [{"run_id": "p1-download-pilot-1", "stage": "P1_download",
                           "status": "SUCCEEDED", "ts": "2026-10-07T00:00:00+00:00",
                           "profile": "pilot"},
                          {"run_id": "p3-reader-1", "stage": "P3_reader",
                           "status": "RUNNING", "ts": "2026-10-07T01:00:00+00:00",
                           "profile": "pilot"}])
    summary = run_report(cfg)
    assert summary["stages"]["download"] == "SUCCEEDED"
    assert summary["stages"]["train_reader"] == "INTERRUPTED"  # RUNNING with no terminal row
    assert summary["stages"]["train_laya_rlcd"] == "NOT_RUN"
    handoff = (root / "reports" / "FINAL_HANDOFF.md").read_text(encoding="utf-8")
    assert "INTERRUPTED" in handoff
    assert "python -m src.pipeline.train_rl" in handoff  # recovery command is always shown
    access = (root / "docs" / "TEST_ACCESS_LOG.md").read_text(encoding="utf-8")
    assert "test split chưa được mở cho đánh giá." in access


# ----------------------------------------------------------------------- test access log

def test_access_log_cannot_prove_order_without_the_selection_freeze(world):
    cfg, root = world
    write_registry(root, [{"run_id": "p7-baselines-pilot-E06-7-1", "stage": "P7_baselines",
                           "status": "SUCCEEDED", "ts": "2026-10-07T02:00:00+00:00",
                           "profile": "pilot", "split": "test_strict"}])
    run_report(cfg)
    access = (root / "docs" / "TEST_ACCESS_LOG.md").read_text(encoding="utf-8")
    assert "KHÔNG THỂ CHỨNG MINH" in access
    assert "best_selection.json" in access
    assert "p7-baselines-pilot-E06-7-1" in access


def test_access_log_proves_order_when_the_freeze_precedes_the_test_row(world):
    cfg, root = world
    write_json(root / "outputs" / "manifests" / "best_selection.json",
               {"created_utc": "2026-10-07T01:00:00+00:00", "selected_update": 12})
    write_registry(root, [{"run_id": "p7-baselines-pilot-E06-7-3", "stage": "P7_baselines",
                           "status": "SUCCEEDED", "ts": "2026-10-07T02:00:00+00:00",
                           "profile": "pilot", "split": "test_strict"}])
    run_report(cfg)
    access = (root / "docs" / "TEST_ACCESS_LOG.md").read_text(encoding="utf-8")
    assert "Đã khoá lựa chọn" in access and "KHÔNG THỂ CHỨNG MINH" not in access


def test_access_log_flags_a_violation_when_the_freeze_is_later(world):
    cfg, root = world
    write_json(root / "outputs" / "manifests" / "best_selection.json",
               {"created_utc": "2026-10-07T03:00:00+00:00"})
    write_registry(root, [{"run_id": "p7-baselines-pilot-E06-7-4", "stage": "P7_baselines",
                           "status": "SUCCEEDED", "ts": "2026-10-07T02:00:00+00:00",
                           "profile": "pilot", "split": "test_strict"}])
    run_report(cfg)
    access = (root / "docs" / "TEST_ACCESS_LOG.md").read_text(encoding="utf-8")
    assert "VI PHẠM THỨ TỰ" in access


# ------------------------------------------------------------------------------ outputs

def test_report_has_the_fifteen_template_sections_and_a_text_version(world):
    cfg, root = world
    run_report(cfg)
    text = (root / "reports" / "REPORT.md").read_text(encoding="utf-8")
    for number, title in ((1, "Giới thiệu và phát biểu vấn đề"), (2, "Cơ sở lý thuyết"),
                           (3, "Dữ liệu, giấy phép và chia tập"), (4, "Phương pháp đề xuất"),
                           (5, "Cài đặt và môi trường"), (6, "Thiết kế thí nghiệm"),
                           (7, "Kết quả quantitative — KHÔNG ĐƯỢC BỎ"), (8, "Phân tích"),
                           (9, "Failure cases thực tế"), (10, "Đánh giá giả thuyết"),
                           (11, "Hạn chế"), (12, "Đạo đức / an toàn"), (13, "Tái lập"),
                           (14, "Kết luận"), (15, "Tài liệu tham khảo")):
        assert f"## {number}. {title}" in text
    assert "## Tóm tắt / Abstract" in text and "## Phụ lục" in text
    assert "[FROM_ARTIFACT:" not in text
    assert "NOT_RUN" in text  # the unfilled rows are visible, not hidden
    plain = (root / "reports" / "REPORT.txt").read_text(encoding="utf-8")
    assert not any(line.startswith("#") for line in plain.splitlines())
    assert "BÁO CÁO CUỐI KỲ" in plain
    assert len(plain) > 0.5 * len(text)
    card = (root / "reports" / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert "Model identifier" in card and "UNVERIFIED" in card
    reproduce = (root / "reports" / "REPRODUCE.md").read_text(encoding="utf-8")
    assert "scripts/reproduce.sh --run-id" in reproduce
    handoff = (root / "reports" / "FINAL_HANDOFF.md").read_text(encoding="utf-8")
    assert "## Executive status" in handoff and "PARTIAL" in handoff  # nothing ran in this world


def test_failure_cases_are_read_from_the_predictions_artifact(world):
    cfg, root = world
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [
        {"qid": "val:0001", "split": "val_strict", "exp_id": "E06", "seed": 7,
         "terminal_action": "ANSWER", "eval_correct": 0, "eval_is_impossible": False,
         "action_name_trace": "SEARCH_BM25|ANSWER", "total_retrievals": 2,
         "eval_correctness_f1": 0.0, "final_candidate": "sai"},
        {"qid": "val:0002", "split": "val_strict", "exp_id": "E06", "seed": 7,
         "terminal_action": "ANSWER", "eval_correct": 1, "eval_is_impossible": False,
         "action_name_trace": "ANSWER", "total_retrievals": 1,
         "eval_correctness_f1": 1.0, "final_candidate": "đúng"},
        {"qid": "val:0003", "split": "val_strict", "exp_id": "E06", "seed": 7,
         "terminal_action": "ANSWER", "eval_correct": 0, "eval_is_impossible": True,
         "action_name_trace": "ANSWER", "total_retrievals": 1,
         "eval_correctness_f1": None, "final_candidate": "bịa"},
    ]
    path = root / "outputs" / "predictions" / "p7-baselines-pilot-E06-7-1_val_strict.parquet"
    pq.write_table(pa.Table.from_pylist(rows), str(path))
    run_report(cfg)
    text = (root / "reports" / "REPORT.md").read_text(encoding="utf-8")
    assert "val:0001" in text and "val:0003" in text
    assert "val:0002" not in text  # a correct episode is not a failure case
    assert "2 mẫu thất bại thực tế trong 3 dòng predictions" in text
    assert "trả lời câu không có đáp án" in text


def test_rerun_with_the_same_now_leaves_every_report_product_identical(world):
    cfg, root = world
    write_baseline_csv(root, "val", [measured_row("E06", "val_strict")])
    run_report(cfg)
    first = {path.name: path.read_bytes()
             for path in sorted((root / "reports").iterdir()) if path.is_file()}
    first_manifest = first.pop("artifact_manifest.sha256").decode("utf-8").splitlines()
    first_access = (root / "docs" / "TEST_ACCESS_LOG.md").read_bytes()
    run_report(cfg)
    second = {path.name: path.read_bytes()
              for path in sorted((root / "reports").iterdir()) if path.is_file()}
    second_manifest = second.pop("artifact_manifest.sha256").decode("utf-8").splitlines()
    assert second == first  # REPORT.md/txt, results.*, MODEL_CARD, REPRODUCE, FINAL_HANDOFF
    assert (root / "docs" / "TEST_ACCESS_LOG.md").read_bytes() == first_access
    # The manifest is a hash inventory of the append-only registry too, so exactly that one
    # line moves between runs; every other line must still match the file on disk.
    differing = [line for line, old in zip(second_manifest, first_manifest) if line != old]
    assert len(second_manifest) == len(first_manifest)
    assert len(differing) == 1 and differing[0].endswith("outputs/runs/index.jsonl")
    for line in second_manifest:
        digest, name = line.split("  ", 1)
        if name == "outputs/runs/index.jsonl":
            continue  # this stage appends its own completion row after writing the manifest
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest


def test_the_stage_records_itself_in_the_registry(world):
    cfg, root = world
    summary = run_report(cfg)
    rows = [json.loads(line) for line in
            (root / "outputs" / "runs" / "index.jsonl").read_text(encoding="utf-8").splitlines()]
    statuses = [row["status"] for row in rows if row["stage"] == "P9_report"]
    assert statuses == ["RUNNING", "SUCCEEDED"]
    assert rows[-1]["run_id"] == summary["run_id"]
    assert rows[-1]["artifacts"]  # checksummed artifacts, not just a status claim


def test_a_placeholder_in_a_source_artifact_fails_the_run_and_is_recorded(world):
    cfg, root = world
    write_json(root / "reports" / "retrieval_report.json",
               {"recall_at_1": {"bm25": "[FROM_ARTIFACT: something]"}})
    with pytest.raises(RuntimeError, match="FROM_ARTIFACT"):
        run_report(cfg)
    rows = [json.loads(line) for line in
            (root / "outputs" / "runs" / "index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["status"] for row in rows if row["stage"] == "P9_report"] == ["RUNNING",
                                                                              "FAILED"]


# ----------------------------------------------------------------------------------- zip

def test_zip_is_off_by_default_and_excludes_weights_when_asked(world):
    cfg, root = world
    weights = root / "outputs" / "models" / "qa_reader" / "model.safetensors"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"weights")
    (root / "scripts").mkdir()
    (root / "scripts" / "run_all.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    summary = run_report(cfg)
    assert summary["zip"] is None
    assert not (root / "reports" / report.ZIP_FILE).exists()
    summary = run_report(cfg, zip_handoff=True)
    assert summary["zip"] and "tệp" in summary["zip"]
    import zipfile
    with zipfile.ZipFile(root / "reports" / report.ZIP_FILE) as archive:
        names = archive.namelist()
    assert "reports/REPORT.md" in names and "scripts/run_all.sh" in names
    assert not any(name.endswith(".safetensors") for name in names)


def test_zip_is_refused_when_the_disk_has_no_headroom(world, monkeypatch):
    cfg, root = world
    usage = collections.namedtuple("usage", "total used free")
    monkeypatch.setattr(report.shutil, "disk_usage", lambda path: usage(1, 1, 0))
    summary = run_report(cfg, zip_handoff=True)
    # The refusal is disclosed in the summary (and the registry row), not silently ignored.
    assert summary["zip"] and "từ chối --zip" in summary["zip"]
    assert not (root / "reports" / report.ZIP_FILE).exists()
