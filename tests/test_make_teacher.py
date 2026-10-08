"""The teacher-oracle stage: JSONL contract, determinism, idempotency, loud failures.

Runs the real stage over the synthetic P1/P2/P3 fixture, then checks the file the way its
consumer does: every ``gold`` row must parse through ``laya.train`` exactly as written, every
state must be the public text of a real environment observation, and a rebuild must produce
byte-identical bytes.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from src.decision.schema import ACTION_ORDER
from src.decision.state_text import STATE_TEXT_VERSION, state_hash
from src.pipeline import make_teacher
from src.utils.hashing import sha256_file
from src.utils.io import read_json, read_jsonl
from tests import p5_fixtures as fx


@pytest.fixture()
def cfg(tmp_path):
    conf = fx.p5_config(tmp_path)
    fx.write_all(conf, n_train=3, n_val=2)
    return conf


def _rows(summary):
    return list(read_jsonl(summary["jsonl"]))


def _run(cfg, **kwargs):
    return make_teacher.run(cfg=cfg, profile="smoke", **kwargs)


def _statuses(cfg):
    return [row["status"] for row in read_jsonl(Path(cfg["paths"]["runs"]) / "index.jsonl")]


def test_rows_follow_the_documented_schema(cfg):
    summary = _run(cfg, limit=2)
    rows = _rows(summary)
    assert summary["rows"] == len(rows) == summary["stats"]["n_states"] > 0
    assert sha256_file(summary["jsonl"]) == summary["data_sha"]

    per_question: Counter[str] = Counter()
    for row in rows:
        assert set(row) == {"state", "questions", "expected", "gold", "meta"}
        assert row["state"].startswith("Câu hỏi:") and "\n" in row["state"]
        assert set(row["questions"]) == {"next_action"}
        question = row["questions"]["next_action"]
        assert question["type"] == "choice"
        assert list(question["criteria"]) == list(ACTION_ORDER)
        assert question["instructions"]

        meta = row["meta"]
        assert meta["state_hash"] == state_hash(row["state"])
        assert meta["state_text_version"] == STATE_TEXT_VERSION
        assert meta["split"] == "train_strict" and meta["privileged"] is False
        assert meta["qid"] in {"train_strict:q000", "train_strict:q001"}
        assert meta["scenario_id"].endswith(meta["qid"] + ":v1")
        assert meta["depth"] == len(meta["history"])
        assert meta["expected"] == row["expected"]["next_action"]
        assert meta["expected"] in meta["valid_actions"]
        assert meta["teacher_mode"] == "soft" and meta["teacher_temperature"] == 0.5
        assert meta["env_state_hash"] != "" and meta["difficulty"] == {}

        soft = {name: float(row["gold"]["next_action"]["probabilities"][name])
                for name in ACTION_ORDER}
        assert sum(soft.values()) == pytest.approx(1.0, abs=1e-9)
        for name in ACTION_ORDER:
            if name not in meta["valid_actions"]:
                assert soft[name] == 0.0, "masked actions must be exactly zero"
        assert meta["expected"] == max(ACTION_ORDER, key=lambda name: soft[name])
        per_question[meta["qid"]] += 1

    assert set(per_question) == {"train_strict:q000", "train_strict:q001"}
    limit = int(cfg["teacher"]["max_states_per_question"])
    assert all(count <= limit for count in per_question.values())
    assert summary["stats"]["states_per_question"]["max"] == max(per_question.values())


def test_run_is_idempotent_and_rebuilds_are_byte_identical(cfg):
    first = _run(cfg, limit=2)
    second = _run(cfg, limit=2)
    assert second["reused"] is True and second["data_sha"] == first["data_sha"]
    assert second["run_id"] == first["run_id"]

    forced = _run(cfg, limit=2, force=True)
    assert forced["reused"] is False and forced["data_sha"] == first["data_sha"]
    assert sha256_file(forced["jsonl"]) == first["data_sha"]

    assert _statuses(cfg) == ["RUNNING", "SUCCEEDED", "RUNNING", "SKIPPED", "RUNNING",
                              "SUCCEEDED"]


def test_a_different_limit_is_a_different_fingerprint(cfg):
    one = _run(cfg, limit=1)
    two = _run(cfg, limit=2)
    assert two["reused"] is False and two["rows"] > one["rows"]
    assert _statuses(cfg) == ["RUNNING", "SUCCEEDED", "RUNNING", "SUCCEEDED"]


def test_stats_recomputed_from_the_rows_match_the_manifest(cfg):
    summary = _run(cfg, limit=2)
    rows = _rows(summary)
    stats = summary["stats"]
    n = len(rows)

    expected_counts = Counter(row["meta"]["expected"] for row in rows)
    for name in ACTION_ORDER:
        entry = stats["action_distribution"][name]
        assert entry["count"] == expected_counts[name]
        assert entry["share"] == pytest.approx(expected_counts[name] / n)
        assert entry["mean_soft_mass"] == pytest.approx(
            sum(row["gold"]["next_action"]["probabilities"][name] for row in rows) / n)
        assert entry["masked_at_states"] == sum(
            1 for row in rows if name not in row["meta"]["valid_actions"])

    proxy = sum(1.0 - row["gold"]["next_action"]["probabilities"][row["meta"]["expected"]]
                for row in rows) / n
    assert stats["target_error_proxy"] == pytest.approx(proxy)
    assert stats["n_ambiguous"] == sum(int(row["meta"]["ambiguous"]) for row in rows)
    assert stats["ambiguous_share"] == pytest.approx(stats["n_ambiguous"] / n)
    assert sum(stats["states_per_question"]["questions_by_bracket"].values()) == \
        stats["n_questions"]

    manifest = read_json(summary["manifest"])
    assert manifest["stats"] == stats and manifest["output"]["sha256"] == summary["data_sha"]


def test_diagnostics_and_merged_reports_are_written(cfg):
    summary = _run(cfg, limit=2)
    paths = make_teacher.output_paths(cfg, "train_strict")

    distribution = paths["distribution"].read_text(encoding="utf-8").splitlines()
    assert distribution[0].startswith("split,action,count,share")
    assert [line.split(",")[1] for line in distribution[1:5]] == list(ACTION_ORDER)
    assert distribution[5].split(",")[1] == "TARGET_ERROR_PROXY"

    ties = paths["ties"].read_text(encoding="utf-8").splitlines()
    assert ties[0] == "split,qid,scenario_id,history,depth,expected,state_hash"
    assert len(ties) - 1 == min(summary["stats"]["n_ambiguous"],
                               make_teacher.MAX_REPORTED_TIES)

    checks = read_json(paths["checks"])["train_strict"]
    assert checks["passed"] is True and checks["reproducible"] is True
    assert checks["n_states"] == summary["rows"]
    assert read_json(paths["oracle_config"])["train_strict"]["mode"] == "soft"

    artifacts = set(summary["artifacts"])
    assert {str(paths[key]) for key in ("jsonl", "manifest", "distribution", "ties", "checks",
                                        "oracle_config")} == artifacts


def test_the_val_split_is_privileged_diagnostic_data(cfg):
    summary = _run(cfg, split="val_strict", limit=1)
    assert summary["jsonl"].endswith("laya_val_diagnostic.jsonl")
    assert summary["stats"]["privileged"] is True
    rows = _rows(summary)
    assert rows and all(row["meta"]["privileged"] is True for row in rows)
    assert all(row["meta"]["split"] == "val_strict" for row in rows)


def test_test_strict_labels_are_never_generated(cfg):
    with pytest.raises(ValueError, match="test_strict"):
        _run(cfg, split="test_strict", limit=1)


def test_a_missing_state_cache_stops_the_run_and_registers_it(cfg):
    parquet, meta = fx.state_cache_paths(cfg, "train_strict")
    parquet.unlink()
    meta.unlink()
    with pytest.raises(FileNotFoundError, match="cache_reader"):
        _run(cfg, limit=1)
    assert not (Path(cfg["paths"]["data_processed"]) / "laya_train.jsonl").exists()
    assert _statuses(cfg) == ["RUNNING", "FAILED"]


def test_the_rows_parse_exactly_as_laya_train_reads_them(cfg):
    """The real consumer, not a hand model: `to_internal` + both target builders."""
    from laya import train

    summary = _run(cfg, limit=1)
    rows = train.read_data(summary["jsonl"])
    assert len(rows) == summary["rows"]

    for row in rows:
        question = train.to_internal("next_action", row["questions"]["next_action"])
        assert list(question["crit"]) == list(ACTION_ORDER)
        soft = train.target_from_gold(question, row["gold"]["next_action"])
        written = [row["gold"]["next_action"]["probabilities"][name] for name in ACTION_ORDER]
        assert soft == pytest.approx(written)
        hard = train.target_from_expected(question, row["expected"]["next_action"])
        index = list(ACTION_ORDER).index(row["expected"]["next_action"])
        assert hard == pytest.approx([1.0 if i == index else 0.0 for i in range(4)])
