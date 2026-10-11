"""P3 pipeline wiring: reader fine-tune -> frozen contract -> exhaustive QA state cache.

Synthetic P1/P2 artifacts are written into ``tmp_path``, the real XLM-R tokenizer and a
2-layer random model stand in for the checkpoint, and the cache test replaces the reader with
a deterministic stub — so the whole stage is exercised on CPU without a GPU or a download.
The overfit gate's own behaviour (a real loss decrease) is tested in ``test_reader.py``; here
the gate is faked so only its wiring is under test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data import artifacts
from src.pipeline import cache_reader, train_reader
from src.qa import state_cache
from src.qa.features import build_feature_set
from src.qa.infer import inference_config_sha
from src.qa.postprocess import NO_SPAN_SCORE, BestSpan
from src.qa.reader import reader_cache_key
from src.qa.train_reader import load_tokenizer, reader_model_hash
from src.retrieval.states import reachable_states, retrieved_doc_ids
from src.utils.config import load_config
from src.utils.hashing import sha256_file
from src.utils.io import read_json, read_jsonl, write_jsonl, write_parquet, write_yaml
from src.utils.runs import RunRegistry

BASE = "FacebookAI/xlm-roberta-base"
TRAIN_N, VAL_N = 40, 12
IMPOSSIBLE_EVERY = 4


@pytest.fixture(scope="session")
def tokenizer():
    return load_tokenizer(BASE)


def _tiny_model(tokenizer):
    import torch
    from transformers import AutoModelForQuestionAnswering, XLMRobertaConfig

    torch.manual_seed(0)  # deterministic init: the pipeline tests assert decoded spans
    config = XLMRobertaConfig(vocab_size=tokenizer.vocab_size, hidden_size=32,
                              num_hidden_layers=2, num_attention_heads=2,
                              intermediate_size=64, max_position_embeddings=512)
    return AutoModelForQuestionAnswering.from_config(config)


# ------------------------------------------------------------------- synthetic artifacts

def _split_rows(split: str, n: int) -> tuple[list[dict], list[dict], list[dict]]:
    """(docs, label-store rows, scenario rows) for ``n`` synthetic questions."""
    docs, questions, scenarios = [], [], []
    for i in range(n):
        doc_id = f"doc:{split}:{i:03d}"
        text = (f"Đoạn văn {i:03d} thuộc bộ {split} kể về sự kiện lịch sử diễn ra "
                f"vào năm {1900 + i} tại một vùng đất cụ thể.")
        answer = str(1900 + i)
        impossible = i % IMPOSSIBLE_EVERY == 0
        qid = f"{split}:q{i:03d}"
        docs.append({"doc_id": doc_id, "text": text, "text_raw": text})
        questions.append({
            "qid": qid, "question": f"Đoạn văn {i:03d} nhắc tới sự kiện năm nào?",
            "gold_doc_id": doc_id, "is_impossible": impossible,
            "answer_texts": [] if impossible else [answer],
            "answer_texts_display": [] if impossible else [answer],
            "answer_starts_display": [] if impossible else [text.index(answer)]})
        scenarios.append({
            "scenario_id": f"sc:{split}:{qid}:v1", "qid": qid, "split": split,
            "corpus_doc_ids": [doc_id, f"doc:{split}:{(i + 1) % n:03d}",
                               f"doc:{split}:{(i + 2) % n:03d}"],
            "scenario_seed": 42, "gold_doc_id": None if impossible else doc_id,
            "reachable_bm25": None, "reachable_dense": None, "difficulty": {}})
    return docs, questions, scenarios


def _write_split(cfg, split: str, n: int) -> None:
    processed, scenarios_dir = (cfg["paths"]["data_processed"], cfg["paths"]["data_scenarios"])
    docs, questions, scenarios = _split_rows(split, n)
    write_parquet(Path(processed) / f"docs_{split}.parquet", docs)
    write_parquet(Path(processed) / f"questions_{split}.parquet", questions)
    write_jsonl(artifacts.scenarios_path(scenarios_dir, split), scenarios)


def _rankings(cfg, split: str, n: int) -> dict[str, dict[str, list[str]]]:
    """Odd questions get the dense tool as the only route to their own document."""
    out = {}
    for i in range(n):
        own = f"doc:{split}:{i:03d}"
        others = [f"doc:{split}:{(i + 1) % n:03d}", f"doc:{split}:{(i + 2) % n:03d}"]
        bm25 = [own, *others] if i % 2 == 0 else [others[0], own, others[1]]
        dense = [own, *others] if i % 2 == 1 else [others[1], others[0], own]
        out[f"{split}:q{i:03d}"] = {"bm25": bm25, "dense": dense}
    return out


def _write_rankings(cfg, split: str, n: int) -> None:
    path = train_reader.rankings_path(cfg, split)
    write_jsonl(path, [{"scenario_id": f"sc:{split}:{qid}:v1", "qid": qid,
                        "rankings": rankings, "scores": {}}
                       for qid, rankings in sorted(_rankings(cfg, split, n).items())])


# ---------------------------------------------------------------------------- fixtures

@pytest.fixture()
def cfg(tmp_path, monkeypatch, tokenizer):
    """Config with all paths under tmp_path and the model/tokenizer patched to tiny ones."""
    from src.qa import train_reader as qa_train

    def fake_load_tokenizer(_name=None, revision=None):
        return tokenizer

    def fake_load_model(name_or_dir=None, revision=None):
        path = Path(str(name_or_dir))
        if path.is_dir() and (path / "config.json").exists():
            from transformers import AutoModelForQuestionAnswering
            return AutoModelForQuestionAnswering.from_pretrained(str(path))
        return _tiny_model(tokenizer)

    monkeypatch.setattr(qa_train, "load_tokenizer", fake_load_tokenizer)
    monkeypatch.setattr(qa_train, "load_model", fake_load_model)
    monkeypatch.setattr(train_reader, "load_tokenizer", fake_load_tokenizer)
    monkeypatch.setattr(train_reader, "load_model", fake_load_model)

    conf = load_config(None)
    keys = ("data_raw", "data_processed", "data_scenarios", "data_indexes", "data_cache",
            "data_diagnostics", "outputs", "models", "runs", "predictions", "manifests",
            "diagnostics", "reports", "plots", "tables")
    conf["paths"] = {**conf["paths"], **{k: str(tmp_path / k) for k in keys}}
    conf["data"]["profile"] = "smoke"
    conf["data"]["caps"]["smoke"] = None  # use every synthetic question
    conf["reader"].update({
        "fp16": False, "epochs": 1, "batch_size": 4, "grad_accum": 8, "warmup_ratio": 0.0,
        # A small grid around the tiny model's deterministic val null margins (they cluster
        # just above -0.45) exercises the D010 selection end-to-end: -0.5 nulls every pair,
        # failing the "answers something" constraint while owning the best raw f1_overall
        # (the all-null row), so feasibility must overrule the objective and pick -0.4.
        "threshold_sweep": {"start": -0.5, "stop": -0.4, "step": 0.1}})
    _write_split(conf, "train_strict", TRAIN_N)
    _write_split(conf, "val_strict", VAL_N)
    return conf


def _registry(cfg) -> RunRegistry:
    return RunRegistry(Path(cfg["paths"]["runs"]) / "index.jsonl")


def _reader_run(cfg, **kwargs):
    return train_reader.run(cfg=cfg, profile="smoke", registry=_registry(cfg), **kwargs)


# -------------------------------------------------------------------- reader end-to-end

def test_reader_run_end_to_end(cfg):
    summary = _reader_run(cfg, skip_gate=True)
    out_dir = Path(summary["reader_dir"])
    for name in ("model.safetensors", "config.json", "tokenizer_config.json",
                 "training_args.json", "reader_inference.json", "reader_eval.json",
                 "reader_predictions_val.parquet", "model_card.md", "checksums.json"):
        assert (out_dir / name).exists(), name

    model_hash = reader_model_hash(out_dir)
    assert summary["reader_model_hash"] == model_hash
    contract = read_json(out_dir / "reader_inference.json")
    assert contract["reader_model_hash"] == model_hash
    assert contract["tokenizer_sha"] == summary["tokenizer_sha"]
    assert contract["null_score_diff_threshold"] == -0.4
    assert contract["tuned_on"] == "val_strict"
    assert contract["inference_config_sha"] == inference_config_sha(
        cfg["reader"], threshold=-0.4, model_hash=model_hash)

    training = read_json(out_dir / "training_args.json")
    assert training["train_rows"] == TRAIN_N
    assert training["profile"] == "smoke"
    assert training["dropped_rows"] == []

    evaluation = read_json(out_dir / "reader_eval.json")
    # Selection-first order: the feasible row precedes the better-scoring all-null row.
    assert [row["threshold"] for row in evaluation["threshold_sweep"]] == [-0.4, -0.5]
    selection = evaluation["threshold_selection"]
    assert selection["objective"] == "f1_overall"
    assert selection["chosen_threshold"] == -0.4
    assert selection["n_feasible_thresholds"] == 1 and selection["n_thresholds"] == 2
    assert selection["constraint_binding"] is True  # the objective alone would pick -0.5
    assert selection["objective_only_threshold"] == -0.5
    oracle = evaluation["oracle_context"]
    assert oracle["n"] == VAL_N
    assert oracle["n_impossible"] == VAL_N // IMPOSSIBLE_EVERY
    assert oracle["acceptance_constraint_met"] is True
    assert oracle["selection_objective"] == pytest.approx(oracle["f1_overall"])
    assert evaluation["retrieved_context"]["status"] == "NOT_RUN"
    assert "build_retrieval" in evaluation["retrieved_context"]["reason"]
    assert evaluation["acceptance"]["val_metrics_saved"]["passed"] is True
    # the gate was skipped, so the acceptance check reports that honestly instead of passing
    assert evaluation["acceptance"]["overfit_loss_decreased"] is None

    rows = _predictions(cfg)
    oracle_rows = [row for row in rows if row["condition"] == "oracle_context"]
    assert len(oracle_rows) == VAL_N
    answered = [row for row in oracle_rows if row["answer_text"] is not None]
    assert answered, "no oracle pair produced a span; decoding or the threshold is broken"
    assert all(row["provenance_ok"] for row in answered)
    example = evaluation["acceptance"]["non_empty_span_with_valid_offset"]
    assert example["passed"] is True and example["example"]["answer_text"]
    assert all(row["is_candidate"] == (row["answer_text"] is not None)
               for row in oracle_rows)

    checksums = read_json(out_dir / "checksums.json")
    # checksums.json holds raw file hashes; reader_model_hash is the manifest hash the
    # cache keys are built from — related but deliberately not the same digest.
    assert checksums["files"]["model.safetensors"] == sha256_file(out_dir / "model.safetensors")
    assert set(checksums["files"]) >= {"model.safetensors", "reader_inference.json",
                                       "reader_eval.json", "reader_predictions_val.parquet",
                                       "model_card.md"}
    assert "train_strict" in (out_dir / "model_card.md").read_text(encoding="utf-8")

    registry_rows = list(read_jsonl(_registry(cfg).path))
    assert [row["status"] for row in registry_rows] == ["RUNNING", "SUCCEEDED"]
    for path in summary["artifacts"]:
        assert registry_rows[-1]["artifacts"][path]["sha256"]


def _predictions(cfg) -> list[dict]:
    import pyarrow.parquet as pq
    path = train_reader.reader_dir(cfg) / train_reader.PREDICTIONS_FILE
    return pq.read_table(str(path)).to_pylist()


def test_reader_run_reports_retrieved_context_when_rankings_exist(cfg):
    _write_rankings(cfg, "val_strict", VAL_N)
    _reader_run(cfg, skip_gate=True)
    evaluation = read_json(train_reader.reader_dir(cfg) / train_reader.EVAL_FILE)
    retrieved = evaluation["retrieved_context"]
    assert retrieved["n"] == VAL_N and "em_overall" in retrieved

    expected = set()
    for qid, ranking in _rankings(cfg, "val_strict", VAL_N).items():
        states = reachable_states(ranking, bootstrap_tool="bm25", bootstrap_passages=1,
                                  passage_per_search_action=1, max_optional_search=2)
        expected |= set(retrieved_doc_ids(states))
    rows = [row for row in _predictions(cfg) if row["condition"] == "retrieved_context"]
    assert rows and {row["doc_id"] for row in rows} == expected
    for qid in {row["qid"] for row in rows}:
        candidates = [row for row in rows if row["qid"] == qid and row["is_candidate"]]
        assert len(candidates) <= 1
        assert all(row["answer_text"] is not None for row in candidates)


def test_reader_is_frozen_after_training(cfg):
    _reader_run(cfg, skip_gate=True)
    with pytest.raises(RuntimeError, match="already holds a trained reader"):
        _reader_run(cfg, skip_gate=True)
    assert _reader_run(cfg, skip_gate=True, force=True)["reader_model_hash"]


def test_reselect_rederives_the_threshold_without_retraining(cfg, monkeypatch):
    # The gate result is written by a normal run; --reselect only re-reads it.
    monkeypatch.setattr(train_reader, "overfit_smoke",
                        lambda *, rows, **kwargs: {"loss_first": 1.0, "loss_last": 0.5,
                                                   "loss_decreased": True, "rows": len(rows)})
    _reader_run(cfg)
    out_dir = train_reader.reader_dir(cfg)
    weights = (out_dir / "model.safetensors").read_bytes()
    training_args = (out_dir / "training_args.json").read_bytes()
    assert read_json(out_dir / train_reader.INFERENCE_FILE)[
        "null_score_diff_threshold"] == -0.4

    cfg["reader"]["threshold_sweep"] = {"start": -0.4, "stop": -0.4, "step": 1.0}
    summary = _reader_run(cfg, reselect_only=True)

    assert summary["reselect"] is True and summary["reader_model_hash"]
    # Only threshold-dependent artifacts are rewritten; weights and training args are not.
    assert (out_dir / "model.safetensors").read_bytes() == weights
    assert (out_dir / "training_args.json").read_bytes() == training_args
    contract = read_json(out_dir / train_reader.INFERENCE_FILE)
    assert contract["null_score_diff_threshold"] == -0.4 and contract["sweep_points"] == 1
    evaluation = read_json(out_dir / train_reader.EVAL_FILE)
    assert evaluation["reselect"]["previous_threshold"] == -0.4
    assert evaluation["threshold_selection"]["chosen_threshold"] == -0.4
    assert [row["status"] for row in read_jsonl(_registry(cfg).path)] == [
        "RUNNING", "SUCCEEDED", "RUNNING", "SUCCEEDED"]
    assert [row["stage"] for row in read_jsonl(_registry(cfg).path)] == [
        "P3_reader", "P3_reader", "P3_reader_reselect", "P3_reader_reselect"]


def test_reselect_refuses_without_a_frozen_reader(cfg):
    with pytest.raises(ValueError, match="mutually exclusive"):
        _reader_run(cfg, force=True, reselect_only=True)
    with pytest.raises(RuntimeError, match="no frozen reader"):
        _reader_run(cfg, reselect_only=True)
    assert [row["status"] for row in read_jsonl(_registry(cfg).path)] == ["RUNNING", "FAILED"]


def test_overfit_gate_blocks_training_when_it_fails(cfg, monkeypatch):
    seen = {}

    def failing_gate(*, rows, **kwargs):
        seen["rows"] = len(rows)
        return {"loss_first": 1.0, "loss_last": 1.0, "loss_decreased": False}

    monkeypatch.setattr(train_reader, "overfit_smoke", failing_gate)
    with pytest.raises(RuntimeError, match="overfit gate did not decrease"):
        _reader_run(cfg)
    assert seen["rows"] >= 32  # the gate always gets at least 32 windows
    assert not (train_reader.reader_dir(cfg) / "model.safetensors").exists()
    assert [row["status"] for row in read_jsonl(_registry(cfg).path)] == ["RUNNING", "FAILED"]


def test_overfit_gate_result_is_recorded_and_used(cfg, monkeypatch):
    monkeypatch.setattr(train_reader, "overfit_smoke",
                        lambda *, rows, **kwargs: {"loss_first": 1.0, "loss_last": 0.5,
                                                   "loss_decreased": True, "rows": len(rows)})
    summary = _reader_run(cfg)
    out_dir = Path(summary["reader_dir"])
    gate = read_json(out_dir / train_reader.GATE_FILE)
    assert gate["loss_decreased"] is True and gate["profile"] == "smoke"
    assert gate["rows"] >= 32
    evaluation = read_json(out_dir / train_reader.EVAL_FILE)
    assert evaluation["acceptance"]["overfit_loss_decreased"] is True


def test_overfit_smoke_only_writes_the_gate_and_nothing_else(cfg):
    summary = _reader_run(cfg, overfit_smoke_only=True)
    out_dir = Path(summary["reader_dir"])
    assert (out_dir / train_reader.GATE_FILE).exists()
    assert not (out_dir / "model.safetensors").exists()
    assert [row["status"] for row in read_jsonl(_registry(cfg).path)] == ["RUNNING", "SUCCEEDED"]


def test_acceptance_checks_are_truthful(cfg):
    spans = {
        ("q1", "d1"): BestSpan(qid="q1", doc_id="d1", answer_text="1975", start_char=0,
                               end_char=4, non_null_score=3.0, null_score=1.0),
        ("q2", "d2"): BestSpan(qid="q2", doc_id="d2", answer_text="", start_char=0, end_char=0,
                               non_null_score=NO_SPAN_SCORE, null_score=1.0),
    }
    rows = [{"qid": "q1", "is_impossible": False}, {"qid": "q2", "is_impossible": True}]
    docs = {"d1": "1975 là năm giải phóng miền Nam.", "d2": "Không có gì liên quan ở đây."}
    checks = train_reader.acceptance(
        {"n": 2}, spans, rows, threshold=0.0, model_hash="h", context_by_doc=docs,
        gate={"loss_decreased": True, "loss_first": 9.0, "loss_last": 1.0})
    assert checks["overfit_loss_decreased"] is True
    assert checks["non_empty_span_with_valid_offset"] == {
        "passed": True, "example": {"qid": "q1", "doc_id": "d1", "answer_text": "1975",
                                    "start_char": 0, "end_char": 4}}
    assert checks["impossible_returns_null"] == {"passed": True, "n_null_on_impossible": 1,
                                                 "example_qid": "q2"}
    assert checks["val_metrics_saved"] == {"passed": True, "n_val_questions": 2}

    # Offsets that do not reproduce the answer text are not offered as evidence.
    bad = {("q1", "d1"): BestSpan(qid="q1", doc_id="d1", answer_text="1975", start_char=2,
                                  end_char=6, non_null_score=3.0, null_score=1.0)}
    checks = train_reader.acceptance({"n": 1}, bad, [rows[0]], threshold=0.0, model_hash="h",
                                     context_by_doc=docs, gate=None)
    assert checks["non_empty_span_with_valid_offset"]["passed"] is False
    assert checks["overfit_loss_decreased"] is None


def test_acceptance_gate_fails_the_run_and_withholds_the_contract(cfg, monkeypatch):
    """D008: an acceptance check that fails must FAIL the run — not be recorded and ignored."""
    seen = {}

    def failing_checks(metrics_oracle, spans, rows, **kwargs):
        seen["threshold"] = kwargs["threshold"]
        return {"overfit_loss_decreased": None,
                "overfit_smoke": {"path": None, "loss_first": None, "loss_last": None},
                "non_empty_span_with_valid_offset": {"passed": False, "example": None},
                "impossible_returns_null": {"passed": True, "n_null_on_impossible": 1,
                                            "example_qid": "q"},
                "val_metrics_saved": {"passed": True, "n_val_questions": int(metrics_oracle["n"])}}

    monkeypatch.setattr(train_reader, "acceptance", failing_checks)
    with pytest.raises(RuntimeError, match="P3 acceptance failed"):
        _reader_run(cfg, skip_gate=True)
    out_dir = train_reader.reader_dir(cfg)
    assert seen["threshold"] == -0.4
    # The evidence stays on disk, but the frozen contract does not: nothing may consume it.
    assert (out_dir / train_reader.EVAL_FILE).exists()
    assert not (out_dir / train_reader.INFERENCE_FILE).exists()
    assert [row["status"] for row in read_jsonl(_registry(cfg).path)] == ["RUNNING", "FAILED"]


def test_require_acceptance_is_a_hard_gate():
    span_fail = {"non_empty_span_with_valid_offset": {"passed": False, "example": None},
                 "impossible_returns_null": {"passed": True, "n_null_on_impossible": 1,
                                             "example_qid": "q"}}
    with pytest.raises(RuntimeError, match="acceptance failed"):
        train_reader.require_acceptance(span_fail, threshold=-8.0)

    # D010: the null side is a hard gate too — the reader must not answer every negative.
    null_fail = {"non_empty_span_with_valid_offset": {"passed": True, "example": None},
                 "impossible_returns_null": {"passed": False, "n_null_on_impossible": 0,
                                             "example_qid": None}}
    with pytest.raises(RuntimeError, match="answer every negative"):
        train_reader.require_acceptance(null_fail, threshold=0.0)

    train_reader.require_acceptance(
        {"non_empty_span_with_valid_offset": {"passed": True, "example": None},
         "impossible_returns_null": {"passed": True, "n_null_on_impossible": 1,
                                     "example_qid": "q"}}, threshold=0.0)


def test_sweep_without_a_feasible_threshold_fails_the_run(cfg):
    """D2: no threshold emits a span -> the selection raises and the run is FAILED."""
    cfg["reader"]["threshold_sweep"] = {"start": -50.0, "stop": -50.0, "step": 1.0}
    with pytest.raises(ValueError, match="no threshold"):
        _reader_run(cfg, skip_gate=True)
    out_dir = train_reader.reader_dir(cfg)
    assert not (out_dir / train_reader.INFERENCE_FILE).exists()
    evaluation = read_json(out_dir / train_reader.EVAL_FILE)
    assert evaluation["status"] == "SELECTION_FAILED"
    assert evaluation["threshold_sweep"][0]["acceptance_constraint_met"] is False
    assert [row["status"] for row in read_jsonl(_registry(cfg).path)] == ["RUNNING", "FAILED"]


def test_sweep_that_answers_every_negative_fails_the_run(cfg):
    """D010 constraint 2: the old +8 grid answered every pair; that must now FAIL the run."""
    cfg["reader"]["threshold_sweep"] = {"start": 8.0, "stop": 8.0, "step": 1.0}
    with pytest.raises(ValueError, match="no threshold"):
        _reader_run(cfg, skip_gate=True)
    out_dir = train_reader.reader_dir(cfg)
    assert not (out_dir / train_reader.INFERENCE_FILE).exists()
    evaluation = read_json(out_dir / train_reader.EVAL_FILE)
    assert evaluation["status"] == "SELECTION_FAILED"
    assert evaluation["threshold_sweep"][0]["acceptance_constraint_met"] is False
    assert [row["status"] for row in read_jsonl(_registry(cfg).path)] == ["RUNNING", "FAILED"]


def test_gate_questions_come_from_train_strict(cfg):
    rows, dropped, docs = train_reader.load_rows(cfg, "train_strict", profile="smoke",
                                                 limit_override=train_reader.GATE_MIN_QUESTIONS)
    assert dropped == []
    features = build_feature_set(rows, docs, tokenizer=load_tokenizer(BASE), max_length=384,
                                doc_stride=128)
    assert len(features) >= 32
    assert all(row["qid"].startswith("train_strict:") for row in rows)


# ---------------------------------------------------------------------- state cache

class _StubInference:
    """Deterministic stand-in: answers only when the passage index matches the question's."""

    def __init__(self, contract):
        self.model_hash = contract["reader_model_hash"]
        self.tokenizer_sha = contract["tokenizer_sha"]

    @classmethod
    def load(cls, model_dir, **_kwargs):
        return cls(read_json(Path(model_dir) / train_reader.INFERENCE_FILE))

    def predict(self, pairs):
        import re
        out = {}
        for pair in pairs:
            number = re.search(r"\d{4}", pair.context)
            same_doc = pair.qid.rsplit("q", 1)[-1] == pair.doc_id.rsplit(":", 1)[-1]
            if same_doc and number:
                out[(pair.qid, pair.doc_id)] = BestSpan(
                    qid=pair.qid, doc_id=pair.doc_id, answer_text=number.group(),
                    start_char=number.start(), end_char=number.end(),
                    non_null_score=5.0, null_score=1.0)
            else:
                out[(pair.qid, pair.doc_id)] = BestSpan(
                    qid=pair.qid, doc_id=pair.doc_id, answer_text="", start_char=0, end_char=0,
                    non_null_score=NO_SPAN_SCORE, null_score=1.0)
        return out


def _write_contract(cfg, *, model_hash="r" * 64, tokenizer_sha="t" * 64, threshold=0.0):
    out_dir = train_reader.reader_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    contract = {"reader_model_hash": model_hash, "tokenizer_sha": tokenizer_sha,
                "inference_config_sha": "i" * 64, "null_score_diff_threshold": threshold,
                "max_length": cfg["reader"]["max_length"],
                "doc_stride": cfg["reader"]["doc_stride"],
                "max_answer_length": cfg["reader"]["max_answer_length"]}
    (out_dir / train_reader.INFERENCE_FILE).write_text(json.dumps(contract), encoding="utf-8")
    return contract


def test_cache_reader_writes_every_reachable_state(cfg, monkeypatch):
    contract = _write_contract(cfg)
    monkeypatch.setattr(cache_reader, "ReaderInference", _StubInference)
    for split, n in (("train_strict", TRAIN_N), ("val_strict", VAL_N)):
        _write_rankings(cfg, split, n)
    summary = cache_reader.run(cfg=cfg, profile="smoke", registry=_registry(cfg))

    for split, n in (("train_strict", TRAIN_N), ("val_strict", VAL_N)):
        parquet_path, meta_path = cache_reader.state_cache_paths(cfg, split)
        # Keyed by (qid, ordered state): the same visible-document tuple can be reached by
        # two questions (their corpora overlap), and the cache key separates them.
        rows = {(row["qid"], tuple(row["visible_doc_ids"])): row
                for row in _read_parquet(parquet_path)}
        expected = []
        for qid, ranking in _rankings(cfg, split, n).items():
            expected += [(qid, state) for state in reachable_states(
                ranking, bootstrap_tool="bm25", bootstrap_passages=1,
                passage_per_search_action=1, max_optional_search=2)]
        assert len(rows) == len(expected) == summary["splits"][split]["n_states"]
        assert set(rows) == set(expected)
        for qid, state in expected:
            row = rows[(qid, state)]
            assert row["cache_key"] == reader_cache_key(qid, state, contract["reader_model_hash"],
                                                        contract["tokenizer_sha"], "i" * 64)
            assert row["split"] == split and row["schema_version"] == state_cache.SCHEMA_VERSION
            assert row["n_visible_docs"] == len(state)
            assert set(row["visible_doc_ids"]) <= set(
                next(s for s in _split_rows(split, n)[2] if s["qid"] == qid)["corpus_doc_ids"])
        # The stub answers exactly when the question's own passage is visible, so each
        # state's candidate must follow that rule — this checks the per-state reduction
        # (threshold + best-across-docs) over the whole tree, null states included.
        for (qid, state), row in rows.items():
            own_visible = f"doc:{split}:{qid.rsplit('q', 1)[-1]}" in state
            assert row["is_null"] is not own_visible
        assert 0 < summary["splits"][split]["n_null_states"] < len(rows)

        meta = state_cache.load_state_cache_meta(meta_path)
        assert meta["split"] == split and meta["n_states"] == len(expected)
        assert meta["reader_model_hash"] == contract["reader_model_hash"]
        assert meta["inference_config_sha"] == "i" * 64
        assert meta["null_score_diff_threshold"] == 0.0
        assert meta["parquet"]["sha256"] == sha256_file(parquet_path)
        assert meta["creation_script_sha"] == sha256_file(Path(cache_reader.__file__))
        for source in meta["sources"].values():
            assert source["sha256"] == sha256_file(source["path"])
        assert meta["retrieval_budget"] == {"bootstrap_tool": "bm25", "bootstrap_passages": 1,
                                            "passage_per_search_action": 1, "optional_searches": 2}

        cache = state_cache.load_state_cache(parquet_path)
        assert len(cache) == len(expected)
        qid, state = expected[-1]
        hit = cache.lookup(state_cache.state_cache_key(meta, qid, state))
        assert hit.qid == qid and hit.non_null_score == rows[(qid, state)]["non_null_score"]
        with pytest.raises(KeyError, match="reader cache miss"):
            cache.lookup(reader_cache_key(qid, state + ("doc:absent",), meta["reader_model_hash"],
                                          meta["tokenizer_sha"], meta["inference_config_sha"]))

    assert not cache_reader.state_cache_paths(cfg, "test_strict")[0].exists()
    assert [row["status"] for row in read_jsonl(_registry(cfg).path)] == ["RUNNING", "SUCCEEDED"]
    assert summary["splits"]["val_strict"]["n_null_states"] > 0


def _read_parquet(path) -> list[dict]:
    import pyarrow.parquet as pq
    return pq.read_table(str(path)).to_pylist()


def test_test_cache_refuses_without_selection_before_any_write(cfg):
    manifest_dir = Path(cfg["paths"]["manifests"])
    assert not (manifest_dir / "best_selection.json").exists()
    with pytest.raises(RuntimeError, match="test_strict reader cache requires a frozen selection"):
        cache_reader.run(cfg=cfg, profile="smoke", splits=("test_strict",))
    assert not cache_reader.state_cache_paths(cfg, "test_strict")[0].exists()
    assert not (Path(cfg["paths"]["runs"]) / "index.jsonl").exists()
    overlay = manifest_dir.parent / "cache_reader_test_overlay.yaml"
    write_yaml(overlay, {"paths": cfg["paths"]})
    with pytest.raises(RuntimeError, match="test_strict reader cache requires a frozen selection"):
        cache_reader.main(["--profile", "smoke", "--splits", "test_strict",
                           "--config", str(overlay)])
    assert not (manifest_dir / "P3_reader_cache_smoke_config.yaml").exists()


def test_test_cache_after_freeze_reaches_reader_precondition(cfg):
    manifest_dir = Path(cfg["paths"]["manifests"])
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "best_selection.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="reader training step"):
        cache_reader.run(cfg=cfg, profile="smoke", splits=("test_strict",))


def test_cache_reader_needs_the_frozen_reader(cfg, monkeypatch):
    monkeypatch.setattr(cache_reader, "ReaderInference", _StubInference)
    with pytest.raises(FileNotFoundError, match="reader training step"):
        cache_reader.run(cfg=cfg, profile="smoke", registry=_registry(cfg))


def test_cache_reader_refuses_a_checkpoint_that_moved(cfg, monkeypatch):
    _write_contract(cfg)

    class _Moved(_StubInference):
        def __init__(self, contract):
            super().__init__(contract)
            self.model_hash = "0" * 64

    monkeypatch.setattr(cache_reader, "ReaderInference", _Moved)
    with pytest.raises(RuntimeError, match="no longer matches reader_inference.json"):
        cache_reader.run(cfg=cfg, profile="smoke", registry=_registry(cfg))


def test_cache_reader_reports_questions_without_a_ranking(cfg, monkeypatch):
    _write_contract(cfg)
    monkeypatch.setattr(cache_reader, "ReaderInference", _StubInference)
    _write_rankings(cfg, "train_strict", TRAIN_N)  # val_strict gets none
    with pytest.raises(FileNotFoundError, match="P2 retrieval step"):
        cache_reader.run(cfg=cfg, profile="smoke", splits=("val_strict",),
                         registry=_registry(cfg))

    _write_rankings(cfg, "val_strict", 1)  # only one question has a ranking
    summary = cache_reader.run(cfg=cfg, profile="smoke", splits=("val_strict",),
                               registry=_registry(cfg))
    assert summary["splits"]["val_strict"]["n_missing_ranking"] == VAL_N - 1


def test_cache_reader_refuses_a_ranking_outside_the_corpus(cfg, monkeypatch):
    """D011: a stale ranking row (built for another corpus) must fail the cache build."""
    _write_contract(cfg)
    monkeypatch.setattr(cache_reader, "ReaderInference", _StubInference)
    _write_rankings(cfg, "train_strict", TRAIN_N)
    path = train_reader.rankings_path(cfg, "train_strict")
    rows = list(read_jsonl(path))
    rows[0]["rankings"]["bm25"].append("doc:absent:999")
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="outside the episode corpus"):
        cache_reader.run(cfg=cfg, profile="smoke", splits=("train_strict",),
                         registry=_registry(cfg))


def test_best_state_prediction_prefers_a_survivor(cfg):
    from src.qa.reader import NULL_PREDICTION, ReaderPrediction

    def prediction(text, score, *, provenance_ok=True):
        return ReaderPrediction(qid="q", doc_id="d", answer_text=text, start_char=0,
                                end_char=1, non_null_score=score, null_score=1.0,
                                null_margin=1.0 - score, reader_model_hash="h",
                                provenance_ok=provenance_ok)

    assert state_cache.best_state_prediction([]) is NULL_PREDICTION
    nulled_but_high = prediction(None, 9.0)
    survivor = prediction("1975", 2.0)
    assert state_cache.best_state_prediction([nulled_but_high, survivor]) is survivor
    bad_provenance = prediction("1975", 8.0, provenance_ok=False)
    assert state_cache.best_state_prediction([bad_provenance, nulled_but_high]) is nulled_but_high


def test_state_cache_key_validates_the_meta(cfg):
    with pytest.raises(ValueError, match="reader_model_hash"):
        state_cache.state_cache_key({}, "q", ("d",))


def test_clis_parse(cfg, capsys):
    for module in (train_reader, cache_reader):
        with pytest.raises(SystemExit) as exit_info:
            module.main(["--help"])
        assert exit_info.value.code == 0
    assert "P3" in capsys.readouterr().out
