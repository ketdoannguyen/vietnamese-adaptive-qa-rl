"""Tests for the P9 demo CLI (``src/inference/cli.py``) — stubs only, no models, no GPU.

The reader and the Laya adapter are stubbed, but everything else on the path is real: the
episode runs through :class:`src.rl.env.AdaptiveQAEnv`, the probability cache is a real
:class:`src.decision.laya_cache.LayaProbCache`, and the policy path uses a real
:class:`src.rl.policy.PolicyNet`. The assertions cover the spec-07 §1/§2 contract: extractive
answers with citations, the exact abstention message, the action-probability label, and the
absence of any gold/label field in the result.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.data.schemas import find_forbidden_keys
from src.decision.laya_cache import LayaProbCache
from src.inference import cli
from src.qa.postprocess import BestSpan
from src.rl.features import FeatureExtractor, FeatureScaler
from src.rl.policy import PolicyNet
from src.utils.config import load_config

DOC_A = "Phạm Văn Đồng giữ chức Thủ tướng Chính phủ Việt Nam từ năm 1955 đến năm 1987."
DOC_B = "Hà Nội là thủ đô của Việt Nam từ năm 1945."
DOC_C = "Phở là món ăn truyền thống của Việt Nam."
DOCS = {"doc_a": DOC_A, "doc_b": DOC_B, "doc_c": DOC_C}
TITLES = {doc_id: f"T-{doc_id}" for doc_id in DOCS}
QUESTION = "Phạm Văn Đồng giữ chức Thủ tướng từ năm nào?"
#: Spans the stub reader is allowed to find, keyed by document (deliberately contradictory).
ANSWERS = {"doc_a": "1955", "doc_b": "1945"}


class StubReader:
    """Extractive stub: returns the span of ``ANSWERS[doc_id]`` when the question matches."""

    model_hash = "stub-reader-hash"

    def __init__(self, *, answerable: bool = True) -> None:
        self.answerable = answerable
        self.calls = 0

    def predict(self, pairs):
        self.calls += 1
        out = {}
        for pair in pairs:
            text = ANSWERS.get(pair.doc_id) if self.answerable else None
            if text is None or text not in pair.context:
                out[(pair.qid, pair.doc_id)] = BestSpan(
                    qid=pair.qid, doc_id=pair.doc_id, answer_text=None, start_char=0, end_char=0,
                    non_null_score=-6.0, null_score=6.0)
                continue
            start = pair.context.index(text)
            out[(pair.qid, pair.doc_id)] = BestSpan(
                qid=pair.qid, doc_id=pair.doc_id, answer_text=text, start_char=start,
                end_char=start + len(text), non_null_score=4.0, null_score=-4.0)
        return out


class StubLaya:
    """Deterministic prior preferring SEARCH_BM25 while the budget lasts, then ANSWER."""

    def __init__(self, probs=(0.30, 0.55, 0.05, 0.10)) -> None:
        self.probs = np.array(probs, dtype=np.float64)
        self.requests = 0

    def model_hash(self) -> str:
        return "stub-laya-hash"

    def predict_probs(self, states):
        self.requests += len(states)
        return np.tile(self.probs, (len(states), 1))


def make_corpus_jsonl(tmp_path, docs=None, titles=None):
    path = tmp_path / "corpus.jsonl"
    docs = DOCS if docs is None else docs
    titles = TITLES if titles is None else titles
    path.write_text("\n".join(json.dumps({"doc_id": doc_id, "text": docs[doc_id],
                                          "title": titles[doc_id]}) for doc_id in sorted(docs)),
                    encoding="utf-8")
    return path


def make_session(tmp_path, *, reader=None, laya=None, tools=("bm25",), policy=None,
                 scaler=None, docs=None) -> cli.Session:
    reader = reader or StubReader()
    return cli.Session(
        cfg=load_config(), corpus=cli.Corpus(docs=docs or DOCS, titles=TITLES, source="stub"),
        reader=reader, reader_model_hash=reader.model_hash, reader_threshold=0.0,
        decision_name="policy_reinforce_best" if policy is not None else "laya_rlcd_argmax",
        tools=tools, laya=laya, laya_cache=None if laya is None else LayaProbCache(tmp_path / "pc"),
        laya_model_hash=None if laya is None else laya.model_hash(), laya_regime="rlcd",
        policy=policy, scaler=scaler, extractor=FeatureExtractor() if policy else None)


def run(tmp_path, question=QUESTION, **kwargs):
    return cli.answer_question(make_session(tmp_path, **kwargs), question)


# ------------------------------------------------------------------------ result contract

def test_result_has_exactly_the_contract_keys(tmp_path):
    result = run(tmp_path, laya=StubLaya())
    assert set(result) == {"question", "answer", "abstained", "reason_short", "citations",
                           "retrieval_steps", "tools_used", "total_cost", "policy_name",
                           "model_hash", "stage_trace", "timings"}
    assert result["question"] == QUESTION
    assert result["policy_name"] == "laya_rlcd_argmax"
    assert result["model_hash"] == "stub-reader-hash"
    assert isinstance(result["stage_trace"], list) and result["stage_trace"]
    assert set(result["timings"]) == {"total_ms", *cli.TIMING_KEYS}


def test_answer_is_extracted_from_the_cited_document(tmp_path):
    result = run(tmp_path, laya=StubLaya())
    assert result["abstained"] is False and result["answer"] is not None
    assert len(result["citations"]) == 1
    citation = result["citations"][0]
    assert set(citation) == {"doc_id", "snippet", "title"}
    assert result["answer"] in DOCS[citation["doc_id"]]
    assert citation["snippet"].strip("…") in DOCS[citation["doc_id"]]
    assert citation["title"] == TITLES[citation["doc_id"]]


def test_retrieval_steps_and_cost_follow_the_environment(tmp_path):
    cfg = load_config()
    result = run(tmp_path, laya=StubLaya())
    # bootstrap (bm25) + two optional SEARCH_BM25 reveals, then ANSWER: 3 retrieval calls.
    assert result["retrieval_steps"] == 3
    assert result["tools_used"] == ["bm25"]
    reward = cfg["reward"]
    expected = (reward["bootstrap_bm25_cost"] + 2 * reward["bm25_cost"])
    assert result["total_cost"] == pytest.approx(expected)
    assert result["timings"]["retrieval_ms"] >= 0.0
    assert result["timings"]["laya_ms"] > 0.0  # the stub prior was consulted


def test_every_step_records_the_mask_action_and_costs(tmp_path):
    result = run(tmp_path, laya=StubLaya())
    steps = [entry for entry in result["stage_trace"] if entry["event"] == "step"]
    searches = [entry for entry in steps if entry["action"].startswith("SEARCH_")]
    # The bootstrap reveal happens inside reset and is not a decision; retrieval_steps
    # counts it, the step trace does not.
    assert len(searches) == result["retrieval_steps"] - 1
    assert [entry["action"] for entry in steps] == ["SEARCH_BM25", "SEARCH_BM25", "ANSWER"]
    for entry in steps:
        assert set(entry["action_probs"]) == set(cli.ACTION_ORDER)
        # recorded at 6-decimal precision, so the row sums to 1 only within that rounding
        assert sum(entry["action_probs"].values()) == pytest.approx(1.0, abs=1e-5)
        assert set(entry["action_mask"]) == set(cli.ACTION_ORDER)
    assert steps[0]["seen_doc_ids"] == ["doc_a"]  # forced bootstrap before the first decision
    assert steps[0]["step_cost"] == load_config()["reward"]["bm25_cost"]
    assert steps[-1]["step_cost"] == 0.0  # ANSWER is terminal, not a search
    assert steps[-1]["cost_so_far"] == result["total_cost"]
    assert steps[-1]["terminal"] is True


def test_action_probabilities_carry_their_own_label(tmp_path):
    result = run(tmp_path, laya=StubLaya())
    steps = [entry for entry in result["stage_trace"] if entry["event"] == "step"]
    for entry in steps:
        assert entry["action_probs_label"] == cli.ACTION_PROB_LABEL
    text = cli.format_result(result, show_trace=True)
    assert cli.ACTION_PROB_LABEL in text
    assert cli.ACTION_PROB_NOTE in text
    banned = {"p_correct", "answer_probability", "correctness", "xác suất đáp án đúng"}
    for entry in steps:
        assert not (banned & set(entry))


def test_no_gold_or_label_field_reaches_the_result(tmp_path):
    reader = StubReader()
    result = run(tmp_path, reader=reader, laya=StubLaya())
    assert find_forbidden_keys(result) == []
    setup = result["stage_trace"][0]
    assert find_forbidden_keys(setup) == []
    assert "reward_note" in setup


def test_abstain_message_is_exact_when_no_span_survives(tmp_path):
    result = run(tmp_path, reader=StubReader(answerable=False), laya=StubLaya())
    assert result["abstained"] is True and result["answer"] is None
    assert result["reason_short"] == ("Chưa tìm thấy bằng chứng đủ để trả lời trong bộ "
                                      "tài liệu hiện tại.")
    assert result["reason_short"] == cli.ABSTAIN_MESSAGE
    assert result["retrieval_steps"] == 3  # abstaining does not undo the searches already paid
    assert result["total_cost"] > 0.0


def test_abstain_result_cites_every_seen_passage(tmp_path):
    result = run(tmp_path, reader=StubReader(answerable=False), laya=StubLaya())
    cited = [citation["doc_id"] for citation in result["citations"]]
    steps = [entry for entry in result["stage_trace"] if entry["event"] == "step"]
    assert cited == steps[-1]["seen_doc_ids"]
    for citation in result["citations"]:
        assert citation["snippet"].strip("…") in DOCS[citation["doc_id"]]


def test_contradictory_evidence_is_flagged_not_silently_resolved(tmp_path):
    result = run(tmp_path, laya=StubLaya())
    final = result["stage_trace"][-1]
    flagged = {item["text"] for item in final["contradictory_evidence"]}
    assert flagged == {"1955", "1945"}
    assert final["contradictory_evidence_note"] == cli.CONTRADICTORY_NOTE
    # The spec allows flagging instead of a silent label-driven choice; the answer stands.
    assert result["abstained"] is False


def test_display_template_is_deterministic_and_extractive(tmp_path):
    result = run(tmp_path, laya=StubLaya())
    text = cli.format_result(result, show_trace=False)
    assert f"Theo tài liệu [{result['citations'][0]['doc_id']}], đáp án là: {result['answer']}" in text
    assert cli.ACTION_PROB_LABEL not in text  # probabilities only with --show-trace
    assert json.dumps(result, ensure_ascii=False)  # the contract stays JSON-serialisable


def test_policy_path_uses_features_scaler_and_reports_its_own_name(tmp_path):
    net = PolicyNet(FeatureExtractor().dim)
    net.save(tmp_path / "policy.safetensors")
    scaler = FeatureScaler(mean=[0.0] * FeatureExtractor().dim,
                           std=[1.0] * FeatureExtractor().dim)
    loaded = PolicyNet.load(tmp_path / "policy.safetensors")
    result = run(tmp_path, laya=StubLaya(), policy=loaded, scaler=scaler)
    assert result["policy_name"] == "policy_reinforce_best"
    assert result["timings"]["policy_ms"] > 0.0
    assert result["timings"]["laya_ms"] > 0.0  # the prior is a policy *feature*, not the actor


def test_session_without_a_decision_model_refuses_to_invent_one(tmp_path):
    session = make_session(tmp_path)
    with pytest.raises(cli.NotReadyError, match="never invents a decision rule"):
        session.decision({})


# ------------------------------------------------------------------------------ corpora

def test_test_split_is_refused_as_demo_corpus():
    with pytest.raises(ValueError, match="test_strict"):
        cli.load_corpus(load_config(), split="test_strict")


def test_user_corpus_jsonl_is_read_and_validated(tmp_path):
    corpus = cli.load_corpus(load_config(), corpus_jsonl=str(make_corpus_jsonl(tmp_path)))
    assert corpus.doc_ids == ["doc_a", "doc_b", "doc_c"]
    assert corpus.titles["doc_a"] == "T-doc_a"
    assert len(corpus.sha256) == 64

    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"doc_id": "d", "text": "a"}\n{"doc_id": "d", "text": "b"}\n',
                   encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate doc_id"):
        cli.load_corpus(load_config(), corpus_jsonl=str(bad))

    empty = tmp_path / "empty.jsonl"
    empty.write_text('{"doc_id": "d"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty 'text'"):
        cli.load_corpus(load_config(), corpus_jsonl=str(empty))

    with pytest.raises(FileNotFoundError, match="corpus file not found"):
        cli.load_corpus(load_config(), corpus_jsonl=str(tmp_path / "nope.jsonl"))


def test_corpus_limit_takes_the_lowest_doc_ids(tmp_path):
    corpus = cli.load_corpus(load_config(), corpus_jsonl=str(make_corpus_jsonl(tmp_path)),
                             limit=2)
    assert corpus.doc_ids == ["doc_a", "doc_b"]
    assert "first 2 docs" in corpus.source


# --------------------------------------------------------------------- missing artifacts

def _cfg_pointing_at(tmp_path):
    cfg = load_config()
    for key in ("models", "data_processed", "data_scenarios", "data_indexes", "data_cache"):
        cfg["paths"][key] = str(tmp_path / key)
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    return cfg


def _write_reader_contract(cfg, *, model_hash="h", threshold=0.0):
    """A minimal P3 contract, so the reader check passes and later checks can be reached."""
    reader_dir = Path(cfg["paths"]["models"]) / "qa_reader"
    reader_dir.mkdir(parents=True, exist_ok=True)
    (reader_dir / "reader_inference.json").write_text(json.dumps({
        "reader_model_hash": model_hash, "tokenizer_sha": "t",
        "inference_config_sha": "c",
        "null_score_diff_threshold": threshold}), encoding="utf-8")
    return reader_dir


def test_missing_reader_contract_names_the_p3_stage(tmp_path):
    cfg = _cfg_pointing_at(tmp_path)
    corpus = make_corpus_jsonl(tmp_path)
    with pytest.raises(cli.NotReadyError, match="train_reader"):
        cli.open_session(cfg, corpus_jsonl=str(corpus))


def test_policy_dir_accepts_the_spec_name_and_the_train_rl_name(tmp_path):
    cfg = _cfg_pointing_at(tmp_path)
    spec_dir = tmp_path / "models" / "policy_reinforce_best"
    code_dir = tmp_path / "models" / "rl_policy"
    # Nothing on disk: the error path must still name the spec's expected directory.
    assert cli._policy_dir(cfg) == spec_dir and not spec_dir.exists()
    code_dir.mkdir(parents=True)
    (code_dir / cli.POLICY_FILE).write_bytes(b"stub")  # existence is all _policy_dir checks
    assert cli._policy_dir(cfg) == code_dir  # train_rl's current directory name is found
    spec_dir.mkdir(parents=True)
    (spec_dir / cli.POLICY_FILE).write_bytes(b"stub")
    assert cli._policy_dir(cfg) == spec_dir  # the spec name wins when both exist
    cfg["inference"] = {"policy_dir": str(tmp_path / "custom")}
    assert cli._policy_dir(cfg) == tmp_path / "custom"  # explicit override always wins


def test_missing_policy_names_the_p6_stage_when_required(tmp_path):
    cfg = _cfg_pointing_at(tmp_path)
    cfg["inference"] = {"decision_rule": "policy"}
    _write_reader_contract(cfg)
    with pytest.raises(cli.NotReadyError, match="train_rl"):
        cli.open_session(cfg, corpus_jsonl=str(make_corpus_jsonl(tmp_path)))


def test_missing_laya_checkpoint_names_the_p5_stage(tmp_path):
    cfg = _cfg_pointing_at(tmp_path)
    cfg["inference"] = {"decision_rule": "laya"}
    _write_reader_contract(cfg)
    with pytest.raises(cli.NotReadyError, match="train_laya"):
        cli.open_session(cfg, corpus_jsonl=str(make_corpus_jsonl(tmp_path)),
                         laya_regime="rlcd")


# --------------------------------------------------------------------------------- main

def test_main_prints_one_json_line_per_question(tmp_path, monkeypatch, capsys):
    session = make_session(tmp_path, laya=StubLaya())
    monkeypatch.setattr(cli, "open_session", lambda *args, **kwargs: session)
    questions = tmp_path / "questions.txt"
    questions.write_text(f"{QUESTION}\n\nHà Nội là thủ đô từ năm nào?\n", encoding="utf-8")
    code = cli.main(["--question-file", str(questions), "--corpus-jsonl", "unused.jsonl",
                     "--json"])
    assert code == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 2
    for line in lines:
        assert set(json.loads(line)) == set(cli.answer_question(session, QUESTION))


def test_main_reports_a_missing_corpus_as_a_configuration_error(tmp_path, capsys):
    code = cli.main(["--question", QUESTION, "--corpus-jsonl", str(tmp_path / "missing.jsonl")])
    assert code == 3
    assert "corpus file not found" in capsys.readouterr().err


def test_main_requires_a_question(capsys):
    code = cli.main(["--corpus-jsonl", "unused.jsonl"])
    assert code == 3
    assert "nothing to answer" in capsys.readouterr().err
