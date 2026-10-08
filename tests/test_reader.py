"""P3 reader tests: feature alignment, SQuAD2 decoding, metrics, and a real loss decrease.

The tokenizer is the production one (``FacebookAI/xlm-roberta-base``) because offset
alignment is the whole point of these tests. The model is a 2-layer random XLM-R built from
a config, so the training path is exercised in seconds without a GPU — the 32-example
overfit gate on the real checkpoint is a pipeline step (``--overfit-smoke``).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.qa.features import build_feature_set, build_pair_features
from src.qa.postprocess import (NO_SPAN_SCORE, apply_threshold, best_prediction_per_question,
                                choose_best_threshold, decode_windows, sweep_threshold)
from src.qa.reader import ReaderPrediction, verify_provenance
from src.qa.reader_eval import evaluate_predictions, gold_index, threshold_grid
from src.qa.train_reader import (load_tokenizer, overfit_smoke, reader_model_hash,
                                 tokenizer_hash, train)

BASE = "FacebookAI/xlm-roberta-base"
CONTEXT = ("Hà Nội là thủ đô của Việt Nam từ năm 1945. "
           "Thành phố Hồ Chí Minh là đô thị lớn nhất về dân số. "
           "Phở là món ăn truyền thống của người Việt Nam.")
ANSWER = "1945"
ANSWER_START = CONTEXT.index(ANSWER)
QUESTION = "Hà Nội là thủ đô từ năm nào?"


@pytest.fixture(scope="session")
def tokenizer():
    return load_tokenizer(BASE)


def _tiny_model(tokenizer):
    from transformers import AutoModelForQuestionAnswering, XLMRobertaConfig
    config = XLMRobertaConfig(vocab_size=tokenizer.vocab_size, hidden_size=32,
                              num_hidden_layers=2, num_attention_heads=2,
                              intermediate_size=64, max_position_embeddings=512)
    return AutoModelForQuestionAnswering.from_config(config)


@pytest.fixture(scope="session")
def tiny_model(tokenizer):
    return _tiny_model(tokenizer)


# ------------------------------------------------------------------ feature alignment

def test_gold_span_maps_to_tokens_that_cover_it(tokenizer):
    features, metas = build_pair_features(
        QUESTION, CONTEXT, tokenizer, qid="q1", doc_id="d1", answer_texts=[ANSWER],
        answer_starts=[ANSWER_START], max_length=128, doc_stride=32)
    assert len(features) == len(metas) == 1
    meta = metas[0]
    assert meta.has_gold_span
    start, end = features[0].start_position, features[0].end_position
    assert 0 < start <= end
    covered = CONTEXT[meta.offset_mapping[start][0]:meta.offset_mapping[end][1]]
    assert ANSWER in covered


def test_impossible_question_is_cls_in_every_window(tokenizer):
    features, metas = build_pair_features(
        "Ai là tổng thống Pháp?", CONTEXT, tokenizer, qid="q2", doc_id="d1",
        is_impossible=True, max_length=128, doc_stride=32)
    assert features
    assert all((f.start_position, f.end_position) == (0, 0) for f in features)
    assert all(not meta.has_gold_span for meta in metas)


def test_long_context_splits_into_windows_and_only_the_covering_one_has_the_span(tokenizer):
    filler = "Một câu dài khác không liên quan gì đến câu hỏi. " * 40
    long_context = filler + "Câu trả lời nằm ở cuối: " + CONTEXT
    features, metas = build_pair_features(
        QUESTION, long_context, tokenizer, qid="q3", doc_id="d1", answer_texts=[ANSWER],
        answer_starts=[long_context.index(ANSWER)], max_length=128, doc_stride=64)
    assert len(features) > 1
    with_span = [i for i, meta in enumerate(metas) if meta.has_gold_span]
    assert len(with_span) >= 1
    for i, feature in enumerate(features):
        if i in with_span:
            assert feature.start_position > 0
        else:
            assert (feature.start_position, feature.end_position) == (0, 0)


def test_build_feature_set_skips_rows_without_a_usable_span(tokenizer):
    docs = {"d1": CONTEXT}
    rows = [
        {"qid": "ok", "question": QUESTION, "gold_doc_id": "d1", "is_impossible": False,
         "answer_texts": [ANSWER], "answer_starts": [ANSWER_START]},
        {"qid": "missing_doc", "question": QUESTION, "gold_doc_id": "nope",
         "is_impossible": False, "answer_texts": [ANSWER], "answer_starts": [0]},
        {"qid": "no_answer_text", "question": QUESTION, "gold_doc_id": "d1",
         "is_impossible": False, "answer_texts": [], "answer_starts": []},
        {"qid": "impossible", "question": "Ai?", "gold_doc_id": "d1", "is_impossible": True,
         "answer_texts": [], "answer_starts": []},
    ]
    fs = build_feature_set(rows, docs, tokenizer, max_length=128, doc_stride=32)
    assert sorted(fs.skipped_no_visible_span) == ["missing_doc", "no_answer_text"]
    assert len(fs) == 2
    assert {meta.qid for meta in fs.metas} == {"ok", "impossible"}


# ------------------------------------------------------------------ SQuAD2 decoding

def _logits(n_tokens: int, *, start: int, end: int, null_score: float = 0.0,
            span_score: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """One row of start/end logits: ``null_score`` at CLS, ``span_score`` at (start, end)."""
    start_logits = np.zeros((1, n_tokens), dtype=np.float32)
    end_logits = np.zeros((1, n_tokens), dtype=np.float32)
    start_logits[0, 0] = end_logits[0, 0] = null_score / 2
    start_logits[0, start] = end_logits[0, end] = span_score / 2
    return start_logits, end_logits


def test_decode_finds_the_span_and_applies_the_null_threshold(tokenizer):
    features, metas = build_pair_features(
        QUESTION, CONTEXT, tokenizer, qid="q1", doc_id="d1", answer_texts=[ANSWER],
        answer_starts=[ANSWER_START], max_length=128, doc_stride=32)
    contexts = {"d1": CONTEXT}
    logits = _logits(len(metas[0].sequence_ids), start=features[0].start_position,
                     end=features[0].end_position, span_score=10.0)

    spans = decode_windows(metas, *logits, contexts, max_answer_length=30)
    span = spans[("q1", "d1")]
    assert span.answer_text == ANSWER
    assert CONTEXT[span.start_char:span.end_char] == ANSWER

    answered = apply_threshold(spans.values(), threshold=0.0, reader_model_hash="sha",
                               context_by_doc=contexts)[("q1", "d1")]
    assert answered.answer_text == ANSWER and answered.provenance_ok
    assert answered.null_margin == -10.0  # the span beats null by 10

    # A threshold below the margin forces abstention while keeping the raw scores.
    abstained = apply_threshold(spans.values(), threshold=-100.0, reader_model_hash="sha",
                                context_by_doc=contexts)[("q1", "d1")]
    assert abstained.answer_text is None and abstained.null_margin == answered.null_margin


def test_window_without_a_usable_span_decodes_to_null(tokenizer):
    """A document with no tokenizable content yields an explicit null, not a dropped pair."""
    _, metas = build_pair_features(
        "Ai là tổng thống Pháp?", "   ", tokenizer, qid="q2", doc_id="d1",
        is_impossible=True, max_length=32, doc_stride=8)
    assert all(meta.context_span() is None for meta in metas)
    n = len(metas[0].sequence_ids)
    spans = decode_windows(metas, np.zeros((len(metas), n), dtype=np.float32),
                           np.zeros((len(metas), n), dtype=np.float32), {"d1": "   "})
    assert spans[("q2", "d1")].non_null_score == NO_SPAN_SCORE
    prediction = apply_threshold(spans.values(), threshold=0.0, reader_model_hash="sha",
                                 context_by_doc={"d1": "   "})[("q2", "d1")]
    assert prediction.answer_text is None


def test_provenance_rejects_a_span_that_is_not_in_the_document():
    bad = ReaderPrediction(qid="q", doc_id="d", answer_text="Sài Gòn", start_char=0,
                           end_char=6, non_null_score=1.0, null_score=0.0, null_margin=-1.0,
                           reader_model_hash="sha", provenance_ok=True)
    checked = verify_provenance(bad, CONTEXT)
    assert checked.provenance_ok is False and checked.answer_text is None
    assert checked.error == "provenance_error"


def test_best_prediction_per_question_ignores_null_and_unprovenanced():
    def pred(qid, score, text="1945", ok=True):
        return ReaderPrediction(qid=qid, doc_id="d", answer_text=text, start_char=0,
                                end_char=4, non_null_score=score, null_score=0.0,
                                null_margin=-score, reader_model_hash="sha", provenance_ok=ok)
    best = best_prediction_per_question([pred("q1", 1.0), pred("q1", 5.0),
                                         pred("q1", 9.0, ok=False), pred("q2", 1.0, text=None)])
    assert best["q1"].non_null_score == 5.0
    assert "q2" not in best


# -------------------------------------------------------------------------- metrics

GOLD = {
    "a1": {"answer_texts": ["1945"], "is_impossible": False},
    "a2": {"answer_texts": ["Phở"], "is_impossible": False},
    "i1": {"answer_texts": [], "is_impossible": True},
    "i2": {"answer_texts": [], "is_impossible": True},
}


def _prediction(qid, text):
    return ReaderPrediction(qid=qid, doc_id="d", answer_text=text,
                            start_char=0 if text else None, end_char=len(text) if text else None,
                            non_null_score=1.0, null_score=0.0, null_margin=-1.0,
                            reader_model_hash="sha", provenance_ok=True)


def test_evaluate_predictions_prices_answers_and_abstentions_separately():
    predictions = {"a1": _prediction("a1", "1945"),      # correct
                   "a2": _prediction("a2", "Bún chả"),   # wrong
                   "i1": _prediction("i1", "Pháp"),      # false answer
                   "i2": _prediction("i2", None)}        # correct abstention
    metrics = evaluate_predictions(predictions, GOLD)
    assert metrics["n"] == 4 and metrics["n_answerable"] == 2 and metrics["n_impossible"] == 2
    assert metrics["em_answerable"] == 0.5
    assert metrics["f1_answerable"] == pytest.approx(0.5)
    assert metrics["no_answer_accuracy"] == 0.5
    assert metrics["false_answer_rate"] == 0.5
    assert metrics["answer_rate_answerable"] == 1.0
    assert metrics["em_overall"] == pytest.approx(0.5)


def test_missing_prediction_counts_as_abstention():
    metrics = evaluate_predictions({}, GOLD)
    assert metrics["em_answerable"] == 0.0
    assert metrics["no_answer_accuracy"] == 1.0
    assert metrics["em_overall"] == 0.5


def test_gold_index_keeps_only_what_metrics_may_see():
    index = gold_index([{"qid": "q", "answer_texts": ["x"], "is_impossible": False,
                         "gold_doc_id": "secret"}])
    assert index == {"q": {"answer_texts": ["x"], "is_impossible": False}}


def test_sweep_picks_the_threshold_with_the_best_overall_em(tokenizer):
    features, metas = build_pair_features(
        QUESTION, CONTEXT, tokenizer, qid="a1", doc_id="d1", answer_texts=[ANSWER],
        answer_starts=[ANSWER_START], max_length=128, doc_stride=32)
    contexts = {"d1": CONTEXT}
    logits = _logits(len(metas[0].sequence_ids), start=features[0].start_position,
                     end=features[0].end_position, span_score=2.0)
    spans = decode_windows(metas, *logits, contexts)
    gold = gold_index([{"qid": "a1", "answer_texts": [ANSWER], "is_impossible": False}])

    sweep = sweep_threshold(spans, gold, threshold_grid(-4.0, 4.0, 0.5),
                            reader_model_hash="sha", context_by_doc=contexts,
                            evaluate=evaluate_predictions)
    best = choose_best_threshold(sweep)
    assert best["em_overall"] == 1.0
    assert best["threshold"] == -2.0  # the margin: below it the answer survives
    assert sweep[0]["em_overall"] >= sweep[-1]["em_overall"]


# -------------------------------------------------------------------------- training

def _training_rows(docs: dict[str, str]) -> list[dict]:
    rows = []
    specs = [(QUESTION, "d1", ANSWER),
             ("Phở là gì?", "d1", "Phở"),
             ("Thành phố nào đông dân nhất?", "d1", "Thành phố Hồ Chí Minh")]
    for i in range(12):
        for question, doc_id, answer in specs:
            rows.append({"qid": f"q{i}-{answer[:4]}", "question": question,
                         "gold_doc_id": doc_id, "is_impossible": False,
                         "answer_texts": [answer],
                         "answer_starts": [docs[doc_id].index(answer)]})
        rows.append({"qid": f"neg{i}", "question": "Ai là tổng thống Pháp?",
                     "gold_doc_id": "d1", "is_impossible": True,
                     "answer_texts": [], "answer_starts": []})
    return rows


def test_overfit_smoke_decreases_loss_on_32_examples(tokenizer, tiny_model, tmp_path):
    fs = build_feature_set(_training_rows({"d1": CONTEXT}), {"d1": CONTEXT}, tokenizer,
                           max_length=128, doc_stride=32)
    assert len(fs) >= 32, "the gate needs 32 windows"
    summary = overfit_smoke(model=tiny_model, tokenizer=tokenizer, rows=fs.rows(),
                            out_dir=tmp_path / "reader", steps=30, lr=5e-4, batch_size=4)
    assert summary["loss_first"] is not None and summary["loss_last"] is not None
    assert summary["loss_decreased"] is True, summary["loss_history"]
    assert (tmp_path / "reader" / "overfit_smoke.json").exists()


def test_saved_reader_is_hashed_by_its_weights(tokenizer, tmp_path):
    out = tmp_path / "reader"
    model = _tiny_model(tokenizer)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    first = reader_model_hash(out)
    assert first == reader_model_hash(out)  # stable
    model.qa_outputs.weight.data.add_(1.0)
    model.save_pretrained(out)
    assert reader_model_hash(out) != first  # a changed checkpoint invalidates the cache
    assert tokenizer_hash(tokenizer) == tokenizer_hash(tokenizer)


def test_training_arguments_recorded_in_summary(tokenizer, tiny_model, tmp_path):
    fs = build_feature_set(_training_rows({"d1": CONTEXT}), {"d1": CONTEXT}, tokenizer,
                           max_length=128, doc_stride=32)
    summary = train(model=tiny_model, tokenizer=tokenizer, rows=fs.rows(),
                    out_dir=tmp_path / "r2", epochs=1, lr=5e-4, batch_size=8, grad_accum=1,
                    max_steps=4, logging_steps=1, warmup_ratio=0.0)
    assert summary["steps"] == 4
    assert summary["effective_batch"] == 8
    assert len(summary["loss_history"]) >= 1
    assert summary["library_versions"]["torch"]
    assert (tmp_path / "r2" / "training_args.json").exists()
    assert (tmp_path / "r2" / "model.safetensors").exists()
    assert list((tmp_path / "r2").glob("tokenizer*")) or (tmp_path / "r2" / "sentencepiece.bpe.model").exists()
