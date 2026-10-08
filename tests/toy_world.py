"""Toy episodes for environment/reward tests.

These are synthetic fixtures only — they never count as experimental results
(`05_TEMPLATES/07_TOY_FIXTURES_AND_ACCEPTANCE_EXAMPLES.md`).
"""

from __future__ import annotations

from src.qa.reader import ReaderPrediction, verify_provenance
from src.rl.env import EpisodeWorld
from src.rl.reward import EpisodePrivate, RewardCalculator, RewardConfig

DOC_A = "Hà Nội là thủ đô của Việt Nam từ năm 1010 dưới triều Lý."
DOC_B = "Phạm Văn Đồng giữ chức Thủ tướng Chính phủ Việt Nam từ năm 1976."
DOC_C = "Hồ Chí Minh đọc Tuyên ngôn Độc lập tại quảng trường Ba Đình năm 1945."
DOC_D = "Đà Nẵng là thành phố lớn nhất miền Trung Việt Nam."


def make_reader(predictions: dict[tuple[str, ...], ReaderPrediction], docs: dict[str, str]):
    """Reader callable backed by an explicit state -> prediction table."""

    def _reader(seen: tuple[str, ...]) -> ReaderPrediction:
        pred = predictions.get(seen)
        if pred is None:
            return ReaderPrediction(
                qid="toy", doc_id=None, answer_text=None, start_char=None, end_char=None,
                non_null_score=float("-inf"), null_score=0.0, null_margin=0.0,
                reader_model_hash="toy", provenance_ok=False)
        return verify_provenance(pred, docs[pred.doc_id]) if pred.doc_id else pred

    return _reader


def span_prediction(qid: str, doc_id: str, doc_text: str, answer: str, score: float,
                    null_score: float = 0.0) -> ReaderPrediction:
    start = doc_text.index(answer)
    return ReaderPrediction(
        qid=qid, doc_id=doc_id, answer_text=answer, start_char=start,
        end_char=start + len(answer), non_null_score=score, null_score=null_score,
        null_margin=score - null_score, reader_model_hash="toy", provenance_ok=True)


def positive_world(*, gold_in_second_result: bool = True) -> EpisodeWorld:
    """Answerable question; the gold passage is reachable (optionally only on BM25 #2)."""
    docs = {"doc_a": DOC_A, "doc_b": DOC_B, "doc_c": DOC_C, "doc_d": DOC_D}
    if gold_in_second_result:
        bm25 = ["doc_a", "doc_b", "doc_c", "doc_d"]
    else:
        bm25 = ["doc_b", "doc_a", "doc_d", "doc_c"]
    rankings = {"bm25": bm25, "dense": ["doc_b", "doc_d", "doc_a", "doc_c"]}
    answer = "1976"
    predictions = {
        # Bootstrap surfaces doc_a, which contains no valid span for this question: the
        # agent must search once before it can answer. This is the "gold in second
        # BM25 result" scenario from spec 05 §9.
        ("doc_a", "doc_b"): span_prediction("toy:pos", "doc_b", DOC_B, answer, 5.0),
        ("doc_b",): span_prediction("toy:pos", "doc_b", DOC_B, answer, 5.0),
        ("doc_b", "doc_a"): span_prediction("toy:pos", "doc_b", DOC_B, answer, 5.0),
    }
    world = EpisodeWorld(
        qid="toy:pos", question="Phạm Văn Đồng giữ chức Thủ tướng từ năm nào?",
        split="train_strict", scenario_id="s_pos",
        corpus=docs, rankings=rankings,
        private=EpisodePrivate(qid="toy:pos", answer_texts=(answer,),
                               is_impossible=False, gold_doc_id="doc_b",
                               gold_in_corpus=True, gold_reachable=True),
        reader=make_reader(predictions, docs),
    )
    return world


def negative_world() -> EpisodeWorld:
    """Unanswerable question: no passage in the corpus contains the answer."""
    docs = {"doc_a": DOC_A, "doc_c": DOC_C, "doc_d": DOC_D}
    rankings = {"bm25": ["doc_a", "doc_c", "doc_d"], "dense": ["doc_c", "doc_d", "doc_a"]}
    # A correctly-behaving reader returns no answer for an unanswerable question, so
    # every reachable state has a null candidate and ANSWER stays masked throughout.
    predictions: dict[tuple[str, ...], ReaderPrediction] = {}
    world = EpisodeWorld(
        qid="toy:neg", question="Ai là tổng thống Hoa Kỳ năm 1865?",
        split="train_strict", scenario_id="s_neg",
        corpus=docs, rankings=rankings,
        private=EpisodePrivate(qid="toy:neg", answer_texts=(), is_impossible=True,
                               gold_doc_id=None, gold_in_corpus=False, gold_reachable=None),
        reader=make_reader(predictions, docs),
    )
    return world


def default_calculator(**overrides) -> RewardCalculator:
    config = RewardConfig(**overrides) if overrides else RewardConfig()
    return RewardCalculator(config)
