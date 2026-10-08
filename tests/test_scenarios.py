"""Hard-negative corpora: the spec-required BM25-hard mix (`02` §6, D012).

The v1 corpora were random-only, so the bootstrap (BM25 top-1) revealed the gold document
for ~98% of answerable questions and SEARCH was never worth its cost; the teacher oracle
was correct, the corpora were too easy. These tests pin the fix: a configurable fraction of
distractor slots is filled with the other-title pool documents the *question* scores
highest under the same BM25 the episode ranks with — and nothing else (the gold answer
never enters selection).
"""

from __future__ import annotations

from src.data.normalizer import NormalizedRow
from src.data.scenarios import SCENARIO_VERSION, build_scenarios, scenario_summary
from src.data.schemas import DocumentRecord, QuestionPrivate, QuestionRecord
from src.retrieval.bm25 import BM25Index

SPLIT = "train_strict"
QUESTION = "Thủ đô của Việt Nam là gì?"

#: The gold document shares only "thủ đô" with the question; doc_hard repeats the question
#: almost verbatim under a different title, so the reference BM25 must rank it first.
DOCS = {
    "doc_gold": "Thủ đô là Hà Nội.",
    "doc_hard": "Thủ đô của Việt Nam là gì? Nhiều nơi từng là thủ đô của Việt Nam.",
    "doc_mild": "Việt Nam có nhiều đô thị lớn.",
    "doc_far_1": "Phở là món ăn truyền thống.",
    "doc_far_2": "Sông Hồng chảy qua đồng bằng Bắc Bộ.",
    "doc_far_3": "Cà phê được trồng nhiều ở Đắk Lắk.",
}
TITLES = {"doc_gold": "Alpha", "doc_hard": "Beta", "doc_mild": "Beta",
          "doc_far_1": "Gamma", "doc_far_2": "Gamma", "doc_far_3": "Gamma"}


def _row(doc_id: str, *, question: str = QUESTION) -> NormalizedRow:
    document = DocumentRecord(doc_id=doc_id, title_group=TITLES[doc_id],
                              text=DOCS[doc_id], source_split="train")
    return NormalizedRow(
        question=QuestionRecord(
            qid="train:q1", source_split="train", title_group=TITLES[doc_id],
            question=question,
            private=QuestionPrivate(answer_texts=("Hà Nội",), answer_starts=(12,),
                                    is_impossible=False, gold_doc_id=doc_id),
            strict_split=SPLIT),
        document=document, context_raw=DOCS[doc_id], title_raw=TITLES[doc_id],
        context_sha_raw=f"sha:{doc_id}", answers_field_missing=False, n_gold_answers=1,
        offset_exact=1)


def _pool() -> dict[str, DocumentRecord]:
    return {doc_id: _row(doc_id).document for doc_id in DOCS}


def _reference_top_other_title(n: int) -> list[str]:
    """Top-``n`` pool documents for the question, gold and its title excluded."""
    index = BM25Index.build({d: DOCS[d] for d in DOCS}, k1=1.5, b=0.75)
    ranked = [doc_id for doc_id, _ in index.rank(QUESTION)
              if doc_id != "doc_gold" and TITLES[doc_id] != TITLES["doc_gold"]]
    return ranked[:n]


def test_hard_slots_take_the_top_bm25_other_title_documents():
    expected = _reference_top_other_title(2)
    scenarios, _ = build_scenarios([_row("doc_gold")], _pool(), split=SPLIT,
                                   n_distractors=2, scenario_seed=42,
                                   hard_negative_ratio=1.0)
    scenario = scenarios[0]
    assert expected == ["doc_hard", "doc_mild"]  # the pool is built to have this order
    assert set(expected) <= set(scenario.corpus_doc_ids)
    assert scenario.difficulty["n_distractors_hard"] == 2
    assert scenario.difficulty["n_distractors_random"] == 0
    assert scenario.difficulty["hard_mean_score"] > 0
    # every hard pick is strictly other-title, and the gold doc stays in the corpus
    assert "doc_gold" in scenario.corpus_doc_ids
    assert all(TITLES[d] != "Alpha" for d in scenario.corpus_doc_ids
               if d != "doc_gold")


def test_half_ratio_splits_slots_between_hard_and_uniform():
    scenarios, _ = build_scenarios([_row("doc_gold")], _pool(), split=SPLIT,
                                   n_distractors=4, scenario_seed=42,
                                   hard_negative_ratio=0.5)
    difficulty = scenarios[0].difficulty
    assert difficulty["n_distractors_hard"] == 2
    assert difficulty["n_distractors_random"] == 2
    assert difficulty["n_distractors_placed"] == 4


def test_zero_ratio_disables_the_reference_index_and_the_mix():
    scenarios, _ = build_scenarios([_row("doc_gold")], _pool(), split=SPLIT,
                                   n_distractors=2, scenario_seed=42,
                                   hard_negative_ratio=0.0)
    difficulty = scenarios[0].difficulty
    assert difficulty["n_distractors_hard"] == 0
    assert difficulty["n_distractors_random"] == 2
    assert difficulty["hard_mean_score"] is None


def test_corpora_are_deterministic_for_a_given_pool_seed_and_version():
    kwargs = dict(split=SPLIT, n_distractors=3, scenario_seed=42, hard_negative_ratio=0.5)
    first, _ = build_scenarios([_row("doc_gold")], _pool(), **kwargs)
    second, _ = build_scenarios([_row("doc_gold")], _pool(), **kwargs)
    assert first[0].corpus_doc_ids == second[0].corpus_doc_ids
    assert first[0].difficulty == second[0].difficulty
    assert first[0].scenario_id.endswith(f":v{SCENARIO_VERSION}")


def test_selection_never_reads_the_gold_answer():
    """Same question, different gold answer text -> identical corpus (no label leak)."""
    kwargs = dict(split=SPLIT, n_distractors=3, scenario_seed=42, hard_negative_ratio=0.5)
    with_answer, _ = build_scenarios([_row("doc_gold")], _pool(), **kwargs)

    def _row_other_answer(doc_id: str) -> NormalizedRow:
        row = _row(doc_id)
        private = QuestionPrivate(answer_texts=("Sài Gòn",), answer_starts=(0,),
                                  is_impossible=False, gold_doc_id=doc_id)
        question = QuestionRecord(qid=row.question.qid, source_split="train",
                                  title_group=row.question.title_group,
                                  question=row.question.question, private=private,
                                  strict_split=SPLIT)
        return NormalizedRow(question=question, document=row.document,
                             context_raw=row.context_raw, title_raw=row.title_raw,
                             context_sha_raw=row.context_sha_raw,
                             answers_field_missing=False, n_gold_answers=1,
                             offset_exact=1)

    with_other_answer, _ = build_scenarios([_row_other_answer("doc_gold")], _pool(), **kwargs)
    assert with_answer[0].corpus_doc_ids == with_other_answer[0].corpus_doc_ids


def test_summary_records_the_hardness_distribution():
    scenarios, _ = build_scenarios([_row("doc_gold")], _pool(), split=SPLIT,
                                   n_distractors=2, scenario_seed=42,
                                   hard_negative_ratio=0.5)
    summary = scenario_summary(scenarios)
    assert summary["hard_distractors_total"] == 1
    assert summary["scenarios_with_hard_distractors"] == 1
    assert summary["hard_mean_score_avg"] > 0
