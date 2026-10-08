"""P1 acceptance tests — gates 1 and 2 (`02_IMPLEMENTATION/01_DOWNLOAD_VALIDATE_PREPROCESS.md` §8).

Gate 1: one positive and one negative row are read and normalized correctly.
Gate 2: ``context_raw[answer_start:answer_start+len(text)] == text``, or the row carries a
        *logged* recoverable/unrecoverable ``offset_issue`` — never a silent repair.

Plus the non-negotiable rule: gold labels stay in ``QuestionRecord.private`` and never reach
a public view (checked with :func:`src.data.schemas.find_forbidden_keys`).
"""

from __future__ import annotations

import json
import unicodedata

import pyarrow.parquet as pq
import pytest

from src.data.normalizer import RecordNormalizer, normalize_text
from src.data.schemas import ScenarioRecord, find_forbidden_keys
from src.utils.hashing import content_doc_id, normalize_title, sha256_text
from tests.fixtures import tiny_raw as fx

PRIVATE_SCENARIO_KEYS = {"gold_doc_id", "reachable_bm25", "reachable_dense", "difficulty"}


def normalize_row(row_id: str, *, occurrence: int = 0, split: str = "train"):
    matches = [r for r in fx.TRAIN_ROWS if r["id"] == row_id]
    return RecordNormalizer().normalize(matches[occurrence], split)


# ------------------------------------------------------------------ gate 1

def test_gate1_positive_row_is_normalized_with_its_gold_span():
    row = normalize_row("1")
    question = row.question
    assert question.qid == "train:1"
    assert question.source_split == "train"
    assert question.title_group == normalize_title("Alpha") == "alpha"
    assert question.question == "Thủ đô của Việt Nam là gì?"
    assert question.private.is_impossible is False
    assert question.private.answer_texts == ("Hà Nội",)
    assert question.private.answer_starts == (0,)
    assert question.private.gold_doc_id == content_doc_id(fx.ALPHA_CTX)
    assert row.document.doc_id == content_doc_id(fx.ALPHA_CTX)
    assert row.document.text == fx.ALPHA_CTX
    assert row.document.source_split == "train"
    assert row.offset_exact == 1
    assert row.offset_recovered == () and row.offset_unrecoverable == ()
    assert row.answer_starts_display == (0,)
    assert row.answer_texts_display == ("Hà Nội",)
    assert row.display_offsets_ok
    assert row.answers_field_missing is False


def test_gate1_negative_row_is_normalized_with_empty_gold():
    row = normalize_row("5")
    private = row.question.private
    assert private.is_impossible is True
    assert private.answer_texts == ()
    assert private.answer_starts == ()
    assert private.plausible_texts == ("Nguyễn Du",)
    assert private.gold_doc_id == content_doc_id(
        "Truyện Kiều là tác phẩm nổi tiếng của văn học Việt Nam.")
    assert row.n_gold_answers == 0
    assert row.answers_field_missing is True
    assert row.offset_exact == 0
    assert row.question.title_group == "alpha"


def test_impossible_row_with_gold_answers_is_rejected():
    example = dict(fx.TRAIN_ROWS[0], is_impossible=True)
    with pytest.raises(ValueError, match="is_impossible=True with non-empty gold"):
        RecordNormalizer().normalize(example, "train")


def test_answerable_row_without_gold_answers_is_rejected():
    with pytest.raises(ValueError, match="answerable row without gold answers"):
        normalize_row("8")


def test_empty_question_is_rejected():
    with pytest.raises(ValueError, match="empty question"):
        normalize_row("9")


def test_duplicate_id_is_suffixed_and_never_overwrites():
    normalizer = RecordNormalizer()
    first = normalizer.normalize(fx.TRAIN_ROWS[0], "train")
    second = normalizer.normalize(fx.TRAIN_ROWS[1], "train")
    assert first.question.qid == "train:1"
    assert second.question.qid.startswith("train:1#")
    assert first.question.qid != second.question.qid
    assert any(note.startswith("duplicate_id:1") for note in second.notes)
    # the duplicate keeps its own gold labels — nothing is overwritten or merged
    assert second.question.private.answer_texts == first.question.private.answer_texts


# ------------------------------------------------------------------ gate 2

def test_gate2_offset_on_the_space_before_the_answer_is_recovered_and_logged():
    row = normalize_row("2")
    assert row.offset_exact == 0
    assert len(row.offset_recovered) == 1
    assert row.offset_recovered[0].startswith("offset_recovered_space_prefix:8->9")
    assert row.question.private.answer_starts == (9,)          # repaired, not silent
    assert row.question.private.offset_issue == row.offset_recovered[0]
    assert row.answer_starts_display == (9,)
    assert row.document.text[9:9 + len("Việt Nam")] == "Việt Nam"


def test_gate2_unique_occurrence_elsewhere_is_recovered_and_logged():
    row = normalize_row("3")
    assert row.offset_exact == 0
    assert row.offset_recovered == ("offset_recovered_unique_match:11->5",)
    assert row.question.private.answer_starts == (5,)
    assert row.document.text[5:5 + len("Mê Kông")] == "Mê Kông"
    assert row.answer_starts_display == (5,)


def test_gate2_unrecoverable_offset_is_reported_and_not_fixed():
    row = normalize_row("4")
    assert row.offset_exact == 0
    assert row.offset_recovered == ()
    assert row.offset_unrecoverable == ("offset_unrecoverable:no_match:3",)
    assert row.question.private.answer_starts == (3,)           # left exactly as stored
    assert "offset_unrecoverable:no_match:3" in (row.question.private.offset_issue or "")
    assert row.answer_starts_display == ()


def test_gate2_every_gold_span_is_either_exact_or_logged():
    """Sweep: no fixture row may hold an unverified gold span without a recorded issue."""
    normalizer = RecordNormalizer()
    checked = 0
    for example in fx.TRAIN_ROWS:
        try:
            row = normalizer.normalize(example, "train")
        except ValueError:
            continue
        checked += 1
        assert row.offset_exact + len(row.offset_recovered) + len(row.offset_unrecoverable) \
            == row.n_gold_answers
        for text, start in zip(row.question.private.answer_texts,
                               row.question.private.answer_starts):
            if row.question.private.offset_issue:
                continue                                        # logged repair / failure
            assert row.context_raw[start:start + len(text)] == text
        for start, text in zip(row.answer_starts_display, row.answer_texts_display):
            assert row.document.text[start:start + len(text)] == text
    assert checked >= 20


def test_nfd_context_keeps_offsets_valid_against_the_display_text():
    row = normalize_row("7")
    assert not unicodedata.is_normalized("NFC", row.context_raw)
    assert unicodedata.is_normalized("NFC", row.document.text)
    assert row.context_raw != row.document.text                  # display really changed
    assert row.document.text == normalize_text(fx.NFD_CTX)
    assert row.offset_exact == 1                                 # raw span is exact
    assert row.display_offsets_ok
    start = row.answer_starts_display[0]
    text = row.answer_texts_display[0]
    assert unicodedata.is_normalized("NFC", text)
    assert row.document.text[start:start + len(text)] == text == "Hoàng Sa"
    # the raw offset is still valid against the raw context
    raw_start = row.question.private.answer_starts[0]
    raw_text = row.question.private.answer_texts[0]
    assert row.context_raw[raw_start:raw_start + len(raw_text)] == raw_text


# ------------------------------------------------------------------ label isolation

def test_public_views_carry_no_gold_labels():
    row = normalize_row("1")
    assert set(row.question.public_view()) == {"qid", "question"}
    assert find_forbidden_keys(row.question.public_view()) == []
    assert find_forbidden_keys(row.document.to_public_dict()) == []
    scenario = ScenarioRecord(
        scenario_id="sc:train_strict:train:1:v1", qid=row.question.qid, split="train_strict",
        corpus_doc_ids=("a", "b"), scenario_seed=1,
        gold_doc_id=row.question.private.gold_doc_id, difficulty={"answerable": True})
    assert find_forbidden_keys(scenario.public_meta()) == []
    # the scanner itself does detect labels, so the checks above are meaningful
    assert find_forbidden_keys({"is_impossible": True}) == ["is_impossible"]
    assert set(find_forbidden_keys(row.question.to_row())) >= {"answer_texts", "is_impossible"}


def test_parquet_label_store_roundtrip_and_label_free_documents(tmp_path):
    # profile "full" (no cap) so every pool row is emitted, not a stratified subsample
    manifest, cfg = fx.run_fixture_pipeline(tmp_path, profile="full", caps={"full": None})
    processed = tmp_path / "processed"

    tables = {name: pq.read_table(processed / f"questions_{name}_strict.parquet").to_pylist()
              for name in ("train", "val", "test")}
    by_qid = {r["qid"]: r for rows in tables.values() for r in rows}
    # the group allocation decides which strict split holds Alpha's rows; find it, don't assume
    positive = by_qid["train:1"]
    strict_split = positive["strict_split"]
    assert positive["question"] == "Thủ đô của Việt Nam là gì?"
    assert positive["answer_texts"] == ["Hà Nội"]
    assert positive["answer_starts_raw"] == [0]
    assert positive["answer_starts_display"] == [0]
    assert positive["is_impossible"] is False
    assert positive["source_split"] == "train"
    assert positive["gold_doc_id"] == content_doc_id(fx.ALPHA_CTX)

    negative = next(r for r in tables[strict_split.replace("_strict", "")]
                    if r["qid"].startswith("train:5"))
    assert negative["is_impossible"] is True
    assert negative["answer_texts"] == []
    assert negative["plausible_texts"] == ["Nguyễn Du"]

    docs = pq.read_table(processed / f"docs_{strict_split}.parquet").to_pylist()
    assert docs and all(set(r) == {"doc_id", "source_split", "strict_split", "title_group",
                                   "text", "text_raw"} for r in docs)
    for doc in docs:
        assert doc["strict_split"] == strict_split
        assert doc["source_split"] == "train"
        assert doc["doc_id"] == content_doc_id(doc["text_raw"])
        assert find_forbidden_keys(doc) == []
    assert manifest["strict_split"]["splits"][strict_split]["docs_emitted"] == len(docs)


def test_scenario_rows_keep_labels_out_of_the_public_part(tmp_path):
    fx.run_fixture_pipeline(tmp_path)
    lines = (tmp_path / "processed" / "scenarios_train.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in lines.splitlines() if line.strip()]
    assert rows
    for row in rows:
        public = {k: v for k, v in row.items() if k not in PRIVATE_SCENARIO_KEYS}
        # `corpus_doc_ids` is environment-side (the agent may see the corpus it retrieves
        # from); gold labels are not, and must never appear here.
        assert set(public) == {"scenario_id", "qid", "split", "corpus_size", "corpus_doc_ids",
                               "scenario_seed"}
        assert find_forbidden_keys(public) == []
        assert row["split"] == "train_strict"
        assert row["corpus_size"] == len(row["corpus_doc_ids"])
        assert row["scenario_seed"] == 42
        # private diagnostics are present, and only for the env/analysis side
        assert set(PRIVATE_SCENARIO_KEYS) <= set(row)


def test_unmappable_display_offset_is_reported_not_silently_dropped():
    """A gold span that collapses away in the display text must be reported, not dropped."""
    example = {"id": "x", "title": "Alpha", "context": "A  B", "question": "Ký tự nào?",
               "answers": {"text": [" "], "answer_start": [1]}, "is_impossible": False,
               "plausible_answers": None}
    row = RecordNormalizer().normalize(example, "train")
    assert row.offset_exact == 1                      # the raw span is exact
    assert row.answer_starts_display == ()
    assert not row.display_offsets_ok
    assert "display_offsets_incomplete" in (row.question.private.offset_issue or "")
    assert any(note.startswith("display_offset_unmappable:") for note in row.notes)


def test_raw_context_hash_is_of_the_raw_text():
    row = normalize_row("1")
    assert row.question.private.offset_issue is None
    assert row.display_offsets_ok
    assert sha256_text(row.context_raw) == row.context_sha_raw
