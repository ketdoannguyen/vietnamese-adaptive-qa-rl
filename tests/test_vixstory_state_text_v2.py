"""Evidence retention is measured before, not just after, Laya's final clamp."""

import pytest

from src.decision.vixstory_state_text_v2 import (
    HEAD_MAX_LEN,
    MAX_LEN,
    STATE_TEXT_VERSION,
    render_with_retention,
    serialize_vixstory_state,
)


class TokenizerStub:
    mask_token = "<mask>"
    mask_token_id = 3
    cls_token_id = 1
    sep_token_id = 2

    def __init__(self):
        self.vocab = {}

    def __call__(self, text, **kwargs):
        return {"input_ids": [self.vocab.setdefault(word, len(self.vocab) + 4)
                              for word in text.split()]}


def observation(n=5, **changes):
    obs = {"qid": "train-1", "question": "Địa điểm nào?",
           "seen_doc_ids": [f"d{i}" for i in range(n)],
           "seen_doc_texts": [f"nguồn{i} " * 300 for i in range(n)],
           "history": ["SEARCH_BM25", "SEARCH_DENSE"],
           "optional_used": 2, "remaining_optional_searches": 0,
           "action_mask": [True, False, False]}
    obs.update(changes)
    return obs


def test_every_passage_has_content_and_measured_loss():
    text, audit = render_with_retention(observation(), tokenizer=TokenizerStub(),
                                       max_len=220, head_max_len=80)
    assert (STATE_TEXT_VERSION, MAX_LEN, HEAD_MAX_LEN) == (
        "evidence-retrieval-v2-len8192", 8192, 192,
    )
    assert audit["truncated"] is False
    assert audit["essential_fields_present"] is True
    assert len(audit["passages"]) == 5
    for index, passage in enumerate(audit["passages"]):
        assert f"doc=d{index} nguồn{index}" in text
        assert passage["full_tokens"] == 300
        assert 0 < passage["retained_tokens"] < 300
        assert 0 < passage["retention"] < 1
        assert passage["cut"] is True
    assert "SEARCH_BM25 > SEARCH_DENSE" in text
    assert "Hành động hợp lệ: STOP" in text


def test_full_short_passages_and_single_serializer_path():
    obs = observation(1, seen_doc_texts=["Đoạn ngắn nguyên vẹn."])
    text, audit = render_with_retention(obs, tokenizer=TokenizerStub())
    assert audit["passages"][0]["retention"] == 1
    assert audit["passages"][0]["cut"] is False
    assert serialize_vixstory_state(obs, tokenizer=TokenizerStub()) == text


def test_five_passages_that_fit_are_never_cut():
    texts = [f"nguồn{i} " * 300 for i in range(5)]
    text, audit = render_with_retention(observation(seen_doc_texts=texts), tokenizer=TokenizerStub())
    assert audit["truncated"] is False
    assert all(not passage["cut"] and passage["retention"] == 1 for passage in audit["passages"])
    assert all(f"nguồn{i}" in text for i in range(5))


def test_long_first_passage_is_not_cut_when_whole_state_fits():
    texts = ["long " * 1000] + [f"short{i} " * 20 for i in range(4)]
    _, audit = render_with_retention(observation(seen_doc_texts=texts), tokenizer=TokenizerStub(),
                                     max_len=1600, head_max_len=192)
    assert [passage["cut"] for passage in audit["passages"]] == [False] * 5
    assert audit["passages"][0]["retained_tokens"] == 1000


def test_whitespace_word_longer_than_equal_share_survives_when_full_state_fits():
    texts = ["https://example.org/" + "x" * 700] + [f"short{i} " * 20 for i in range(4)]
    _, audit = render_with_retention(observation(seen_doc_texts=texts), tokenizer=TokenizerStub(),
                                     max_len=240, head_max_len=80)
    assert all(not passage["cut"] and passage["retained_tokens"] > 0
               for passage in audit["passages"])


def test_overbudget_state_uses_deterministic_nonempty_prefixes():
    obs = observation(seen_doc_texts=[f"source{i} " * 2500 for i in range(5)])
    first = render_with_retention(obs, tokenizer=TokenizerStub())
    second = render_with_retention(obs, tokenizer=TokenizerStub())
    assert first == second
    assert all(passage["cut"] and passage["retained_tokens"] > 0
               for passage in first[1]["passages"])
    assert first[1]["truncated"] is False


@pytest.mark.parametrize("field", ["gold_context_ids", "context_ids", "answer", "coverage", "q_values"])
def test_reject_private_fields(field):
    with pytest.raises(ValueError):
        serialize_vixstory_state(observation(**{field: ["private"]}), tokenizer=TokenizerStub())


def test_reject_overbudget_essential_fields():
    with pytest.raises(ValueError, match="essential"):
        serialize_vixstory_state(observation(question="long " * 500), tokenizer=TokenizerStub(), max_len=220)


def test_reject_missing_content_instead_of_silently_dropping_passage():
    with pytest.raises(ValueError, match="empty|content"):
        serialize_vixstory_state(observation(1, seen_doc_texts=[""]), tokenizer=TokenizerStub())


def test_legitimate_gold_words_in_public_passages_are_not_blocked():
    text = serialize_vixstory_state(observation(1, seen_doc_texts=["gold_context_ids answer reward"]),
                                   tokenizer=TokenizerStub())
    assert "gold_context_ids answer reward" in text


def test_real_pinned_tokenizer_keeps_five_passages_that_fit():
    from pathlib import Path

    from transformers import AutoTokenizer

    path = (Path.home() / ".cache/huggingface/hub/models--convaiinnovations--laya-multilingual"
            / "snapshots/e4e9ddf21a7b1903b7acffd8814ad4307bf63a67/tokenizer")
    if not path.is_dir():
        pytest.skip("pinned Laya tokenizer not cached")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    text, audit = render_with_retention(observation(), tokenizer=tokenizer)
    assert audit["head_options_distinct"] == 3
    assert audit["truncated"] is False
    assert audit["essential_fields_present"] is True
    assert all(p["retained_tokens"] > 0 and not p["cut"] for p in audit["passages"])
    assert all(f"doc=d{i}" in text for i in range(5))


def test_real_tokenizer_keeps_long_whitespace_url_when_full_state_fits():
    from pathlib import Path

    from transformers import AutoTokenizer

    path = (Path.home() / ".cache/huggingface/hub/models--convaiinnovations--laya-multilingual"
            / "snapshots/e4e9ddf21a7b1903b7acffd8814ad4307bf63a67/tokenizer")
    if not path.is_dir():
        pytest.skip("pinned Laya tokenizer not cached")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    url = "https://example.org/" + "x" * 700
    texts = [url] + [f"nguồn{i} ít từ" for i in range(4)]
    text, audit = render_with_retention(observation(seen_doc_texts=texts),
                                        tokenizer=tokenizer, max_len=1024, head_max_len=192)
    assert url in text
    assert all(not passage["cut"] for passage in audit["passages"])
    assert audit["passages"][0]["retained_tokens"] > 1
    assert not audit["truncated"]


def test_mask_literal_in_evidence_uses_laya_token_accounting():
    from laya.common import build_sequence
    from laya.train import to_internal

    from src.decision.schema_v2 import build_questions_field

    tokenizer = TokenizerStub()
    text, audit = render_with_retention(
        observation(1, seen_doc_texts=["<mask> nguồn giữ nguyên"]), tokenizer=tokenizer,
    )
    question = to_internal("next_action", build_questions_field()["next_action"])
    _, markers, head_stats, stats = build_sequence(
        tokenizer, text, question, MAX_LEN, HEAD_MAX_LEN,
        return_stats=True, return_truncation_stats=True,
    )
    assert audit["state_tokens"] == stats["state_tokens"]
    assert not stats["truncated"] and len(markers) == head_stats["options_distinct"] == 3
    assert audit["passages"][0]["cut"] is False


def test_reject_four_action_mask():
    with pytest.raises(ValueError, match="mask"):
        serialize_vixstory_state(observation(action_mask=[True] * 4), tokenizer=TokenizerStub())
