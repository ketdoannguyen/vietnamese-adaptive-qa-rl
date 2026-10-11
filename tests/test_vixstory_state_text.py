"""The same token-bounded public state must reach Laya training and inference."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.decision.schema import build_questions_field
from src.decision.vixstory_state_text import STATE_TEXT_VERSION, serialize_vixstory_state
from src.pipeline.vixstory_arms import LayaPrior


class TokenizerStub:
    mask_token = "<mask>"
    mask_token_id = 3
    cls_token_id = 1
    sep_token_id = 2

    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text.split())))}



def _observation(**overrides):
    obs = {"question": "Địa điểm nào?", "seen_doc_ids": ("d1", "d2"),
           "seen_doc_texts": ("chi tiết " * 100, "nguồn thứ hai " * 100),
           "candidate": {"text": "Hà Nội", "cited_ids": ("d1",),
                         "citation_valid": True, "answer_present": True},
           "history": (1,), "optional_used": 1, "remaining_optional_searches": 1,
           "action_mask": [True, True, True, True]}
    obs.update(overrides)
    return obs


def test_essential_fields_survive_laya_right_truncation():
    from laya.common import build_sequence
    from laya.train import to_internal

    tokenizer = TokenizerStub()
    q = to_internal("next_action", build_questions_field()["next_action"])
    state = serialize_vixstory_state(_observation(), tokenizer=tokenizer,
                                      max_len=200, head_max_len=90)
    assert STATE_TEXT_VERSION == "v2"
    assert "Ứng viên trả lời: Hà Nội" in state
    assert "Trích dẫn: d1" in state
    assert "Số lượt tìm đã dùng: 1 | còn lại: 1" in state
    assert "Hành động hợp lệ: ANSWER, SEARCH_BM25, SEARCH_DENSE, ABSTAIN" in state
    _, _, stats = build_sequence(tokenizer, state, q, 200, 90,
                                 return_truncation_stats=True)
    assert stats["truncated"] is False
    assert "doc=d1" in state and "doc=d2" in state


def test_evidence_allocation_preserves_later_passages_when_space_is_small():
    tokenizer = TokenizerStub()
    state = serialize_vixstory_state(_observation(), tokenizer=tokenizer,
                                      max_len=145, head_max_len=90)
    assert "doc=d1" in state and "doc=d2" in state
    assert "nguồn" in state


def test_essential_fields_over_budget_raise_instead_of_silent_truncation():
    with pytest.raises(ValueError, match="essential"):
        serialize_vixstory_state(_observation(question="question " * 500),
                                  tokenizer=TokenizerStub(), max_len=200, head_max_len=90)


def test_serializer_rejects_label_bearing_observations():
    with pytest.raises(ValueError, match="forbidden"):
        serialize_vixstory_state(_observation(gold_answer="bí mật"), tokenizer=TokenizerStub())


def test_laya_prior_rejects_checkpoint_without_v2_manifest_before_loading(tmp_path, monkeypatch):
    import json
    from src.decision.laya_adapter import LayaAdapter

    checkpoint = tmp_path / "E04_ce"
    checkpoint.mkdir()
    (tmp_path / "E04_ce.manifest.json").write_text(
        json.dumps({"state_text_version": "v1"}), encoding="utf-8")
    monkeypatch.setattr(LayaAdapter, "load", lambda *a, **kw: pytest.fail("model loaded"))
    with pytest.raises(ValueError, match="state text v2"):
        LayaPrior(checkpoint, device="cpu")


def test_laya_prior_rejects_mutated_teacher_provenance_before_model_load(tmp_path, monkeypatch):
    import json
    from src.decision.laya_adapter import LayaAdapter
    from src.utils.hashing import sha256_file

    teacher = tmp_path / "train300.jsonl"
    teacher.write_text("{}\n")
    teacher.with_suffix(".provenance.json").write_text(json.dumps({"sha256": "wrong"}))
    checkpoint = tmp_path / "E04_ce"
    checkpoint.mkdir()
    (tmp_path / "E04_ce.manifest.json").write_text(json.dumps({
        "state_text_version": "v2", "data": {"path": str(teacher),
                                              "sha256": sha256_file(teacher)},
        "teacher_contract": {"sha256": sha256_file(teacher), "state_text_version": "v2"},
    }))
    monkeypatch.setattr(LayaAdapter, "load", lambda *a, **kw: pytest.fail("model loaded"))
    with pytest.raises(ValueError, match="provenance"):
        LayaPrior(checkpoint, device="cpu")


def test_laya_prior_closes_adapter_when_loaded_tokenizer_contract_fails(tmp_path, monkeypatch):
    import json
    from src.decision.laya_adapter import LayaAdapter
    from src.utils.hashing import sha256_file

    teacher = tmp_path / "train300.jsonl"
    teacher.write_text("{}\n")
    from src.decision import vixstory_state_text

    provenance = {"sha256": sha256_file(teacher), "state_text_version": "v2",
                  "state_text_sha256": sha256_file(vixstory_state_text.__file__),
                  "tokenizer_contract_sha256": "expected", "max_len": 768,
                  "head_max_len": 192}
    teacher.with_suffix(".provenance.json").write_text(json.dumps(provenance))
    checkpoint = tmp_path / "E04_ce"
    checkpoint.mkdir()
    (tmp_path / "E04_ce.manifest.json").write_text(json.dumps({
        "state_text_version": "v2", "data": {"path": str(teacher),
                                           "sha256": sha256_file(teacher)},
        "teacher_contract": provenance, "tokenizer_contract_sha256": "expected",
        "max_len": 768, "head_max_len": 192,
    }))
    closed = []

    class BrokenAdapter:
        _agent = type("Agent", (), {"tok": object(), "cfg": {"max_len": 768,
                                                      "head_max_len": 192}})()

        def close(self):
            closed.append(True)

    monkeypatch.setattr(LayaAdapter, "load", lambda *a, **kw: BrokenAdapter())
    with pytest.raises(AttributeError, match="backend_tokenizer"):
        LayaPrior(checkpoint, device="cpu")
    assert closed == [True]


def test_real_tokenizer_keeps_fields_and_all_evidence_ids_when_available():
    from transformers import AutoTokenizer
    from laya.common import build_sequence
    from laya.train import to_internal

    root = Path(__file__).resolve().parents[1]
    tok_dir = root / "outputs/experiments/vixstory-pilot-300-2026-10-11/laya/E04_ce/tokenizer"
    if not tok_dir.is_dir():
        pytest.skip("local Laya tokenizer not available")
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, local_files_only=True)
    obs = _observation()
    q = to_internal("next_action", build_questions_field()["next_action"])
    state = serialize_vixstory_state(obs, tokenizer=tokenizer)
    _, _, stats = build_sequence(tokenizer, state, q, 768, 192,
                                 return_truncation_stats=True)
    assert stats["truncated"] is False
    for required in (obs["question"], obs["candidate"]["text"], "doc=d1", "doc=d2",
                     "Số lượt tìm đã dùng: 1 | còn lại: 1", "Hành động hợp lệ:"):
        assert required in state
