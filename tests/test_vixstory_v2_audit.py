"""Retention summaries must expose truncation performed before Laya's clamp."""

import pytest

from scripts.vixstory_v2_audit import audit_split, summarize_retention, write_new_json
from tests.test_vixstory_state_text_v2 import TokenizerStub, observation


def test_retention_buckets_do_not_hide_removed_evidence():
    result = summarize_retention([
        {"n_seen": 5, "passages": [
            {"retention": 0.1, "retained_tokens": 10, "cut": True},
            {"retention": 0.3, "retained_tokens": 30, "cut": True},
        ]},
        {"n_seen": 1, "passages": [
            {"retention": 1.0, "retained_tokens": 100, "cut": False},
        ]},
    ])
    assert result["5"]["retention_p50"] == pytest.approx(0.2)
    assert result["5"]["retained_tokens_p50"] == 20
    assert result["5"]["cut_rate"] == 1
    assert result["1"]["cut_rate"] == 0
    assert result["2"]["passages"] == 0
    assert "no states observed" in result["2"]["note"]


def test_gold_annotations_are_added_after_public_rendering():
    from types import SimpleNamespace

    obs = observation()
    world = SimpleNamespace(qid=obs["qid"], question=obs["question"],
                            gold_context_ids=("d4",),
                            corpus=dict(zip(obs["seen_doc_ids"], obs["seen_doc_texts"])))

    class Teacher:
        def __init__(self, world):
            self.world = world

        def enumerate_states(self):
            return [obs]

    summary, rows, samples = audit_split([world], TokenizerStub(), Teacher, split="train")
    assert summary["n_states"] == 1
    assert summary["gold_passage_present_occurrences"] == 1
    assert summary["gold_passage_cut_occurrences"] == 0
    assert rows[0]["passages"][4]["is_gold"] is True
    assert rows[0]["passages"][4]["cut"] is False
    assert not samples
    assert "gold_context_ids" not in obs


def test_gold_cut_samples_are_private_and_preserve_full_passages(monkeypatch):
    from types import SimpleNamespace

    from scripts import vixstory_v2_audit
    from src.decision.vixstory_state_text_v2 import render_with_retention

    obs = observation()
    world = SimpleNamespace(qid=obs["qid"], question=obs["question"],
                            gold_context_ids=("d4",),
                            corpus=dict(zip(obs["seen_doc_ids"], obs["seen_doc_texts"])))

    class Teacher:
        def __init__(self, world):
            self.world = world

        def enumerate_states(self):
            return [obs]

    def constrained_render(public_obs, *, tokenizer):
        return render_with_retention(public_obs, tokenizer=tokenizer, max_len=220, head_max_len=80)

    monkeypatch.setattr(vixstory_v2_audit, "render_with_retention", constrained_render)
    summary, rows, samples = audit_split([world], TokenizerStub(), Teacher, split="train")
    assert summary["gold_passage_cut_occurrences"] == 1
    assert rows[0]["passages"][4]["is_gold"] is True
    assert samples[0]["doc_id"] == "d4"
    assert samples[0]["full_passage"] == world.corpus["d4"]
    assert "gold_context_ids" not in obs


def test_private_audit_output_permissions_and_no_overwrite(tmp_path):
    path = tmp_path / "private.json"
    write_new_json(path, {"private": True}, private=True)
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        write_new_json(path, {})
