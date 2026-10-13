"""The V2 bridge: explicit 8192/192 serialization, by-name three-action rows, fail-loud decoding.

A stub agent and a stub tokenizer stand in for ``laya`` — no checkpoint, no torch model, no
GPU, no network. What is tested is the contract the V2 RL stage depends on: public
observations become ``evidence-retrieval-v2-len8192`` state text at an explicit 8192/192
budget, rows follow input order, columns follow the three-action ``ACTION_ORDER`` by *name*,
the checkpoint digest is taken after loading (the library normalizes the tokenizer config in
place), and anything that is not a decoded three-action distribution raises.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.decision.laya_adapter import checkpoint_model_hash
from src.decision.laya_adapter_v2 import CHOICE_TOLERANCE, QUESTION_ID, LayaAdapterV2
from src.decision.schema_v2 import ACTION_ORDER, build_questions_field
from src.decision.vixstory_state_text_v2 import HEAD_MAX_LEN, MAX_LEN, serialize_vixstory_state

#: Sums to exactly 1; keys deliberately in neither canonical nor alphabetical order.
PROBS = {"SEARCH_DENSE": 0.2, "STOP": 0.1, "SEARCH_BM25": 0.7}


class TokenizerStub:
    """Word-counting tokenizer: enough of the Laya tokenizer surface for the V2 serializer."""

    mask_token = "<mask>"
    mask_token_id = 3
    cls_token_id = 1
    sep_token_id = 2

    def __init__(self):
        self.vocab = {}

    def __call__(self, text, **kwargs):
        return {"input_ids": [self.vocab.setdefault(word, len(self.vocab) + 4)
                              for word in text.split()]}


def observation(n=1, **changes):
    obs = {"qid": "q1", "question": "Địa điểm nào?",
           "seen_doc_ids": [f"d{i}" for i in range(n)],
           "seen_doc_texts": [f"nguồn{i} " * 30 for i in range(n)],
           "history": [], "optional_used": 0, "remaining_optional_searches": 2,
           "action_mask": [True, True, True]}
    obs.update(changes)
    return obs


def _row(probabilities=None, choice="SEARCH_BM25"):
    return {"answers": {QUESTION_ID: {"type": "choice", "choice": choice,
                                      "probabilities": dict(probabilities or PROBS)}}}


class _StubAgent:
    def __init__(self, rows=None, *, device="cpu"):
        self.device = device
        self.rows = rows
        self.calls = []

    def predict_batch(self, texts, questions, batch_size=8, **kwargs):
        self.calls.append({"texts": list(texts), "questions": questions,
                           "batch_size": batch_size, **kwargs})
        if self.rows is not None:
            return self.rows
        return [_row() for _ in texts]


def _adapter(rows=None, *, batch_size=8):
    agent = _StubAgent(rows, device="cpu")
    return LayaAdapterV2(agent, TokenizerStub(), checkpoint_dir="/nonexistent",
                         model_hash="m" * 64, batch_size=batch_size), agent


def test_columns_follow_the_three_action_order_by_name_and_texts_use_the_v2_serializer():
    adapter, agent = _adapter()
    obs = [observation(), observation(n=3, history=["SEARCH_BM25"], optional_used=1,
                                      remaining_optional_searches=1)]
    probs = adapter.predict_probs(obs)
    assert probs.shape == (2, 3)
    assert probs.dtype == np.float64
    # PROBS is keyed by name in a shuffled order; the row must be per-name, three columns.
    assert probs[0].tolist() == pytest.approx([PROBS[name] for name in ACTION_ORDER])
    assert np.allclose(probs.sum(axis=1), 1.0)
    call = agent.calls[0]
    assert call["texts"] == [serialize_vixstory_state(o, tokenizer=TokenizerStub(),
                                                      max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN)
                             for o in obs]
    assert call["questions"] == build_questions_field()
    assert set(call["questions"][QUESTION_ID]["criteria"]) == set(ACTION_ORDER)
    assert call["batch_size"] == 8
    # The inference clamp is the same explicit budget the serializer verified against.
    assert (call["max_len"], call["head_max_len"]) == (MAX_LEN, HEAD_MAX_LEN) == (8192, 192)


def test_each_row_maps_its_own_probabilities_by_name():
    rows = [_row({"SEARCH_DENSE": 0.6, "SEARCH_BM25": 0.3, "STOP": 0.1}, choice="SEARCH_DENSE"),
            _row({"STOP": 0.5, "SEARCH_DENSE": 0.3, "SEARCH_BM25": 0.2}, choice="STOP")]
    adapter, _ = _adapter(rows)
    probs = adapter.predict_probs([observation(), observation()])
    assert probs[0].tolist() == pytest.approx([0.1, 0.3, 0.6])
    assert probs[1].tolist() == pytest.approx([0.5, 0.2, 0.3])


def test_rounded_sums_are_renormalized():
    rounded = {"STOP": 0.7, "SEARCH_BM25": 0.2, "SEARCH_DENSE": 0.0999}
    adapter, _ = _adapter([_row(rounded, choice="STOP")])
    probs = adapter.predict_probs([observation()])
    assert probs.sum() == pytest.approx(1.0, abs=1e-12)
    assert probs[0].tolist() == pytest.approx([0.7, 0.2, 0.0999], abs=1e-4)


def test_empty_observations_return_an_empty_matrix_without_calling_the_agent():
    adapter, agent = _adapter()
    probs = adapter.predict_probs([])
    assert probs.shape == (0, 3) and agent.calls == []


def test_a_row_count_mismatch_is_an_error():
    adapter, _ = _adapter([_row()])
    with pytest.raises(ValueError, match="1 results for 2 states"):
        adapter.predict_probs([observation(), observation()])


@pytest.mark.parametrize("row, match", [
    ({"answers": {}}, "missing choice answer"),
    ({"answers": {QUESTION_ID: {"type": "noul", "choice": "STOP", "probabilities": PROBS}}},
     "missing choice answer"),
    (_row({"STOP": 0.1, "SEARCH_BM25": 0.7, "SEARCH_DENSE": 0.2, "EXTRA": 0.0}), "keyed by"),
    (_row({"ANSWER": 0.1, "SEARCH_BM25": 0.7, "SEARCH_DENSE": 0.2}), "keyed by"),
    (_row({"STOP": 0.01, "SEARCH_BM25": 0.01, "SEARCH_DENSE": 0.01}), "not a distribution"),
    (_row({"STOP": float("nan"), "SEARCH_BM25": 0.5, "SEARCH_DENSE": 0.5}), "non-finite"),
    (_row({"STOP": -0.1, "SEARCH_BM25": 0.6, "SEARCH_DENSE": 0.5}), "non-finite or negative"),
    (_row({"STOP": 0.6, "SEARCH_BM25": 0.2, "SEARCH_DENSE": 0.2}, choice="SEARCH_BM25"),
     "not the most probable"),
    (_row(PROBS, choice="FOO"), "not one of"),
])
def test_broken_rows_raise(row, match):
    adapter, _ = _adapter([row])
    with pytest.raises(ValueError, match=match):
        adapter.predict_probs([observation()])


def test_a_choice_within_the_rounding_tolerance_is_accepted():
    half = CHOICE_TOLERANCE / 2
    near = {"STOP": 0.4 - half / 2, "SEARCH_BM25": 0.4 + half / 2, "SEARCH_DENSE": 0.2}
    adapter, _ = _adapter([_row(near, choice="STOP")])
    probs = adapter.predict_probs([observation()])
    assert probs[0].tolist() == pytest.approx([0.39975, 0.40025, 0.2], abs=1e-12)


def test_private_label_fields_in_an_observation_are_rejected_before_any_inference():
    adapter, agent = _adapter()
    with pytest.raises(ValueError, match="label-bearing"):
        adapter.predict_probs([observation(gold_context_ids=["d0"])])
    assert agent.calls == []


def test_a_non_mapping_observation_is_rejected():
    adapter, agent = _adapter()
    with pytest.raises(ValueError, match="must be a mapping"):
        adapter.predict_probs(["already serialized text"])
    assert agent.calls == []


def test_an_over_budget_public_state_fails_loudly_before_any_inference():
    adapter, agent = _adapter()
    with pytest.raises(ValueError, match="essential"):
        adapter.predict_probs([observation(question="dài " * 9000)])
    assert agent.calls == []


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError, match="batch_size"):
        LayaAdapterV2(_StubAgent(), TokenizerStub(), checkpoint_dir="/x", model_hash="m",
                      batch_size=0)


def test_close_is_idempotent_and_predicting_afterwards_fails():
    adapter, _ = _adapter()
    adapter.close()
    adapter.close()
    with pytest.raises(RuntimeError, match="closed"):
        adapter.predict_probs([observation()])


def test_load_rejects_a_missing_checkpoint_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint directory not found"):
        LayaAdapterV2.load(tmp_path / "absent")


def _write_checkpoint(root):
    (root / "tokenizer").mkdir(parents=True, exist_ok=True)
    (root / "encoder").mkdir(exist_ok=True)
    (root / "model.safetensors").write_bytes(b"weights")
    (root / "rl_agent_config.json").write_text('{"policy": "laya"}', encoding="utf-8")
    (root / "tokenizer" / "tokenizer_config.json").write_text('{"model_max_length": 8192}',
                                                              encoding="utf-8")
    (root / "encoder" / "config.json").write_text('{"hidden_size": 768}', encoding="utf-8")


def test_load_hashes_the_checkpoint_after_the_library_normalizes_the_tokenizer(tmp_path, monkeypatch):
    root = tmp_path / "ckpt"
    _write_checkpoint(root)
    before = checkpoint_model_hash(root)
    seen = {}

    def fake_load(path, device=None):
        # What laya.load does before reading the tokenizer: repair tokenizer_config.json in place.
        cfg_path = root / "tokenizer" / "tokenizer_config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        agent = _StubAgent()
        agent.tok = TokenizerStub()
        seen["path"], seen["device"] = path, device
        return agent

    import laya

    monkeypatch.setattr(laya, "load", fake_load)
    adapter = LayaAdapterV2.load(root, device="cpu")
    try:
        assert seen == {"path": str(root), "device": "cpu"}
        # The digest describes the checkpoint as loaded, not the pre-normalization files.
        assert adapter.model_hash() == checkpoint_model_hash(root)
        assert adapter.model_hash() != before
        # The loaded agent's tokenizer is the one the bridge serializes with.
        assert adapter.predict_probs([observation()]).shape == (1, 3)
    finally:
        adapter.close()
