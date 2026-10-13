"""The frozen P5->P6 adapter: canonical column order, renormalization, fail-loud rows.

A stub agent stands in for ``laya`` — no checkpoint, no torch model, no GPU. What is
tested is the contract the RL stage depends on: rows follow input order, columns follow
``ACTION_ORDER`` by *name*, and anything that is not a decoded distribution raises.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.decision.laya_adapter import CHOICE_TOLERANCE, QUESTION_ID, LayaAdapter, checkpoint_model_hash
from src.decision.schema import ACTION_ORDER, build_questions_field

#: Sums to exactly 1; keys deliberately in neither canonical nor alphabetical order.
PROBS = {"ABSTAIN": 0.1, "SEARCH_DENSE": 0.2, "SEARCH_BM25": 0.3, "ANSWER": 0.4}


def _row(probabilities=None, choice="ANSWER"):
    return {"answers": {QUESTION_ID: {"type": "choice", "choice": choice,
                                      "probabilities": dict(probabilities or PROBS)}}}


class _StubAgent:
    def __init__(self, rows=None, *, device="cpu"):
        self.device = device
        self.rows = rows
        self.calls = []

    def predict_batch(self, texts, questions, batch_size=8):
        self.calls.append({"texts": list(texts), "questions": questions, "batch_size": batch_size})
        if self.rows is not None:
            return self.rows
        return [_row() for _ in texts]


def _adapter(rows=None, *, checkpoint_dir="/nonexistent", batch_size=8):
    agent = _StubAgent(rows, device="cpu")
    return LayaAdapter(agent, checkpoint_dir=checkpoint_dir, model_hash="m" * 64,
                       batch_size=batch_size), agent


def test_columns_follow_the_canonical_order_by_name():
    adapter, agent = _adapter()
    probs = adapter.predict_probs(["state a", "state b"])
    assert probs.shape == (2, 4)
    assert probs.dtype == np.float64
    # PROBS is keyed by name and inserted in a shuffled order; the row must be per-name.
    assert probs[0].tolist() == pytest.approx([PROBS[name] for name in ACTION_ORDER])
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert agent.calls[0]["texts"] == ["state a", "state b"]
    assert agent.calls[0]["questions"] == build_questions_field()
    assert agent.calls[0]["batch_size"] == 8


def test_rounded_sums_are_renormalized():
    rounded = {"ANSWER": 0.4, "SEARCH_BM25": 0.3, "SEARCH_DENSE": 0.2, "ABSTAIN": 0.0999}
    adapter, _ = _adapter([_row(rounded)])
    probs = adapter.predict_probs(["state"])
    assert probs.sum() == pytest.approx(1.0, abs=1e-12)
    assert probs[0].tolist() == pytest.approx([0.4, 0.3, 0.2, 0.0999], abs=1e-4)


def test_empty_states_return_an_empty_matrix_without_calling_the_agent():
    adapter, agent = _adapter()
    probs = adapter.predict_probs([])
    assert probs.shape == (0, 4) and agent.calls == []


def test_a_row_count_mismatch_is_an_error():
    adapter, _ = _adapter([_row()])
    with pytest.raises(ValueError, match="1 results for 2 states"):
        adapter.predict_probs(["a", "b"])


@pytest.mark.parametrize("row, match", [
    ({"answers": {}}, "missing choice answer"),
    ({"answers": {QUESTION_ID: {"type": "noul", "choice": "ANSWER",
                                "probabilities": PROBS}}}, "missing choice answer"),
    (_row({"ABSTAIN": 0.1, "SEARCH_DENSE": 0.2, "SEARCH_BM25": 0.3, "ANSWER": 0.3,
           "EXTRA": 0.1}), "keyed by"),
    (_row({"ANSWER": 0.01, "SEARCH_BM25": 0.01, "SEARCH_DENSE": 0.01, "ABSTAIN": 0.01}),
     "not a distribution"),
    (_row({"ANSWER": float("nan"), "SEARCH_BM25": 0.3, "SEARCH_DENSE": 0.3, "ABSTAIN": 0.4}),
     "non-finite"),
    (_row({"ANSWER": -0.1, "SEARCH_BM25": 0.5, "SEARCH_DENSE": 0.3, "ABSTAIN": 0.3}),
     "non-finite or negative"),
    (_row({"ANSWER": 0.3, "SEARCH_BM25": 0.3, "SEARCH_DENSE": 0.2, "ABSTAIN": 0.2},
          choice="ABSTAIN"), "not the most probable"),
    (_row(PROBS, choice="SEARCH_BM25"), "not the most probable"),  # legal name, not argmax
    (_row(PROBS, choice="FOO"), "not one of"),
])
def test_broken_rows_raise(row, match):
    adapter, _ = _adapter([row])
    with pytest.raises(ValueError, match=match):
        adapter.predict_probs(["state"])


def test_a_choice_within_the_rounding_tolerance_is_accepted():
    half = CHOICE_TOLERANCE / 2
    near = {"ANSWER": 0.3 - half / 2, "SEARCH_BM25": 0.3 + half / 2,
            "SEARCH_DENSE": 0.2, "ABSTAIN": 0.2}
    adapter, _ = _adapter([_row(near, choice="ANSWER")])
    probs = adapter.predict_probs(["state"])
    assert probs[0].tolist() == pytest.approx([0.29975, 0.30025, 0.2, 0.2], abs=1e-12)


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError, match="batch_size"):
        LayaAdapter(_StubAgent(), checkpoint_dir="/x", model_hash="m", batch_size=0)


def test_close_is_idempotent_and_predicting_afterwards_fails():
    adapter, _ = _adapter()
    adapter.close()
    adapter.close()
    with pytest.raises(RuntimeError, match="closed"):
        adapter.predict_probs(["state"])


def test_load_rejects_a_missing_checkpoint_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint directory not found"):
        LayaAdapter.load(tmp_path / "absent")


def _write_checkpoint(root):
    (root / "tokenizer").mkdir(parents=True, exist_ok=True)
    (root / "encoder").mkdir(exist_ok=True)
    (root / "model.safetensors").write_bytes(b"weights")
    (root / "rl_agent_config.json").write_text('{"policy": "laya"}', encoding="utf-8")
    (root / "tokenizer" / "tokenizer_config.json").write_text('{"model_max_length": 1024}',
                                                              encoding="utf-8")
    (root / "encoder" / "config.json").write_text('{"hidden_size": 768}', encoding="utf-8")


def test_model_hash_is_stable_and_sensitive_to_every_identity_file(tmp_path):
    root = tmp_path / "ckpt"
    _write_checkpoint(root)
    digest = checkpoint_model_hash(root)
    assert digest == checkpoint_model_hash(root) and len(digest) == 64

    (root / "model.safetensors").write_bytes(b"other weights")
    assert checkpoint_model_hash(root) != digest
    (root / "model.safetensors").write_bytes(b"weights")
    assert checkpoint_model_hash(root) == digest  # same files, same hash

    (root / "encoder" / "config.json").unlink()  # present optional files are part of identity
    without_encoder = checkpoint_model_hash(root)
    assert without_encoder != digest
    (root / "encoder" / "config.json").write_text('{"hidden_size": 1024}', encoding="utf-8")
    assert checkpoint_model_hash(root) != without_encoder

    # Required files missing -> the checkpoint is not loadable, never a hash of what is left.
    (root / "rl_agent_config.json").unlink()
    with pytest.raises(FileNotFoundError, match="not a Laya checkpoint"):
        checkpoint_model_hash(root)
    (root / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="not a Laya checkpoint"):
        checkpoint_model_hash(root)
