"""Offline contract for the planned GLiNER2 decision-prior adapter (``src/decision/gliner_adapter.py``).

The adapter is the only GLiNER surface the RL stage will see and mirrors
:class:`~src.decision.laya_adapter.LayaAdapter`::

    adapter = GLiNERAdapter(classifier, schema=schema, model_hash=digest)
    probs = adapter.predict_probs(state_texts)   # float64 [N, 4], rows sum to 1
    adapter.model_hash()                         # schema + base revision + adapter weights
    adapter.close()

Two properties the RL stage relies on:

* **Order.** Columns follow ``ACTION_ORDER`` by label *name*, never by position. The
  library returns labels in its own order and accepts a permuted schema; the row must
  not change.
* **Unmasked raw prior.** The four columns are the softmax of the four raw logits the
  scoring API exposes. No action mask is applied here, and the legality line inside the
  state text never changes the row; masking is the policy sampler's job.

No gliner2, no checkpoint, no network: ``_FakeClassifier`` mirrors the score-object
surface the P0 gate verified (``scores.tasks`` carries label names, ``scores.logit(task,
label)`` the raw score, ``batch_score`` returns one object per input text). Its
``probability()`` deliberately returns the sigmoid, so an adapter that delegates to the
library activation instead of softmaxing the raw logits fails these tests. The identity
function is exercised on tiny fake directories; ``load`` is only asked to reject a
missing base dir, which must happen before any gliner2 import.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.decision.gliner_adapter import GLiNERAdapter, QUESTION_ID, checkpoint_model_hash
from src.decision.schema import ACTION_ORDER, build_questions_field
from src.utils.hashing import canonical_json

ROOT = Path(__file__).resolve().parents[1]

#: Stands in for the ``ClassificationSchema`` the real loader builds; the adapter must pass
#: it through to the classifier untouched, so the tests only need an identifiable object.
SCHEMA = "schema-sentinel"
HASH = "g" * 64

#: Canonical schema identity, derived from the shared action schema, not a hand-typed dict.
SCHEMA_JSON = canonical_json(build_questions_field())

#: One finite logit per action; distinctive values so column swaps cannot hide.
DEFAULT_LOGITS = {"ANSWER": 2.0, "SEARCH_BM25": 1.0, "SEARCH_DENSE": 0.5, "ABSTAIN": -1.0}


def _softmax(values):
    maximum = max(values)
    exps = [math.exp(value - maximum) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


class _FakeScores:
    """Mirrors the gliner2 ``ClassificationScores`` surface seen by the P0 gate."""

    def __init__(self, logits, *, order=None):
        self._logits = dict(logits)
        self.tasks = {QUESTION_ID: list(order) if order is not None else list(self._logits)}

    def logit(self, task, label):
        if task != QUESTION_ID:
            raise KeyError(task)
        return self._logits[label]

    def probability(self, task, label):
        # Deliberately the sigmoid (non-exclusive activation), not the softmax: an adapter
        # that delegates to the library probability instead of softmaxing the raw logits
        # must fail these tests.
        value = self._logits[label]
        return 1.0 / (1.0 + math.exp(-value))

    def top(self, task, n=1):
        ranked = sorted(((label, self.probability(task, label))
                         for label in self.tasks[task]), key=lambda item: -item[1])
        return ranked[:n]


class _FakeClassifier:
    """``batch_score``-compatible stand-in; records calls for order/parity checks."""

    def __init__(self, *, logits=None, default=None, order=None, rows=None):
        self.logits = dict(logits or {})  # text -> {label: logit}
        self.default = dict(default if default is not None else DEFAULT_LOGITS)
        self.order = order
        self.rows = rows  # explicit response override, for broken-batch cases
        self.calls = []  # (texts, schema) per batch_score call

    def _scores(self, text):
        return _FakeScores(self.logits.get(text, self.default), order=self.order)

    def score(self, text, schema, **kwargs):
        return self._scores(text)

    def batch_score(self, texts, schema, **kwargs):
        self.calls.append((list(texts), schema))
        if self.rows is not None:
            return list(self.rows)
        return [self._scores(text) for text in texts]


def _adapter(classifier=None, *, batch_size=8, model_hash=HASH, schema=SCHEMA):
    classifier = classifier if classifier is not None else _FakeClassifier()
    adapter = GLiNERAdapter(classifier, schema=schema, model_hash=model_hash,
                            batch_size=batch_size)
    return adapter, classifier


# --------------------------------------------------- mapping order and softmax


def test_columns_follow_the_canonical_order_by_name():
    shuffled = ("ABSTAIN", "SEARCH_DENSE", "SEARCH_BM25", "ANSWER")
    logits = {"ANSWER": 2.0, "SEARCH_BM25": 1.0, "SEARCH_DENSE": 0.5, "ABSTAIN": -1.0}
    classifier = _FakeClassifier(logits={"state a": logits, "state b": logits}, order=shuffled)
    adapter, _ = _adapter(classifier)

    probs = adapter.predict_probs(["state a", "state b"])

    assert probs.shape == (2, 4) and probs.dtype == np.float64
    expected = _softmax([logits[action] for action in ACTION_ORDER])
    assert probs[0].tolist() == pytest.approx(expected)
    assert np.allclose(probs.sum(axis=1), 1.0)
    # Inputs go in unchanged and in order, with the injected schema passed through.
    assert classifier.calls == [(["state a", "state b"], SCHEMA)]


def test_four_finite_logits_become_the_softmax_of_the_raw_logits():
    logits = {"ANSWER": 2.5, "SEARCH_BM25": 1.0, "SEARCH_DENSE": 0.5, "ABSTAIN": -1.5}
    adapter, _ = _adapter(_FakeClassifier(default=logits))

    probs = adapter.predict_probs(["state"])

    expected = _softmax([logits[action] for action in ACTION_ORDER])
    assert probs[0].tolist() == pytest.approx(expected, abs=1e-12)
    assert probs[0].sum() == pytest.approx(1.0, abs=1e-12)
    assert probs[0].argmax() == ACTION_ORDER.index("ANSWER")


def test_extreme_but_finite_logits_stay_finite():
    logits = {"ANSWER": 1000.0, "SEARCH_BM25": 1000.0, "SEARCH_DENSE": -1000.0,
              "ABSTAIN": -1000.0}
    adapter, _ = _adapter(_FakeClassifier(default=logits))

    probs = adapter.predict_probs(["state"])

    assert np.all(np.isfinite(probs)) and probs.sum() == pytest.approx(1.0)


def test_a_permuted_label_order_yields_identical_probabilities():
    logits = {"ANSWER": 2.0, "SEARCH_BM25": 1.0, "SEARCH_DENSE": 0.5, "ABSTAIN": -1.0}
    canonical, _ = _adapter(_FakeClassifier(default=logits, order=ACTION_ORDER))
    permuted, _ = _adapter(_FakeClassifier(default=logits,
                                           order=tuple(reversed(ACTION_ORDER))))

    first = canonical.predict_probs(["state"])
    second = permuted.predict_probs(["state"])

    np.testing.assert_allclose(first, second, rtol=0.0, atol=1e-12)


# --------------------------------------------------- batch/single parity and emptiness


def test_batch_and_single_calls_agree_row_for_row():
    rows = {
        "a": {"ANSWER": 3.0, "SEARCH_BM25": 0.0, "SEARCH_DENSE": -1.0, "ABSTAIN": -2.0},
        "b": {"ANSWER": -2.0, "SEARCH_BM25": 3.0, "SEARCH_DENSE": 0.0, "ABSTAIN": -1.0},
        "c": {"ANSWER": -2.0, "SEARCH_BM25": -1.0, "SEARCH_DENSE": 0.0, "ABSTAIN": 3.0},
    }
    adapter, _ = _adapter(_FakeClassifier(logits=rows))

    batch = adapter.predict_probs(["a", "b", "c"])
    single = np.vstack([adapter.predict_probs([text]) for text in ("a", "b", "c")])

    np.testing.assert_allclose(batch, single, rtol=0.0, atol=1e-12)
    for index, text in enumerate(("a", "b", "c")):
        expected = _softmax([rows[text][action] for action in ACTION_ORDER])
        assert batch[index].tolist() == pytest.approx(expected)


def test_empty_states_return_an_empty_matrix_without_calling_the_classifier():
    adapter, classifier = _adapter()

    probs = adapter.predict_probs([])

    assert probs.shape == (0, 4) and probs.dtype == np.float64
    assert classifier.calls == []


# --------------------------------------------------- fail loud on broken rows


def test_a_row_count_mismatch_is_an_error():
    classifier = _FakeClassifier(rows=[_FakeScores(DEFAULT_LOGITS)])
    adapter, _ = _adapter(classifier)

    with pytest.raises(ValueError, match=r"1 results for 2 states"):
        adapter.predict_probs(["a", "b"])


def test_a_missing_label_is_an_error():
    order = [action for action in ACTION_ORDER if action != "ABSTAIN"]
    classifier = _FakeClassifier(rows=[_FakeScores(DEFAULT_LOGITS, order=order)])
    adapter, _ = _adapter(classifier)

    with pytest.raises(ValueError, match="missing"):
        adapter.predict_probs(["state"])


def test_a_duplicate_label_is_an_error():
    order = ["ANSWER", "ANSWER", "SEARCH_BM25", "SEARCH_DENSE", "ABSTAIN"]
    classifier = _FakeClassifier(rows=[_FakeScores(DEFAULT_LOGITS, order=order)])
    adapter, _ = _adapter(classifier)

    with pytest.raises(ValueError, match="duplicate"):
        adapter.predict_probs(["state"])


def test_an_unknown_label_is_an_error():
    logits = dict(DEFAULT_LOGITS, FOO=0.0)
    order = ["ANSWER", "SEARCH_BM25", "SEARCH_DENSE", "FOO"]
    classifier = _FakeClassifier(rows=[_FakeScores(logits, order=order)])
    adapter, _ = _adapter(classifier)

    with pytest.raises(ValueError, match=r"unknown|missing"):
        adapter.predict_probs(["state"])


def test_a_missing_next_action_task_is_an_error():
    scores = _FakeScores(DEFAULT_LOGITS)
    scores.tasks = {"other_task": list(ACTION_ORDER)}
    classifier = _FakeClassifier(rows=[scores])
    adapter, _ = _adapter(classifier)

    with pytest.raises(ValueError, match="next_action"):
        adapter.predict_probs(["state"])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_logit_is_an_error(bad):
    logits = dict(DEFAULT_LOGITS, ANSWER=bad)
    adapter, _ = _adapter(_FakeClassifier(default=logits))

    with pytest.raises(ValueError, match="non-finite"):
        adapter.predict_probs(["state"])


# --------------------------------------------------- unmasked prior contract


def test_the_prior_is_unmasked_whatever_the_state_text_says():
    logits = {"ANSWER": 3.0, "SEARCH_BM25": 1.0, "SEARCH_DENSE": 0.5, "ABSTAIN": -2.0}
    adapter, _ = _adapter(_FakeClassifier(default=logits))
    only_abstain_legal = "Hành động hợp lệ: ABSTAIN"
    all_legal = "Hành động hợp lệ: ANSWER, SEARCH_BM25, SEARCH_DENSE, ABSTAIN"

    restricted = adapter.predict_probs([only_abstain_legal])
    unrestricted = adapter.predict_probs([all_legal])

    # The legality line never reaches the prior: both rows are the same raw softmax, and
    # the ANSWER column stays the argmax even where ANSWER is illegal. Masking is the
    # policy sampler's job.
    np.testing.assert_allclose(restricted, unrestricted, rtol=0.0, atol=1e-12)
    assert restricted[0].tolist() == pytest.approx(
        _softmax([logits[action] for action in ACTION_ORDER]))
    assert restricted[0, ACTION_ORDER.index("ANSWER")] == restricted[0].max()


# --------------------------------------------------- lifecycle


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError, match="batch_size"):
        GLiNERAdapter(_FakeClassifier(), schema=SCHEMA, model_hash=HASH, batch_size=0)


def test_close_is_idempotent_and_predicting_afterwards_fails():
    adapter, _ = _adapter()

    adapter.close()
    adapter.close()

    with pytest.raises(RuntimeError, match="closed"):
        adapter.predict_probs(["state"])


def test_model_hash_returns_the_identity_bound_at_construction():
    adapter, _ = _adapter(model_hash="f" * 64)
    assert adapter.model_hash() == "f" * 64


def test_load_rejects_a_missing_base_dir(tmp_path):
    # In this environment gliner2 is not installed, so reaching FileNotFoundError also
    # proves the dir check runs before any gliner2 import.
    with pytest.raises(FileNotFoundError, match=r"absent|not found"):
        GLiNERAdapter.load(tmp_path / "absent")


def test_gliner2_is_not_imported_for_module_load_or_a_missing_dir_load():
    """Fresh process: the module imports lazily and a bad path fails before the heavy import."""
    script = (
        "import sys\n"
        "from src.decision.gliner_adapter import GLiNERAdapter\n"
        "assert 'gliner2' not in sys.modules\n"
        "try:\n"
        "    GLiNERAdapter.load('/nonexistent/definitely-absent')\n"
        "except FileNotFoundError:\n"
        "    pass\n"
        "else:\n"
        "    raise SystemExit('load did not fail on a missing base dir')\n"
        "print('GLINER2:' + str('gliner2' in sys.modules))\n"
    )
    proc = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("GLINER2:False")


# --------------------------------------------------- model identity


def _write_base(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.safetensors").write_bytes(b"base weights")
    (root / "config.json").write_text('{"architecture": "boundary"}', encoding="utf-8")
    (root / "tokenizer_config.json").write_text('{"model_max_length": 512}', encoding="utf-8")


def _write_adapter(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "adapter_model.safetensors").write_bytes(b"lora weights")
    (root / "adapter_config.json").write_text('{"r": 8, "lora_alpha": 16}', encoding="utf-8")


def test_model_hash_is_stable_and_hex(tmp_path):
    base = tmp_path / "base"
    _write_base(base)

    digest = checkpoint_model_hash(base, revision="rev-a", schema_json=SCHEMA_JSON)

    assert digest == checkpoint_model_hash(base, revision="rev-a", schema_json=SCHEMA_JSON)
    assert (checkpoint_model_hash(base, revision="rev-a")
            == checkpoint_model_hash(base, revision="rev-a"))
    assert len(digest) == 64 and all(ch in "0123456789abcdef" for ch in digest)


def test_model_hash_changes_when_the_base_revision_changes(tmp_path):
    base = tmp_path / "base"
    _write_base(base)

    assert (checkpoint_model_hash(base, revision="rev-a", schema_json=SCHEMA_JSON)
            != checkpoint_model_hash(base, revision="rev-b", schema_json=SCHEMA_JSON))


def test_model_hash_changes_when_the_schema_changes(tmp_path):
    base = tmp_path / "base"
    _write_base(base)
    other = canonical_json(build_questions_field(instruction_text="Chọn hành động khác."))

    assert (checkpoint_model_hash(base, revision="rev-a", schema_json=SCHEMA_JSON)
            != checkpoint_model_hash(base, revision="rev-a", schema_json=other))


def test_model_hash_changes_when_the_base_weights_change(tmp_path):
    base = tmp_path / "base"
    _write_base(base)
    digest = checkpoint_model_hash(base, revision="rev-a", schema_json=SCHEMA_JSON)

    (base / "model.safetensors").write_bytes(b"other weights")

    assert checkpoint_model_hash(base, revision="rev-a", schema_json=SCHEMA_JSON) != digest


def test_model_hash_changes_when_the_adapter_weights_change(tmp_path):
    base, adapter_dir = tmp_path / "base", tmp_path / "adapter"
    _write_base(base)
    _write_adapter(adapter_dir)
    bound = dict(base_dir=base, adapter_dir=adapter_dir, revision="rev-a",
                 schema_json=SCHEMA_JSON)
    digest = checkpoint_model_hash(**bound)
    # The adapter is part of the identity: an adapter-less base is a different prior.
    assert digest != checkpoint_model_hash(base_dir=base, revision="rev-a",
                                           schema_json=SCHEMA_JSON)

    (adapter_dir / "adapter_model.safetensors").write_bytes(b"other lora weights")
    assert checkpoint_model_hash(**bound) != digest
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"lora weights")
    assert checkpoint_model_hash(**bound) == digest  # same files, same identity

    (adapter_dir / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
    assert checkpoint_model_hash(**bound) != digest


def test_model_hash_requires_the_base_and_adapter_identity_files(tmp_path):
    base, adapter_dir = tmp_path / "base", tmp_path / "adapter"
    _write_base(base)
    _write_adapter(adapter_dir)

    (base / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match=r"model\.safetensors"):
        checkpoint_model_hash(base, revision="rev-a", schema_json=SCHEMA_JSON)

    _write_base(base)
    (base / "config.json").unlink()
    with pytest.raises(FileNotFoundError, match=r"config\.json"):
        checkpoint_model_hash(base, revision="rev-a", schema_json=SCHEMA_JSON)

    _write_base(base)
    (adapter_dir / "adapter_model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match=r"adapter_model\.safetensors"):
        checkpoint_model_hash(base, adapter_dir=adapter_dir, revision="rev-a",
                              schema_json=SCHEMA_JSON)
