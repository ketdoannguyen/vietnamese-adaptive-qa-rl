"""Frozen V2 Laya bridge: three evidence-retrieval actions from a local checkpoint.

V2 (plan ``.claude/plans/2026-10-12-vixstory-evidence-first-v2.md``) replaces the V1
four-action reader loop with three retrieval actions — ``STOP``, ``SEARCH_BM25``,
``SEARCH_DENSE`` — and the RL stage must not know how Laya is loaded, how a public
observation becomes state text, or in which order the decoder returns options. It asks this
module for probabilities in the canonical :data:`~src.decision.schema_v2.ACTION_ORDER` and
nothing else::

    adapter = LayaAdapterV2.load(checkpoint_dir)        # cuda if available, else cpu
    probs = adapter.predict_probs(public_observations)  # float64 [N, 3], rows sum to 1
    adapter.model_hash()                                # weights + config digest
    adapter.close()                                     # unload before exiting

Properties the V2 route relies on:

* **Three actions, by name.** Columns follow ``ACTION_ORDER``; the decoder keys its
  probabilities by action name and the mapping here is by name, never by position.
* **The bridge owns the serializer.** Inputs are public observations (the environment's
  ``PUBLIC_OBS_KEYS`` mapping) serialized with the explicit V2 state serializer at
  8192/192 (``evidence-retrieval-v2-len8192``); ``predict_batch`` is called with the same
  explicit ``max_len``/``head_max_len`` so inference clamps with the budget the serializer
  verified. Label-bearing fields never enter: the serializer rejects them structurally.
* **Raw probabilities.** Masking is the environment's job, not this adapter's: an illegal
  action still gets its model probability.
* **Hash after normalization.** ``laya.load`` repairs ``tokenizer/tokenizer_config.json``
  in place, so the digest is taken after loading, with the same
  :func:`~src.decision.laya_adapter.checkpoint_model_hash` the training path records. The
  frozen V1 adapter module is reused unmodified.

The decoder rounds each option to 4 decimals, so raw rows do not sum to exactly 1;
:meth:`predict_probs` renormalizes each row and fails loudly when a row is not a
distribution at all.
"""

from __future__ import annotations

import gc
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.decision.laya_adapter import checkpoint_model_hash
from src.decision.schema_v2 import ACTION_ORDER, N_ACTIONS, build_questions_field
from src.decision.vixstory_state_text_v2 import HEAD_MAX_LEN, MAX_LEN, serialize_vixstory_state

#: Question id shared with the V2 training records and the environment contract.
QUESTION_ID = "next_action"

#: A decoded row must sum to this before renormalization. The decoder rounds each of the
#: three options to 4 decimals, so a real row is within 2e-4 of 1; anything outside this
#: window is not a decoded distribution at all.
MIN_PLAUSIBLE_SUM = 0.9
MAX_PLAUSIBLE_SUM = 1.1

#: Rounding to 4 decimals can put two options at the same value, where ``argmax`` picks the
#: first. Below this tolerance the decoder's own ``choice`` and the row's argmax may
#: legitimately disagree; above it they cannot.
CHOICE_TOLERANCE = 1e-3


class LayaAdapterV2:
    """Load-back wrapper around a frozen V2 Laya checkpoint; the only Laya API the RL stage sees."""

    def __init__(self, agent: Any, tokenizer: Any, *, checkpoint_dir: str | Path,
                 model_hash: str, batch_size: int = 8) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self._agent = agent
        self._tokenizer = tokenizer
        self.checkpoint_dir = Path(checkpoint_dir)
        self._model_hash = model_hash
        self.batch_size = batch_size
        self._questions = build_questions_field()

    @classmethod
    def load(cls, checkpoint_dir: str | Path, *, device: str | None = None,
             batch_size: int = 8) -> "LayaAdapterV2":
        """Load a local checkpoint directory; CUDA is used when available."""
        path = Path(checkpoint_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"checkpoint directory not found: {path}")

        import laya
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        agent = laya.load(str(path), device=device)
        # After load: the library repairs tokenizer_config.json in place on first use, so a
        # digest taken earlier would not describe the checkpoint that was actually loaded.
        return cls(agent, agent.tok, checkpoint_dir=path,
                   model_hash=checkpoint_model_hash(path), batch_size=batch_size)

    def model_hash(self) -> str:
        """Identifies the weights + config this adapter loaded; cache keys embed it."""
        return self._model_hash

    def predict_probs(self, observations: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """``[N, 3]`` float64 probabilities in ``ACTION_ORDER``, one row per observation."""
        if self._agent is None:
            raise RuntimeError("this adapter was closed; load a checkpoint before predicting")
        texts = []
        for row, obs in enumerate(observations):
            if not isinstance(obs, Mapping):
                raise ValueError(f"observation {row} must be a mapping; got {type(obs).__name__}")
            texts.append(serialize_vixstory_state(obs, tokenizer=self._tokenizer,
                                                  max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN))
        out = np.zeros((len(texts), N_ACTIONS), dtype=np.float64)
        if not texts:
            return out
        results = self._agent.predict_batch(texts, self._questions, batch_size=self.batch_size,
                                            max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN)
        if len(results) != len(texts):
            raise ValueError(f"predict_batch returned {len(results)} results for {len(texts)} states")
        for row, result in enumerate(results):
            answers = result.get("answers") or {}
            answer = answers.get(QUESTION_ID)
            if not isinstance(answer, dict) or answer.get("type") != "choice":
                raise ValueError(f"state {row}: missing choice answer {QUESTION_ID!r} in {sorted(answers)}")
            probs = answer.get("probabilities")
            if not isinstance(probs, dict) or set(probs) != set(ACTION_ORDER):
                raise ValueError(
                    f"state {row}: probabilities must be keyed by {sorted(ACTION_ORDER)}, "
                    f"got {sorted(probs) if isinstance(probs, dict) else type(probs).__name__}")
            values = np.array([float(probs[action]) for action in ACTION_ORDER], dtype=np.float64)
            if not np.all(np.isfinite(values)) or values.min() < 0.0:
                raise ValueError(f"state {row}: non-finite or negative probabilities {values.tolist()}")
            total = float(values.sum())
            if not MIN_PLAUSIBLE_SUM <= total <= MAX_PLAUSIBLE_SUM:
                raise ValueError(f"state {row}: probabilities sum to {total!r}, not a distribution")
            values = values / total
            choice = answer.get("choice")
            if choice not in ACTION_ORDER:
                raise ValueError(f"state {row}: decoder choice {choice!r} is not one of "
                                 f"{list(ACTION_ORDER)}")
            # The decoder derives `choice` and `probabilities` from the same vector, so a
            # disagreement beyond the rounding tolerance means the response was garbled.
            if values[ACTION_ORDER.index(choice)] < values.max() - CHOICE_TOLERANCE:
                raise ValueError(
                    f"state {row}: decoder choice {choice!r} is not the most probable option "
                    f"({dict(zip(ACTION_ORDER, values.tolist()))})")
            out[row] = values
        return out

    def close(self) -> None:
        """Drop the model and release GPU memory (safe to call twice)."""
        if self._agent is None:
            return
        self._agent = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover - torch is a hard dependency in practice
            pass
