"""Frozen P5 -> P6 interface: 4-action probabilities from a fine-tuned Laya checkpoint.

The RL stage must not know how Laya is loaded, which question id is used, or in which order
the decoder returns options. It asks this module for probabilities in the canonical
:data:`src.decision.schema.ACTION_ORDER` and nothing else:

    adapter = LayaAdapter.load(checkpoint_dir)          # cuda if available, else cpu
    probs = adapter.predict_probs(list_of_state_texts)  # float64 [N, 4], rows sum to 1
    adapter.model_hash()                                # weights + config digest
    adapter.close()                                     # unload before exiting

Two properties the RL stage relies on:

* **Order.** Rows follow ``states`` exactly (the library preserves input order; its optional
  length sorting is off) and columns follow ``ACTION_ORDER``. The decoder keys probabilities
  by action name, so the mapping here is by name, never by position.
* **Raw probabilities.** Masking is the environment's job, not this adapter's: an illegal
  action still gets its model probability. Applying the action mask is what makes the policy
  legal, and it must happen where the mask is known.

The library rounds its decoded probabilities to 4 decimals, so the raw values do not sum to
exactly 1; :meth:`predict_probs` renormalizes each row and fails loudly if a row is not a
probability distribution at all. :meth:`model_hash` must be computed after the checkpoint is
loaded, because loading repairs the tokenizer config in place.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.decision.schema import ACTION_ORDER, build_questions_field
from src.utils.hashing import canonical_json, sha256_file, sha256_text

#: Question id shared with the training JSONL and the environment contract.
QUESTION_ID = "next_action"

#: Files that identify a checkpoint: weights plus every config that changes its output.
IDENTITY_FILES = ("model.safetensors", "rl_agent_config.json",
                  "tokenizer/tokenizer_config.json", "encoder/config.json")

#: A decoded row must sum to this before renormalization. The decoder rounds each of the
#: four options to 4 decimals, so a real row is within 2e-4 of 1; anything outside this
#: window is not a decoded distribution at all.
MIN_PLAUSIBLE_SUM = 0.9
MAX_PLAUSIBLE_SUM = 1.1

#: Rounding to 4 decimals can put two options at the same value, where ``argmax`` picks the
#: first. Below this tolerance the decoder's own ``choice`` and the row's argmax may
#: legitimately disagree; above it they cannot.
CHOICE_TOLERANCE = 1e-3


def checkpoint_model_hash(checkpoint_dir: str | Path) -> str:
    """Digest of the checkpoint files that define behaviour (weights + configs).

    ``model.safetensors`` and ``rl_agent_config.json`` are required; the tokenizer and encoder
    configs are hashed when present. Callers must hash *after* loading, because loading
    rewrites ``tokenizer/tokenizer_config.json`` the first time it needs repairing.
    """
    root = Path(checkpoint_dir)
    for required in ("model.safetensors", "rl_agent_config.json"):
        if not (root / required).is_file():
            raise FileNotFoundError(f"{root}: not a Laya checkpoint (no {required})")
    present = {name: sha256_file(root / name) for name in IDENTITY_FILES if (root / name).is_file()}
    return sha256_text(canonical_json(present))


class LayaAdapter:
    """Load-back wrapper around a Laya checkpoint; the only Laya API the RL stage sees."""

    def __init__(self, agent: Any, *, checkpoint_dir: str | Path, model_hash: str,
                 batch_size: int = 8) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self._agent = agent
        self.checkpoint_dir = Path(checkpoint_dir)
        self._model_hash = model_hash
        self.batch_size = batch_size
        self.device = str(getattr(agent, "device", "unknown"))
        self._questions = build_questions_field()

    @classmethod
    def load(cls, checkpoint_dir: str | Path, *, device: str | None = None,
             batch_size: int = 8) -> "LayaAdapter":
        """Load a local checkpoint directory; CUDA is used when available."""
        path = Path(checkpoint_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"checkpoint directory not found: {path}")

        import torch

        import laya

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        agent = laya.load(str(path), device=device)
        # After load: the library repairs tokenizer_config.json in place on first use.
        return cls(agent, checkpoint_dir=path, model_hash=checkpoint_model_hash(path),
                   batch_size=batch_size)

    def model_hash(self) -> str:
        """Identifies the weights + config this adapter loaded; cache keys embed it."""
        return self._model_hash

    def predict_probs(self, states: Sequence[str]) -> np.ndarray:
        """``[N, 4]`` float64 probabilities in ``ACTION_ORDER``, one row per input state."""
        if self._agent is None:
            raise RuntimeError("this adapter was closed; load a checkpoint before predicting")
        texts = [str(state) for state in states]
        out = np.zeros((len(texts), len(ACTION_ORDER)), dtype=np.float64)
        if not texts:
            return out
        results = self._agent.predict_batch(texts, self._questions, batch_size=self.batch_size)
        if len(results) != len(texts):
            raise ValueError(f"predict_batch returned {len(results)} results for {len(texts)} states")
        for row, (text, result) in enumerate(zip(texts, results)):
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
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover - torch is a hard dependency in practice
            pass
