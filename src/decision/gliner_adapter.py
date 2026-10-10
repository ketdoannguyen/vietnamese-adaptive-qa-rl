"""Frozen P5 -> P6 interface: 4-action probabilities from a fine-tuned GLiNER2 checkpoint.

Mirrors :class:`~src.decision.laya_adapter.LayaAdapter` so the RL stage sees one prior
surface regardless of backend::

    adapter = GLiNERAdapter.load(base_dir, adapter_dir=..., revision=...)  # cuda if available
    probs = adapter.predict_probs(list_of_state_texts)  # float64 [N, 4], rows sum to 1
    adapter.model_hash()  # base + adapter + schema digest
    adapter.close()  # unload before exiting

Two properties the RL stage relies on:

* **Order.** Columns follow ``ACTION_ORDER`` by label *name*: the scoring API exposes
  ``scores.tasks[QUESTION_ID]`` as a name-keyed mapping and tolerates a permuted schema, so
  each canonical action is looked up by name, never by position.
* **Unmasked raw prior.** The four columns are the softmax of the four raw logits the
  scoring API exposes (``scores.logit``), not the library's activation: no action mask is
  applied here and the legality line inside the state text never reaches the prior.
  Masking is the policy sampler's job.

``gliner2``, ``peft`` and ``torch`` are imported inside :meth:`GLiNERAdapter.load` only, so
importing this module (and the Laya path) does not require the GLiNER environment. The
surface coded against here was verified by the P0 gate on the pinned checkpoint
(``ClassificationSchema.single`` -> softmax over exactly the four labels,
``Classifier.batch_score``, ``PeftModel.from_pretrained`` reload): see
``outputs/experiments/2026-10-09-gliner-p0/audit/gate_summary.md``.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.decision.schema import ACTION_CRITERIA, ACTION_ORDER, INSTRUCTIONS, build_questions_field
from src.utils.hashing import canonical_json, sha256_file, sha256_text

#: Question id shared with the training JSONL and the environment contract.
QUESTION_ID = "next_action"

#: Canonical schema identity. The adapter's schema is always derived from the shared action
#: schema, so this string is what :func:`checkpoint_model_hash` records by default.
SCHEMA_JSON = canonical_json(build_questions_field())

#: Base-checkpoint files that define behaviour. The weights and the top-level config must
#: exist; the encoder config, tokenizer and tokenizer config are hashed when present.
BASE_IDENTITY_FILES = ("model.safetensors", "config.json", "encoder_config/config.json",
                       "tokenizer.json", "tokenizer_config.json")

#: LoRA adapter files. The weights must exist; the PEFT config is hashed when present.
ADAPTER_IDENTITY_FILES = ("adapter_model.safetensors", "adapter_config.json")


def checkpoint_model_hash(base_dir: str | Path, *, adapter_dir: str | Path | None = None,
                          revision: str | None = None,
                          schema_json: str = SCHEMA_JSON) -> str:
    """Digest of everything that defines the prior: base files, adapter, revision, schema.

    The base weights/config are required, as are the adapter weights when ``adapter_dir``
    is given; the remaining identity files are hashed when present, so a snapshot missing
    an optional file still hashes deterministically. ``revision`` is the base snapshot's
    HF revision and is part of the identity: the same files re-downloaded at a different
    revision are a different prior.
    """
    base = Path(base_dir)
    for required in ("model.safetensors", "config.json"):
        if not (base / required).is_file():
            raise FileNotFoundError(f"{base}: not a GLiNER base model directory (no {required})")
    payload: dict[str, Any] = {
        "revision": revision,
        "schema": schema_json,
        "base": {name: sha256_file(base / name) for name in BASE_IDENTITY_FILES
                 if (base / name).is_file()},
        "adapter": None,
    }
    if adapter_dir is not None:
        adapter = Path(adapter_dir)
        if not (adapter / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(
                f"{adapter}: not a LoRA adapter directory (no adapter_model.safetensors)")
        payload["adapter"] = {name: sha256_file(adapter / name)
                              for name in ADAPTER_IDENTITY_FILES
                              if (adapter / name).is_file()}
    return sha256_text(canonical_json(payload))


def _row_probs(result: Any, row: int) -> np.ndarray:
    """Softmax of one result's four raw logits, mapped by label name.

    Fails loudly on anything but exactly the four canonical labels: a silently reordered
    or truncated score vector would corrupt the RL prior without any visible symptom.
    """
    labels = result.tasks.get(QUESTION_ID)
    if labels is None:
        raise ValueError(f"state {row}: scores have no {QUESTION_ID!r} task "
                         f"(got {sorted(result.tasks)})")
    names = list(labels)
    if len(set(names)) != len(names):
        raise ValueError(f"state {row}: duplicate label(s) for {QUESTION_ID!r}: {names}")
    unknown = set(names) - set(ACTION_ORDER)
    if unknown:
        raise ValueError(f"state {row}: unknown label(s) {sorted(unknown)} "
                         f"for {QUESTION_ID!r}")
    missing = set(ACTION_ORDER) - set(names)
    if missing:
        raise ValueError(f"state {row}: missing label(s) {sorted(missing)} "
                         f"for {QUESTION_ID!r}")
    values = np.array([float(result.logit(QUESTION_ID, action)) for action in ACTION_ORDER],
                      dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"state {row}: non-finite logits "
                         f"{dict(zip(ACTION_ORDER, values.tolist()))}")
    # Subtracting the maximum keeps exp() in (0, 1], so the softmax is finite for any
    # finite input and the denominator is at least 1.
    exps = np.exp(values - values.max())
    return exps / exps.sum()


class GLiNERAdapter:
    """Load-back wrapper around a GLiNER2 classifier; the only GLiNER API the RL stage sees."""

    def __init__(self, classifier: Any, *, schema: Any, model_hash: str,
                 batch_size: int = 8) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self._classifier = classifier
        self.schema = schema
        self._model_hash = model_hash
        self.batch_size = batch_size

    @classmethod
    def load(cls, base_dir: str | Path, *, adapter_dir: str | Path | None = None,
             revision: str | None = None, device: str | None = None,
             batch_size: int = 8) -> "GLiNERAdapter":
        """Load a local base snapshot plus, when given, its LoRA adapter.

        CUDA bf16 is used when available (verified on the P0 gate's L4), else CPU fp32.
        ``revision`` is the base snapshot's HF revision and is recorded in the model hash.
        """
        base = Path(base_dir)
        if not base.is_dir():
            raise FileNotFoundError(f"base model directory not found: {base}")
        adapter = Path(adapter_dir) if adapter_dir is not None else None
        if adapter is not None and not adapter.is_dir():
            raise FileNotFoundError(f"adapter directory not found: {adapter}")
        # Hash before the heavy import: a broken snapshot must fail without pulling
        # gliner2/torch into the process.
        model_hash = checkpoint_model_hash(base, adapter_dir=adapter, revision=revision)

        import torch
        from gliner2 import AutoExtractor
        from gliner2.classification.engine import Classifier
        from gliner2.classification.schema import ClassificationSchema

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.device(device).type == "cuda" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        extractor = AutoExtractor.from_pretrained(str(base))
        if adapter is not None:
            from peft import PeftModel

            extractor = PeftModel.from_pretrained(extractor, str(adapter))
        extractor.to(device=device, dtype=dtype)
        classifier = Classifier(extractor, device=device, dtype=dtype)

        schema = ClassificationSchema()
        schema.single(QUESTION_ID, dict(ACTION_CRITERIA), instruction=INSTRUCTIONS)
        return cls(classifier, schema=schema, model_hash=model_hash, batch_size=batch_size)

    def model_hash(self) -> str:
        """Identifies base + adapter + schema this adapter loaded; cache keys embed it."""
        return self._model_hash

    def predict_probs(self, states: Sequence[str]) -> np.ndarray:
        """``[N, 4]`` float64 probabilities in ``ACTION_ORDER``, one row per input state."""
        if self._classifier is None:
            raise RuntimeError("this adapter was closed; load a checkpoint before predicting")
        texts = [str(state) for state in states]
        out = np.zeros((len(texts), len(ACTION_ORDER)), dtype=np.float64)
        if not texts:
            return out
        scores: list = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start:start + self.batch_size]
            results = self._classifier.batch_score(chunk, self.schema)
            if len(results) != len(chunk):
                raise ValueError(f"batch_score returned {len(results)} results "
                                 f"for {len(chunk)} states")
            scores.extend(results)
        for row, result in enumerate(scores):
            out[row] = _row_probs(result, row)
        return out

    def close(self) -> None:
        """Drop the model and release GPU memory (safe to call twice)."""
        if self._classifier is None:
            return
        self._classifier = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover - torch is present wherever a real model ran
            pass
