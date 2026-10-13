"""Pinned dense encoder for VixSTORY V2 retrieval: AITeamVN/Vietnamese_Embedding.

V2 plan §6 pins the model (revision ``dea33aa1ab339f38d66ae0a40e6c40e0a9249568``, an
xlm-roberta/bge-m3-style encoder with CLS pooling), the runtime embedding dimension (1024)
and max sequence length (2048), and the behavior contract:

* raw Vietnamese text on both sides — **no** E5-style ``query:``/``passage:`` prefixes;
* L2-normalized embeddings, so inner product == cosine similarity;
* CUDA OOM halves the batch size; a failure at batch size 1 raises — there is deliberately
  no silent fallback to another model (in particular no E5 substitution);
* ``trust_remote_code=False`` plus a strict ``modules.json`` allow-list, so the pinned
  snapshot can never execute arbitrary remote code, and ONNX/evaluation artifacts are not
  part of ``MODEL_FILES`` and are therefore never downloaded.

Ranking reuses :func:`src.retrieval.dense.rank_by_inner_product`; this module only owns
the encoder.
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from huggingface_hub import snapshot_download

from src.utils.hashing import sha256_file

MODEL_REPO = "AITeamVN/Vietnamese_Embedding"
MODEL_REVISION = "dea33aa1ab339f38d66ae0a40e6c40e0a9249568"
EMBED_DIM = 1024
MAX_SEQ_LENGTH = 2048
EXPECTED_POOLING = "cls"
DEFAULT_BATCH_SIZE = 32

#: Exactly the files the runtime needs — no ONNX export, no README, no evaluation code.
MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "modules.json",
    "sentence_bert_config.json",
    "config_sentence_transformers.json",
    "1_Pooling/config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
)

_ALLOWED_MODULES = ("Transformer", "Pooling", "Normalize")
_MODULE_PREFIX = "sentence_transformers.models."


def download_model(*, cache_dir: str | Path | None = None) -> tuple[Path, dict]:
    """Download exactly :data:`MODEL_FILES` and verify the resolved revision + config.

    Returns ``(snapshot_path, {name: {sha256, bytes}})``.
    """
    path = Path(snapshot_download(MODEL_REPO, revision=MODEL_REVISION,
                                  allow_patterns=list(MODEL_FILES),
                                  cache_dir=str(cache_dir) if cache_dir else None))
    if path.name != MODEL_REVISION:
        raise ValueError(
            f"snapshot_download resolved {path.name!r} != pinned revision {MODEL_REVISION!r}")
    verify_model_files(path)
    files = {name: {"sha256": sha256_file(path / name),
                    "bytes": (path / name).stat().st_size} for name in MODEL_FILES}
    return path, files


def verify_model_files(snapshot: str | Path) -> dict:
    """Strictly verify the pinned snapshot layout; returns a config report.

    Refuses anything that is not the pinned configuration: remote/unsupported module types
    in ``modules.json``, missing Transformer/Pooling/Normalize modules, CLS pooling off, a
    different embedding dimension, or a missing runtime file.
    """
    snapshot = Path(snapshot)
    modules_path = snapshot / "modules.json"
    if not modules_path.is_file():
        raise ValueError(
            f"refusing {snapshot}: modules.json is missing — not a SentenceTransformer snapshot")
    modules = json.loads(modules_path.read_text(encoding="utf-8"))
    kinds: list[str] = []
    pooling_config_path: Path | None = None
    for module in modules:
        module_type = str(module.get("type", ""))
        if not module_type.startswith(_MODULE_PREFIX):
            raise ValueError(
                f"refusing module type {module_type!r}: remote or unsupported code is not permitted")
        kind = module_type[len(_MODULE_PREFIX):]
        if kind not in _ALLOWED_MODULES:
            raise ValueError(
                f"refusing module type {module_type!r}: remote or unsupported code is not permitted")
        kinds.append(kind)
        if kind == "Pooling":
            rel = module.get("path") or "1_Pooling"
            pooling_config_path = snapshot / rel / "config.json"
    missing_modules = [kind for kind in _ALLOWED_MODULES if kind not in kinds]
    if missing_modules:
        raise ValueError(f"modules.json is missing required module(s): {missing_modules}")
    for name in MODEL_FILES:
        if not (snapshot / name).is_file():
            raise ValueError(f"refusing {snapshot}: required model file {name} is missing")
    pooling = json.loads(pooling_config_path.read_text(encoding="utf-8"))
    if not pooling.get("pooling_mode_cls_token"):
        raise ValueError(
            f"the pinned model requires CLS pooling (pooling_mode_cls_token=True), "
            f"but {pooling_config_path} says otherwise")
    dimension = pooling.get("word_embedding_dimension")
    if dimension != EMBED_DIM:
        raise ValueError(f"word_embedding_dimension {dimension} != pinned {EMBED_DIM}")
    bert_config = json.loads((snapshot / "sentence_bert_config.json").read_text(encoding="utf-8"))
    return {"repo": MODEL_REPO, "revision": MODEL_REVISION, "pooling": EXPECTED_POOLING,
            "embedding_dimension": int(dimension),
            "config_max_seq_length": bert_config.get("max_seq_length"),
            "modules": kinds, "files": list(MODEL_FILES)}


def _build_sentence_transformer(model_path: Any, device: str | None = None):
    """Build the SentenceTransformer with remote code execution disabled."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(str(model_path), device=device or "cpu", trust_remote_code=False)


def _is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


@dataclass
class VietnameseEmbeddingEncoder:
    """Prefix-free, L2-normalized encoder for the pinned Vietnamese_Embedding model."""

    model_path: str | Path
    batch_size: int = DEFAULT_BATCH_SIZE
    device: str | None = "cuda"

    _model: object | None = field(default=None, init=False, repr=False, compare=False)
    config_report: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    effective_batch_size: int | None = field(default=None, init=False, repr=False, compare=False)

    def load(self):
        if self._model is None:
            self.config_report = verify_model_files(self.model_path)
            model = _build_sentence_transformer(self.model_path, device=self.device)
            model.max_seq_length = MAX_SEQ_LENGTH
            dimension = int(model.get_embedding_dimension())
            if dimension != EMBED_DIM:
                raise ValueError(
                    f"loaded model embedding dimension {dimension} != pinned {EMBED_DIM}")
            self._model = model
        return self._model

    def unload(self) -> None:
        self._model = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def dimension(self) -> int:
        return int(self.load().get_embedding_dimension())

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, "passages")

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, "queries")

    def _encode(self, texts: Sequence[str], what: str) -> np.ndarray:
        texts = [str(text) for text in texts]
        if not texts:
            raise ValueError(f"refusing to encode an empty batch of {what}")
        model = self.load()
        batch = int(self.batch_size)
        while True:
            try:
                vectors = model.encode(texts, batch_size=batch, normalize_embeddings=True,
                                       convert_to_numpy=True, show_progress_bar=False)
                break
            except RuntimeError as exc:
                if not _is_oom(exc):
                    raise
                if batch <= 1:
                    raise RuntimeError(
                        f"CUDA out of memory at batch_size=1 encoding {what}; stopping — "
                        "the pinned model has no fallback (no E5 substitution)") from exc
                batch = batch // 2
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape != (len(texts), EMBED_DIM):
            raise ValueError(
                f"expected {(len(texts), EMBED_DIM)} embeddings, got {vectors.shape}")
        if not np.all(np.isfinite(vectors)):
            raise ValueError("encoder produced non-finite values")
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError("encoder output vectors are not normalized")
        self.effective_batch_size = batch
        return vectors
