"""Dense retrieval with multilingual-E5-small.

Spec `02_IMPLEMENTATION/02_BM25_DENSE_RETRIEVAL.md` §3:
- E5 requires **asymmetric prefixes**: ``query: `` for questions, ``passage: `` for
  documents. Encoding both sides with the same prefix degrades the embedding space, so
  the prefixes are part of the cache key.
- Vectors are L2-normalised, so inner product == cosine similarity.
- The model is loaded, used, and unloaded — it must never be resident while the QA
  reader or Laya are on the GPU (15 GB of system RAM makes this non-negotiable).
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

DEFAULT_MODEL = "intfloat/multilingual-e5-small"
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


@dataclass
class DenseEncoder:
    """Thin wrapper around SentenceTransformer with E5 prefix discipline."""

    model_name: str = DEFAULT_MODEL
    revision: str | None = None
    batch_size: int = 32
    max_length: int = 512
    device: str | None = None

    _model: object | None = field(default=None, init=False, repr=False, compare=False)

    def load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            kwargs = {"device": self.device} if self.device else {}
            self._model = SentenceTransformer(self.model_name, revision=self.revision, **kwargs)
            self._model.max_seq_length = self.max_length
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
        return self._encode([PASSAGE_PREFIX + t for t in texts])

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([QUERY_PREFIX + t for t in texts])

    def _encode(self, prefixed: Sequence[str]) -> np.ndarray:
        model = self.load()
        vectors = model.encode(list(prefixed), batch_size=self.batch_size,
                               normalize_embeddings=True, convert_to_numpy=True,
                               show_progress_bar=False)
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(f"expected 2-D embeddings, got shape {vectors.shape}")
        if not np.all(np.isfinite(vectors)):
            raise ValueError("encoder produced non-finite values")
        return vectors


def rank_by_inner_product(query_vec: np.ndarray, doc_vecs: np.ndarray,
                          doc_ids: Sequence[str]) -> list[tuple[str, float]]:
    """Cosine/IP ranking. Both sides are already L2-normalised."""
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)
    if doc_vecs.shape[1] != query_vec.shape[1]:
        raise ValueError(f"dimension mismatch: docs {doc_vecs.shape[1]} vs query {query_vec.shape[1]}")
    scores = doc_vecs @ query_vec[0]
    pairs = list(zip(doc_ids, (float(s) for s in scores)))
    pairs.sort(key=lambda pair: (-pair[1], pair[0]))
    return pairs


class DenseBank:
    """Passage vectors for a fixed document set, with a doc_id -> row mapping."""

    def __init__(self, doc_ids: Sequence[str], vectors: np.ndarray) -> None:
        if len(doc_ids) != vectors.shape[0]:
            raise ValueError(f"{len(doc_ids)} ids but {vectors.shape[0]} vectors")
        self.doc_ids = list(doc_ids)
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.row_of = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}

    @classmethod
    def build(cls, docs: Mapping[str, str], encoder: DenseEncoder) -> "DenseBank":
        doc_ids = list(docs.keys())
        vectors = encoder.encode_passages([docs[d] for d in doc_ids])
        return cls(doc_ids, vectors)

    def subset(self, doc_ids: Sequence[str]) -> "DenseBank":
        rows = [self.row_of[d] for d in doc_ids]
        return DenseBank(list(doc_ids), self.vectors[rows])

    def rank(self, query: str, encoder: DenseEncoder, *,
             restrict_to: Sequence[str] | None = None) -> list[tuple[str, float]]:
        allowed = list(restrict_to) if restrict_to is not None else self.doc_ids
        bank = self.subset(allowed)
        query_vec = encoder.encode_queries([query])
        return rank_by_inner_product(query_vec, bank.vectors, bank.doc_ids)

    def save(self, path: str, *, meta: dict | None = None) -> None:
        """Persist ids/vectors plus a JSON metadata string.

        ``allow_pickle=False`` on load means nothing here can execute code — the doc ids
        travel as a plain unicode array and the metadata as JSON text.
        """
        np.savez_compressed(path,
                            doc_ids=np.array(self.doc_ids, dtype=np.str_),
                            vectors=self.vectors,
                            meta_json=np.array(json.dumps(meta or {}, ensure_ascii=False)))

    @classmethod
    def load(cls, path: str) -> tuple["DenseBank", dict]:
        data = np.load(path, allow_pickle=False)
        bank = cls([str(d) for d in data["doc_ids"]], data["vectors"])
        meta = json.loads(str(data["meta_json"])) if "meta_json" in data else {}
        return bank, meta
