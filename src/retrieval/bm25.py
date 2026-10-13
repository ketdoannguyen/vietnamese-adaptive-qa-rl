"""BM25 retrieval over an explicitly supplied document set.

BM25 has no pretrained weights; it uses corpus statistics and the same Vietnamese
tokenizer for documents and queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from rank_bm25 import BM25Okapi

from src.retrieval.tokenize import tokenize_vi


@dataclass
class BM25Index:
    """BM25Okapi over one document set, with an explicit doc_id order."""

    doc_ids: list[str]
    index: BM25Okapi
    k1: float
    b: float

    @classmethod
    def build(cls, docs: Mapping[str, str], *, k1: float = 1.5, b: float = 0.75,
              remove_stopwords: bool = False) -> "BM25Index":
        doc_ids = list(docs.keys())
        if not doc_ids:
            raise ValueError("cannot build a BM25 index over zero documents")
        corpus = [tokenize_vi(docs[d], remove_stopwords=remove_stopwords) for d in doc_ids]
        # rank_bm25 divides by the average document length; an all-empty corpus makes
        # that zero, so substitute a placeholder token to keep the maths finite.
        corpus = [tokens if tokens else ["<empty>"] for tokens in corpus]
        return cls(doc_ids=doc_ids, index=BM25Okapi(corpus, k1=k1, b=b), k1=k1, b=b)

    def rank(self, query: str, *, top_n: int | None = None) -> list[tuple[str, float]]:
        """Ranked (doc_id, score) pairs, best first, ties broken by doc_id for stability."""
        tokens = tokenize_vi(query)
        if not tokens:
            tokens = ["<empty>"]
        scores = self.index.get_scores(tokens)
        pairs = list(zip(self.doc_ids, (float(s) for s in scores)))
        pairs.sort(key=lambda pair: (-pair[1], pair[0]))
        return pairs[:top_n] if top_n else pairs


def rank_documents(query: str, docs: Mapping[str, str], *, k1: float = 1.5, b: float = 0.75,
                   top_n: int | None = None) -> list[tuple[str, float]]:
    """Convenience wrapper for one-shot ranking of a small episodic corpus."""
    return BM25Index.build(docs, k1=k1, b=b).rank(query, top_n=top_n)
