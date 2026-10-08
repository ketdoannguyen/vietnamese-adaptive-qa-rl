"""Deterministic stand-ins for the dense side of retrieval (P2 tests).

The real encoder needs a model download and torch, which makes it a poor unit-test
dependency. These stubs keep every property the engine relies on — prefix discipline,
L2-normalised vectors, inner-product ranking — while being pure numpy. Ranking
determinism can therefore be checked in-process *and* in a fresh subprocess.

The corpus intentionally mixes an easy lexical match (doc_hanoi), a semantic-only
distractor (doc_hcm), and a document that lives in the bank but outside the scenario
corpus (doc_outside) so leakage can be tested rather than assumed.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from src.retrieval.dense import PASSAGE_PREFIX, QUERY_PREFIX, DenseBank
from src.retrieval.engine import ScenarioRanking, rank_scenario
from src.utils.hashing import sha256_json

DIM = 16


def _vector(text: str) -> np.ndarray:
    """Deterministic unit vector: one hash-seeded axis per token, summed and normalised."""
    vec = np.zeros(DIM, dtype=np.float32)
    for token in text.lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        axis = int.from_bytes(digest[:4], "big") % DIM
        vec[axis] += 1.0 if digest[4] % 2 == 0 else -1.0
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        vec[0] = 1.0
        return vec
    return vec / norm


class StubEncoder:
    """Same call surface as ``DenseEncoder``: no model, no network, no torch."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        self.calls.append("query")
        return np.stack([_vector(QUERY_PREFIX + t) for t in texts])

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        self.calls.append("passage")
        return np.stack([_vector(PASSAGE_PREFIX + t) for t in texts])


SCENARIO_ID = "s-001"
QID = "q-001"
QUESTION = "Thủ đô của Việt Nam là gì?"
CORPUS_IDS = ["doc_hanoi", "doc_hcm", "doc_pho", "doc_river"]
DOCS = {
    "doc_hanoi": "Hà Nội là thủ đô của Việt Nam từ năm 1945.",
    "doc_hcm": "Thành phố Hồ Chí Minh là đô thị lớn nhất Việt Nam về dân số.",
    "doc_pho": "Phở là món ăn truyền thống của người Việt Nam.",
    "doc_river": "Sông Mekong chảy qua nhiều quốc gia Đông Nam Á.",
    # Deliberate bait: this document *would* win both rankings on content, so it must be
    # excluded by the corpus restriction alone. If it ever shows up, cross-split leakage
    # has a path into candidate lists.
    "doc_outside": "Hà Nội là thủ đô của Việt Nam. Tài liệu này thuộc split khác.",
}


def build_bank(docs: dict[str, str] | None = None, encoder: StubEncoder | None = None) -> DenseBank:
    encoder = encoder or StubEncoder()
    return DenseBank.build(docs if docs is not None else DOCS, encoder)


def build_ranking(*, question: str = QUESTION, corpus_ids: Sequence[str] = CORPUS_IDS,
                  docs: dict[str, str] | None = None, top_n: int = 32,
                  encoder: StubEncoder | None = None) -> ScenarioRanking:
    encoder = encoder or StubEncoder()
    bank = build_bank(docs, encoder)
    return rank_scenario(scenario_id=SCENARIO_ID, qid=QID, question=question,
                         docs=docs if docs is not None else DOCS,
                         corpus_doc_ids=list(corpus_ids), dense_bank=bank,
                         dense_encoder=encoder, top_n=top_n)


def rows_hash() -> str:
    """Stable digest of the stub scenario's ranking — comparable across processes."""
    return sha256_json(build_ranking().to_row())


if __name__ == "__main__":  # used by the cross-process determinism test
    print(rows_hash())
