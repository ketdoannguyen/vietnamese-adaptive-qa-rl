"""Retrieval engine: full rankings cached per scenario, consumed through a cursor.

Two design points that matter for correctness:

* **Cursor, not re-query.** Spec 05 §4 requires a second BM25 call to return a different
  passage. A stateless "top-k again" would return the same passage forever and make
  multi-step rollouts meaningless. Rankings are computed once per scenario (both tools)
  and walked with ``seen_doc_ids`` as the cursor.
* **Per-scenario BM25 index.** BM25's IDF and average document length are corpus
  statistics. Indexing the whole dataset would let test-split documents influence the
  scores of training scenarios, so the index is built over the scenario corpus alone —
  a scenario's ranking is then a pure function of that scenario.

Rankings never look at gold answers; they are keyed on the question text only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from src.retrieval.bm25 import BM25Index
from src.retrieval.dense import DenseBank, DenseEncoder, rank_by_inner_product
from src.utils.hashing import sha256_json

TOOLS = ("bm25", "dense")


def scenario_corpus_sha(corpus_doc_ids: Sequence[str]) -> str:
    """Fingerprint of one scenario's episode corpus (order included).

    A ``scenario_id`` is ``sc:<split>:<qid>:v<version>`` — it does not capture the corpus,
    and the corpus is not identical across profiles: distractor sampling draws from the
    capped split's document pool, so the same qid gets different distractors at different
    caps (D011). Cached rankings are reused only while this fingerprint matches.
    """
    return sha256_json(list(corpus_doc_ids))


@dataclass
class ScenarioRanking:
    """Ranked doc ids for one scenario, per tool, with the scores that produced them."""

    scenario_id: str
    qid: str
    rankings: dict[str, list[str]] = field(default_factory=dict)
    scores: dict[str, dict[str, float]] = field(default_factory=dict)
    #: Fingerprint of the corpus this ranking was built over; see :func:`scenario_corpus_sha`.
    corpus_sha: str = ""

    def ranked(self, tool: str) -> list[str]:
        if tool not in self.rankings:
            raise KeyError(f"scenario {self.scenario_id!r} has no ranking for tool {tool!r}")
        return self.rankings[tool]

    def score(self, tool: str, doc_id: str) -> float:
        return self.scores.get(tool, {}).get(doc_id, 0.0)

    def to_row(self) -> dict:
        return {"scenario_id": self.scenario_id, "qid": self.qid,
                "rankings": {t: list(v) for t, v in self.rankings.items()},
                "scores": self.scores, "corpus_sha": self.corpus_sha}

    @classmethod
    def from_row(cls, row: Mapping) -> "ScenarioRanking":
        return cls(scenario_id=row["scenario_id"], qid=row["qid"],
                   rankings={t: list(v) for t, v in row["rankings"].items()},
                   scores={t: dict(v) for t, v in row["scores"].items()},
                   corpus_sha=row.get("corpus_sha", ""))


def rank_scenario(*, scenario_id: str, qid: str, question: str,
                  docs: Mapping[str, str], corpus_doc_ids: Sequence[str],
                  dense_bank: DenseBank, dense_encoder: DenseEncoder,
                  top_n: int = 32, bm25_k1: float = 1.5, bm25_b: float = 0.75,
                  ) -> ScenarioRanking:
    """Rank one scenario's corpus with both tools.

    ``docs`` must contain every id in ``corpus_doc_ids``. Both tools see the same
    restricted corpus; a document outside it can never appear in a candidate list.
    Dense ranking is mandatory here — the cache-only path must read ``ScenarioRanking``
    rows through :class:`RankingCache`, never recompute with a stand-in.
    """
    corpus = list(corpus_doc_ids)
    missing = [d for d in corpus if d not in docs]
    if missing:
        raise KeyError(f"scenario {scenario_id}: {len(missing)} corpus doc(s) missing from docs")
    if len(set(corpus)) != len(corpus):
        raise ValueError(f"scenario {scenario_id}: duplicate doc ids in corpus")

    corpus_docs = {doc_id: docs[doc_id] for doc_id in corpus}
    index = BM25Index.build(corpus_docs, k1=bm25_k1, b=bm25_b)
    bm25_pairs = index.rank(question, top_n=top_n)

    bank = dense_bank.subset(corpus)
    query_vec = dense_encoder.encode_queries([question])
    dense_pairs = rank_by_inner_product(query_vec, bank.vectors, bank.doc_ids)[:top_n]

    return ScenarioRanking(
        scenario_id=scenario_id, qid=qid,
        rankings={"bm25": [d for d, _ in bm25_pairs],
                  "dense": [d for d, _ in dense_pairs]},
        scores={"bm25": {d: s for d, s in bm25_pairs},
                "dense": {d: s for d, s in dense_pairs}},
        corpus_sha=scenario_corpus_sha(corpus),
    )


def next_unseen(ranking: Sequence[str], seen_doc_ids: Iterable[str], n: int = 1) -> list[str]:
    """Cursor step: the first ``n`` ranked documents the agent has not already opened."""
    seen = set(seen_doc_ids)
    out = []
    for doc_id in ranking:
        if doc_id not in seen:
            out.append(doc_id)
            seen.add(doc_id)
            if len(out) >= n:
                break
    return out


class RankingCache:
    """Ranking store keyed by scenario id, with a version stamp.

    Loading a persisted cache is what lets the retrieval stage run without a GPU or a
    sentence-transformers model (`retrieval_from_cache_only`); a miss raises rather than
    silently substituting empty or BM25-only candidates.
    """

    def __init__(self, version: str = "v1") -> None:
        self.version = version
        self._by_scenario: dict[str, ScenarioRanking] = {}

    def add(self, ranking: ScenarioRanking) -> None:
        self._by_scenario[ranking.scenario_id] = ranking

    def get(self, scenario_id: str) -> ScenarioRanking:
        try:
            return self._by_scenario[scenario_id]
        except KeyError as exc:
            raise KeyError(
                f"no cached ranking for scenario {scenario_id!r}; a missing ranking must "
                "fail loudly rather than silently fall back to an empty candidate list"
            ) from exc

    def __contains__(self, scenario_id: object) -> bool:
        return scenario_id in self._by_scenario

    def __len__(self) -> int:
        return len(self._by_scenario)

    def rows(self) -> list[dict]:
        return [r.to_row() for r in self._by_scenario.values()]

    @classmethod
    def from_rows(cls, rows: Iterable[Mapping], *, version: str = "v1") -> "RankingCache":
        cache = cls(version=version)
        for row in rows:
            cache.add(ScenarioRanking.from_row(row))
        return cache
