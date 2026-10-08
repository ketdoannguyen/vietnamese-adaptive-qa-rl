"""Retrieval diagnostics (`02_IMPLEMENTATION/02_BM25_DENSE_RETRIEVAL.md` §5).

These metrics decide whether the two search actions are *meaningfully different*. If
BM25 and E5 produce nearly the same top-k, the `SEARCH_*` actions are decorative and the
RL result would be uninterpretable — so this is reported, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from src.retrieval.engine import ScenarioRanking


def recall_at_k(ranking: Sequence[str], gold_doc_id: str | None, k: int) -> float:
    """Document recall@k. Returns float('nan') for negatives — they have no gold doc."""
    if gold_doc_id is None:
        return float("nan")
    return 1.0 if gold_doc_id in list(ranking)[:k] else 0.0


def reciprocal_rank(ranking: Sequence[str], gold_doc_id: str | None) -> float:
    if gold_doc_id is None:
        return float("nan")
    for i, doc_id in enumerate(ranking, start=1):
        if doc_id == gold_doc_id:
            return 1.0 / i
    return 0.0


def top_k_overlap(a: Sequence[str], b: Sequence[str], k: int) -> float:
    """Fraction of the two top-k lists that coincide."""
    top_a, top_b = set(list(a)[:k]), set(list(b)[:k])
    if not top_a and not top_b:
        return 1.0
    union = top_a | top_b
    return len(top_a & top_b) / len(union) if union else 1.0


def novelty_rate(ranking: Sequence[str], seen_doc_ids: Sequence[str], k: int) -> float:
    """Share of the top-k that the agent has not already opened."""
    top = list(ranking)[:k]
    if not top:
        return 0.0
    seen = set(seen_doc_ids)
    return sum(1 for d in top if d not in seen) / len(top)


@dataclass
class RetrievalReport:
    n_scenarios: int
    n_answerable: int
    recall_at_1: dict[str, float]
    recall_at_3: dict[str, float]
    recall_at_5: dict[str, float]
    mrr: dict[str, float]
    union_recall_at_5: float
    reachable_rate: dict[str, float]
    topk_overlap_bm25_dense: float
    diff_rate: float

    def to_dict(self) -> dict:
        return {
            "n_scenarios": self.n_scenarios,
            "n_answerable": self.n_answerable,
            "recall_at_1": self.recall_at_1,
            "recall_at_3": self.recall_at_3,
            "recall_at_5": self.recall_at_5,
            "mrr": self.mrr,
            "union_recall_at_5": self.union_recall_at_5,
            "reachable_rate": self.reachable_rate,
            "topk_overlap_bm25_dense": self.topk_overlap_bm25_dense,
            "diff_rate": self.diff_rate,
        }


def _mean(values: list[float]) -> float:
    usable = [v for v in values if v == v]  # drop NaN
    return sum(usable) / len(usable) if usable else float("nan")


def summarise_retrieval(rankings: list[ScenarioRanking],
                        gold_doc_ids: dict[str, str | None], *,
                        reach_budget: Mapping[str, int], overlap_k: int = 3) -> RetrievalReport:
    """Aggregate recall/MRR/reachability/difference over scenarios.

    ``gold_doc_ids`` maps scenario_id -> gold doc id (None for negatives). ``reach_budget``
    is the number of passages each tool can surface in one episode — BM25 gets the forced
    bootstrap plus the optional searches, dense only the optional searches — so the two
    tools are deliberately not given the same budget.

    This is a diagnostic function: it reads gold, and must never run inside a policy path.
    """
    answerable = [(r, gold_doc_ids.get(r.scenario_id)) for r in rankings
                  if gold_doc_ids.get(r.scenario_id)]
    r1: dict[str, float] = {}
    r3: dict[str, float] = {}
    r5: dict[str, float] = {}
    mrr: dict[str, float] = {}
    reachable: dict[str, float] = {}
    for tool in ("bm25", "dense"):
        r1[tool] = _mean([recall_at_k(r.ranked(tool), g, 1) for r, g in answerable])
        r3[tool] = _mean([recall_at_k(r.ranked(tool), g, 3) for r, g in answerable])
        r5[tool] = _mean([recall_at_k(r.ranked(tool), g, 5) for r, g in answerable])
        mrr[tool] = _mean([reciprocal_rank(r.ranked(tool), g) for r, g in answerable])
        reachable[tool] = _mean([
            1.0 if g in list(r.ranked(tool))[:reach_budget[tool]] else 0.0
            for r, g in answerable])

    union_hits = []
    for r, g in answerable:
        union = list(r.ranked("bm25"))[:5] + list(r.ranked("dense"))[:5]
        union_hits.append(1.0 if g in union else 0.0)

    overlaps = [top_k_overlap(r.ranked("bm25"), r.ranked("dense"), overlap_k) for r in rankings]
    overlap_mean = _mean(overlaps)
    diff_rate = _mean([
        1.0 if top_k_overlap(r.ranked("bm25"), r.ranked("dense"), overlap_k) < 1.0 else 0.0
        for r in rankings])

    return RetrievalReport(
        n_scenarios=len(rankings), n_answerable=len(answerable),
        recall_at_1=r1, recall_at_3=r3, recall_at_5=r5, mrr=mrr,
        union_recall_at_5=_mean(union_hits), reachable_rate=reachable,
        topk_overlap_bm25_dense=overlap_mean, diff_rate=diff_rate)


def differing_examples(rankings: list[ScenarioRanking], *, k: int = 3,
                       limit: int = 20) -> list[dict]:
    """Concrete cases where BM25 and E5 disagree — required by spec 02 §8.6."""
    out = []
    for ranking in rankings:
        if top_k_overlap(ranking.ranked("bm25"), ranking.ranked("dense"), k) >= 1.0:
            continue
        out.append({
            "scenario_id": ranking.scenario_id,
            "qid": ranking.qid,
            "top_bm25": list(ranking.ranked("bm25"))[:k],
            "top_dense": list(ranking.ranked("dense"))[:k],
        })
        if len(out) >= limit:
            break
    return out
