"""Episodic controlled-corpus construction (``04_DATASET_VALIDITY...`` §5).

For every question one episode corpus is built from documents of the *same* strict split:
the question's own context plus ``n_distractors`` unique contexts, preferring a different
``title_group``. Sampling is seeded from ``sha256(qid, scenario_seed, scenario_version)``
(never Python's ``hash()``) and drawn from the sorted document pool, so a scenario is
bit-identical for a given pool and seed and changes when the seed changes.

``reachable_bm25`` / ``reachable_dense`` stay ``None`` here — P2 measures them.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from src.data.schemas import DocumentRecord, ScenarioRecord
from src.data.normalizer import NormalizedRow, tokenize_words
from src.utils.hashing import stable_seed

SCENARIO_VERSION = 1


def scenario_id_for(split: str, qid: str, version: int = SCENARIO_VERSION) -> str:
    return f"sc:{split}:{qid}:v{version}"


def build_scenarios(
    rows: Sequence[NormalizedRow],
    docs_by_id: Mapping[str, DocumentRecord],
    *,
    split: str,
    n_distractors: int,
    scenario_seed: int,
    version: int = SCENARIO_VERSION,
) -> tuple[list[ScenarioRecord], list[dict[str, str]]]:
    """Build one scenario per row; return ``(scenarios, exclusions)``.

    Exclusions are logged, never silent: a positive whose gold context cannot be placed in
    the corpus (or whose gold offsets are unrecoverable) is dropped *with a reason*.
    """
    pool_ids = sorted(docs_by_id)
    scenarios: list[ScenarioRecord] = []
    exclusions: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda r: r.question.qid):
        q = row.question
        private = q.private
        source_doc = private.gold_doc_id
        assert source_doc is not None, f"{q.qid}: missing gold_doc_id"
        if source_doc not in docs_by_id:
            exclusions.append({"qid": q.qid, "split": split, "stage": "scenario",
                               "reason": "gold_context_missing_from_pool"})
            continue
        if not private.is_impossible and row.offset_unrecoverable:
            exclusions.append({"qid": q.qid, "split": split, "stage": "scenario",
                               "reason": "offset_unrecoverable:" + row.offset_unrecoverable[0]})
            continue

        rng = np.random.default_rng(stable_seed(q.qid, scenario_seed, version))
        source_group = docs_by_id[source_doc].title_group
        others = [d for d in pool_ids if d != source_doc]
        other_title = [d for d in others if docs_by_id[d].title_group != source_group]
        same_title = [d for d in others if docs_by_id[d].title_group == source_group]
        chosen: list[str] = []
        for bucket in (other_title, same_title):
            if len(chosen) >= n_distractors:
                break
            take = min(n_distractors - len(chosen), len(bucket))
            if take:
                order = rng.permutation(len(bucket))[:take]
                chosen.extend(bucket[int(i)] for i in order)

        corpus = [source_doc] + chosen
        shuffle = rng.permutation(len(corpus))
        corpus = [corpus[int(i)] for i in shuffle]
        assert len(set(corpus)) == len(corpus), f"{q.qid}: duplicate doc in corpus"

        n_other = sum(1 for d in chosen if docs_by_id[d].title_group != source_group)
        difficulty = {
            "answerable": not private.is_impossible,
            "source_doc_id": source_doc,
            "source_context_in_corpus": source_doc in corpus,
            "gold_context_in_corpus": None if private.is_impossible else private.gold_doc_id in corpus,
            "n_distractors_requested": n_distractors,
            "n_distractors_placed": len(chosen),
            "n_distractors_other_title": n_other,
            "n_distractors_same_title": len(chosen) - n_other,
            "n_pool_docs": len(pool_ids),
            "question_tokens": len(tokenize_words(q.question)),
            "corpus_tokens_total": sum(len(tokenize_words(docs_by_id[d].text)) for d in corpus),
        }
        scenarios.append(ScenarioRecord(
            scenario_id=scenario_id_for(split, q.qid, version),
            qid=q.qid,
            split=split,
            corpus_doc_ids=tuple(corpus),
            scenario_seed=scenario_seed,
            gold_doc_id=None if private.is_impossible else private.gold_doc_id,
            reachable_bm25=None,
            reachable_dense=None,
            difficulty=difficulty,
        ))
    return scenarios, exclusions


def scenario_summary(scenarios: Sequence[ScenarioRecord]) -> dict[str, Any]:
    """Aggregate corpus statistics for the split manifest (no labels leaked outward)."""
    if not scenarios:
        return {"scenarios": 0}
    sizes = [len(s.corpus_doc_ids) for s in scenarios]
    placed = [s.difficulty.get("n_distractors_placed", 0) for s in scenarios]
    requested = [s.difficulty.get("n_distractors_requested", 0) for s in scenarios]
    return {
        "scenarios": len(scenarios),
        "corpus_size_min": int(min(sizes)),
        "corpus_size_max": int(max(sizes)),
        "corpus_size_mean": round(sum(sizes) / len(sizes), 3),
        "scenarios_with_full_distractors": int(sum(1 for a, b in zip(placed, requested) if a == b)),
        "scenarios_with_fewer_distractors": int(sum(1 for a, b in zip(placed, requested) if a < b)),
        "answerable": int(sum(1 for s in scenarios if s.difficulty.get("answerable"))),
        "impossible": int(sum(1 for s in scenarios if not s.difficulty.get("answerable"))),
        "gold_context_in_corpus_all": all(
            s.difficulty.get("gold_context_in_corpus") is not False for s in scenarios),
    }
