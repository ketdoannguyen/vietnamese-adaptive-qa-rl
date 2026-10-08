"""Exhaustive reader-prediction cache over the reachable retrieval states (`03_...md` §6).

One row per (question, visible-document state): the frozen reader's best candidate for that
exact visible set, keyed by the reader-candidate key of `09_...md` §2. The search budget
bounds the retrieval tree to at most 7 states, so the cache is exhaustive over it — an RL
rollout over the same rankings can never miss. A miss therefore means an input changed
(rankings, reader weights, threshold), and the RL side must rebuild rather than invent a
prediction: :func:`load_state_cache` returns a cache whose ``lookup`` raises on a miss.

Rows carry no gold and no labels: ``visible_doc_ids`` is what the agent saw, everything else
is the reader's own output. Rollout usage is two lines:

    meta = load_state_cache_meta(meta_path)
    candidate = cache.lookup(state_cache_key(meta, qid, seen_doc_ids))
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from src.qa.infer import PairInput
from src.qa.postprocess import apply_threshold
from src.qa.reader import NULL_PREDICTION, ReaderCache, ReaderPrediction, reader_cache_key
from src.retrieval.states import reachable_states, retrieved_doc_ids

SCHEMA_VERSION = "v1"
#: Bounded report of inputs that were skipped, so a partial cache is never silent.
MAX_REPORTED = 20


def state_cache_key(meta: Mapping[str, Any], qid: str, visible_doc_ids: Sequence[str]) -> str:
    """The key a rollout computes for ``(qid, seen docs)`` from the cache's own meta."""
    for field in ("reader_model_hash", "tokenizer_sha", "inference_config_sha"):
        if not meta.get(field):
            raise ValueError(f"state cache meta is missing {field!r}")
    return reader_cache_key(qid, visible_doc_ids, meta["reader_model_hash"],
                            meta["tokenizer_sha"], meta["inference_config_sha"])


def best_state_prediction(predictions: Sequence[ReaderPrediction]) -> ReaderPrediction:
    """The environment's rule for one visible-document state (``best_across_docs``).

    Highest-scoring candidate that survived the threshold; when none did, the highest-scoring
    pair is still reported — with ``answer_text=None`` — so the state keeps its honest scores
    instead of a fabricated one. Selection uses the reader's own score, never gold.
    """
    if not predictions:
        return NULL_PREDICTION
    survivors = [p for p in predictions if p.answer_text is not None and p.provenance_ok]
    return max(survivors or list(predictions), key=lambda p: p.non_null_score)


def build_state_cache(*, scenarios: Sequence[Any], questions: Mapping[str, str],
                      docs: Mapping[str, str], rankings: Mapping[str, Mapping[str, Sequence[str]]],
                      inference: Any, threshold: float, reader_model_hash: str,
                      tokenizer_sha: str, inference_config_sha: str,
                      retrieval: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rows for every reachable state of every scenario, plus build statistics.

    ``inference`` is a :class:`src.qa.infer.ReaderInference` (duck-typed: ``predict(pairs)``).
    A scenario without a ranking is skipped and reported; a document referenced by a ranking
    but absent from ``docs`` is skipped and reported — never silently answered from nothing.
    A ranking that references a document outside the scenario's own corpus raises: such a row
    is stale (built for a different corpus, D011) and the states it would produce are ones
    the environment must refuse, so the cache is never built from it.
    """
    by_qid: dict[str, Any] = {}
    states_by_qid: dict[str, list[tuple[str, ...]]] = {}
    missing_ranking: list[str] = []
    for scenario in scenarios:
        qid = scenario.qid
        by_qid[qid] = scenario
        ranking = rankings.get(qid)
        if not ranking:
            missing_ranking.append(qid)
            continue
        corpus = set(scenario.corpus_doc_ids)
        for tool, ranked in ranking.items():
            outside = [doc_id for doc_id in ranked if doc_id not in corpus]
            if outside:
                raise ValueError(
                    f"{scenario.scenario_id}: ranking for tool {tool!r} references "
                    f"{len(outside)} doc(s) outside the episode corpus, e.g. {outside[0]!r} "
                    "— the P2 ranking cache is stale for this scenario; rerun build_retrieval "
                    "(D011)")
        states_by_qid[qid] = reachable_states(
            ranking, bootstrap_tool=retrieval["initial_bootstrap"],
            bootstrap_passages=int(retrieval["bootstrap_passages"]),
            passage_per_search_action=int(retrieval["passage_per_search_action"]),
            max_optional_search=int(retrieval["optional_searches"]))

    # Deduplicate pairs: the bootstrap document appears in every state of a question.
    pairs: list[PairInput] = []
    seen_pairs: set[tuple[str, str]] = set()
    missing_docs: list[str] = []
    for qid, states in states_by_qid.items():
        if qid not in questions:
            raise KeyError(f"scenario for {qid!r} has no question row")
        for doc_id in retrieved_doc_ids(states):
            if doc_id not in docs:
                missing_docs.append(f"{qid}:{doc_id}")
                continue
            if (qid, doc_id) in seen_pairs:
                continue
            seen_pairs.add((qid, doc_id))
            pairs.append(PairInput(qid=qid, doc_id=doc_id, question=questions[qid],
                                   context=docs[doc_id]))
    spans = inference.predict(pairs)

    rows: list[dict[str, Any]] = []
    n_null = 0
    for qid in sorted(states_by_qid):
        scenario = by_qid[qid]
        for state in states_by_qid[qid]:
            state_spans = [spans[(qid, doc_id)] for doc_id in state if (qid, doc_id) in spans]
            predictions = apply_threshold(state_spans, threshold=threshold,
                                          reader_model_hash=reader_model_hash,
                                          context_by_doc=docs)
            candidate = best_state_prediction(list(predictions.values()))
            n_null += int(candidate.answer_text is None)
            rows.append({
                "cache_key": reader_cache_key(qid, state, reader_model_hash, tokenizer_sha,
                                              inference_config_sha),
                "qid": qid, "split": scenario.split, "scenario_id": scenario.scenario_id,
                "visible_doc_ids": list(state), "n_visible_docs": len(state),
                "doc_id": candidate.doc_id, "answer_text": candidate.answer_text,
                "start_char": candidate.start_char, "end_char": candidate.end_char,
                "non_null_score": candidate.non_null_score,
                "null_score": candidate.null_score, "null_margin": candidate.null_margin,
                "provenance_ok": bool(candidate.provenance_ok),
                "is_null": candidate.answer_text is None,
                "reader_model_hash": reader_model_hash, "schema_version": SCHEMA_VERSION,
            })
    stats = {
        "n_scenarios": len(scenarios), "n_questions": len(states_by_qid),
        "n_states": len(rows), "n_pairs": len(pairs), "n_null_states": n_null,
        "n_missing_ranking": len(missing_ranking),
        "questions_without_ranking": missing_ranking[:MAX_REPORTED],
        "n_missing_docs": len(missing_docs), "missing_docs": missing_docs[:MAX_REPORTED],
    }
    return rows, stats


def load_state_cache(path: str | Path) -> ReaderCache:
    """Rows -> :class:`ReaderCache` keyed by ``cache_key``; a miss raises (no fallback)."""
    import pyarrow.parquet as pq

    entries: dict[str, ReaderPrediction] = {}
    for row in pq.read_table(str(path)).to_pylist():
        if row.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{path}: schema {row.get('schema_version')!r} "
                             f"!= {SCHEMA_VERSION!r}")
        entries[row["cache_key"]] = ReaderPrediction(
            qid=row["qid"], doc_id=row["doc_id"], answer_text=row["answer_text"],
            start_char=row["start_char"], end_char=row["end_char"],
            non_null_score=float(row["non_null_score"]), null_score=float(row["null_score"]),
            null_margin=float(row["null_margin"]), reader_model_hash=row["reader_model_hash"],
            provenance_ok=bool(row["provenance_ok"]))
    return ReaderCache(entries)


def load_state_cache_meta(path: str | Path) -> dict[str, Any]:
    """The meta sidecar, validated: the hashes a rollout needs must be present."""
    from src.utils.io import read_json

    meta = read_json(path)
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: schema {meta.get('schema_version')!r} != {SCHEMA_VERSION!r}")
    for field in ("reader_model_hash", "tokenizer_sha", "inference_config_sha"):
        if not meta.get(field):
            raise ValueError(f"{path}: missing {field!r}")
    return meta
