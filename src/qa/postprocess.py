"""SQuAD2-style span decoding (`02_IMPLEMENTATION/03_...md` §3, §4).

Decoding is split in two so the null threshold can be tuned on validation **without
re-running the model**: :func:`decode_windows` turns logits into the best non-null span per
(question, document) plus the CLS/null score, and :func:`apply_threshold` makes the final
null-or-answer decision. That is what makes a threshold sweep cheap — and it keeps the
comparison "null span vs best non-null span" visible in one place.

``answer_text`` is always sliced out of the document text, so a prediction is a substring
of the visible evidence *by construction*; :func:`verify_provenance` is still applied as
defence in depth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from src.qa.features import FeatureMeta
from src.qa.reader import NULL_PREDICTION, ReaderPrediction, verify_provenance


@dataclass(frozen=True)
class BestSpan:
    """Best non-null candidate for one (question, document) pair, before the threshold."""

    qid: str
    doc_id: str
    answer_text: str
    start_char: int
    end_char: int
    non_null_score: float
    null_score: float

    @property
    def null_margin(self) -> float:
        return self.null_score - self.non_null_score

    def to_dict(self) -> dict:
        return {"qid": self.qid, "doc_id": self.doc_id, "answer_text": self.answer_text,
                "start_char": self.start_char, "end_char": self.end_char,
                "non_null_score": self.non_null_score, "null_score": self.null_score,
                "null_margin": self.null_margin}

    def to_prediction(self, *, threshold: float, reader_model_hash: str,
                      doc_text: str) -> ReaderPrediction:
        """Apply the null threshold and verify provenance against the cited document."""
        prediction = ReaderPrediction(
            qid=self.qid, doc_id=self.doc_id, answer_text=self.answer_text,
            start_char=self.start_char, end_char=self.end_char,
            non_null_score=self.non_null_score, null_score=self.null_score,
            null_margin=self.null_margin, reader_model_hash=reader_model_hash,
            provenance_ok=True)
        if self.null_score - self.non_null_score > threshold:
            return ReaderPrediction(**{**prediction.to_dict(), "answer_text": None,
                                       "start_char": None, "end_char": None})
        return verify_provenance(prediction, doc_text)


#: Score of a window that contains no usable span at all. Keeps the "null" decision
#: explicit: ``null_score - (-inf)`` is always above any threshold, so such a pair is
#: reported as null rather than silently dropped from the cache.
NO_SPAN_SCORE = float("-inf")


def best_span_in_window(meta: FeatureMeta, start_logits: np.ndarray, end_logits: np.ndarray,
                        context_text: str, *, max_answer_length: int) -> BestSpan:
    """Highest-scoring valid span inside this window's context.

    Returns the sentinel span (:data:`NO_SPAN_SCORE`, empty answer) when the window has no
    valid span, so every (question, document) pair always has a decoded row.
    """
    null_score = float(start_logits[0] + end_logits[0])
    span = meta.context_span()
    if span is None:
        return _no_span(meta, null_score)
    context_start, context_end = span
    best: tuple[float, int, int] | None = None
    for start in range(context_start, context_end + 1):
        start_offset = meta.offset_mapping[start]
        if start_offset[0] == start_offset[1]:  # a special token inside the sequence
            continue
        for end in range(start, min(start + max_answer_length, context_end + 1)):
            end_offset = meta.offset_mapping[end]
            if end_offset[0] == end_offset[1]:
                continue
            score = float(start_logits[start] + end_logits[end])
            if best is None or score > best[0]:
                best = (score, start, end)
    if best is None:
        return _no_span(meta, null_score)
    score, start, end = best
    start_char, end_char = meta.offset_mapping[start][0], meta.offset_mapping[end][1]
    if end_char <= start_char:
        return _no_span(meta, null_score)
    return BestSpan(qid=meta.qid, doc_id=meta.doc_id,
                    answer_text=context_text[start_char:end_char],
                    start_char=start_char, end_char=end_char,
                    non_null_score=score, null_score=null_score)


def _no_span(meta: FeatureMeta, null_score: float) -> BestSpan:
    return BestSpan(qid=meta.qid, doc_id=meta.doc_id, answer_text="",
                    start_char=0, end_char=0, non_null_score=NO_SPAN_SCORE,
                    null_score=null_score)


def decode_windows(metas: Sequence[FeatureMeta], start_logits: np.ndarray, end_logits: np.ndarray,
                   context_by_doc: Mapping[str, str], *,
                   max_answer_length: int = 30) -> dict[tuple[str, str], BestSpan]:
    """Best span per (qid, doc_id) across all sliding windows of that pair."""
    if start_logits.shape[0] != len(metas):
        raise ValueError(f"{start_logits.shape[0]} logit rows for {len(metas)} features")
    out: dict[tuple[str, str], BestSpan] = {}
    for i, meta in enumerate(metas):
        if meta.doc_id not in context_by_doc:
            raise KeyError(f"no context text for document {meta.doc_id!r}")
        span = best_span_in_window(meta, start_logits[i], end_logits[i],
                                   context_by_doc[meta.doc_id], max_answer_length=max_answer_length)
        key = (span.qid, span.doc_id)
        if key not in out or span.non_null_score > out[key].non_null_score:
            out[key] = span
    return out


def apply_threshold(spans: Iterable[BestSpan], *, threshold: float, reader_model_hash: str,
                    context_by_doc: Mapping[str, str]) -> dict[tuple[str, str], ReaderPrediction]:
    return {(span.qid, span.doc_id): span.to_prediction(
        threshold=threshold, reader_model_hash=reader_model_hash,
        doc_text=context_by_doc[span.doc_id]) for span in spans}


def sweep_threshold(spans: Mapping[tuple[str, str], BestSpan], gold: Mapping[str, Mapping],
                    thresholds: Sequence[float], *, reader_model_hash: str,
                    context_by_doc: Mapping[str, str], evaluate) -> list[dict]:
    """Evaluate every threshold on the same decoded spans (validation only).

    Each threshold is scored under the rule the pipeline actually uses: the highest-scoring
    candidate that survived the threshold becomes that question's prediction
    (:func:`best_prediction_per_question`). ``evaluate`` is
    :func:`src.qa.reader_eval.evaluate_predictions`, injected to keep this module free of
    metric policy.

    Rows keep every per-threshold metric plus two derived audit fields
    (:func:`acceptance_constraint_met`, :func:`threshold_objective`) and are sorted by
    :func:`selection_key`, so ``sweep[0]`` is the row the selection rule picks.

    Selection follows D010: feasibility first — both §10 acceptance requirements
    (`02_IMPLEMENTATION/03_QA_READER_TRAINING_AND_CACHE.md`: >=1 non-empty span with valid
    provenance, and >=1 impossible question returned null) — then ``f1_overall``, then
    ``em_overall``, then the lower threshold. The earlier D008 sum
    (``f1_answerable + no_answer_accuracy``) is algebraically ``1 + (f1_answerable -
    false_answer_rate)`` and cannot prefer answering for a weak reader; the full argument
    and the medium-reader evidence live in :func:`threshold_objective`. Per-row metrics are
    kept as-is so both the constraints and the objective stay auditable.
    """
    rows = []
    for threshold in thresholds:
        predictions = apply_threshold(spans.values(), threshold=threshold,
                                      reader_model_hash=reader_model_hash,
                                      context_by_doc=context_by_doc)
        metrics = evaluate(best_prediction_per_question(predictions.values()), gold)
        row = {"threshold": float(threshold), **metrics}
        row["acceptance_constraint_met"] = acceptance_constraint_met(row)
        row["selection_objective"] = threshold_objective(row)
        rows.append(row)
    rows.sort(key=selection_key)
    return rows


def acceptance_constraint_met(row: Mapping) -> bool:
    """Both of §10's acceptance requirements, per threshold.

    (1) at least one non-empty span with valid provenance on the val predictions;
    (2) at least one impossible question returned null (``answered_impossible <
    n_impossible``, equivalently ``no_answer_accuracy > 0``). A threshold satisfying only
    (1) below every impossible margin would answer every negative, the other degenerate
    extreme §8.6 flags — the reader is a candidate generator (§1), so the sweep must never
    freeze either an always-abstain or an always-answer reader.

    ``answered_*`` come from :func:`src.qa.reader_eval.evaluate_predictions` (a question
    counts as answered when the threshold left a non-null span), and ``provenance_failures``
    are subtracted so the count stays correct if a caller passes unverified predictions.
    """
    answered = int(row["answered_answerable"]) + int(row["answered_impossible"])
    non_empty_span = answered - int(row.get("provenance_failures", 0)) >= 1
    nulls_an_impossible = int(row["answered_impossible"]) < int(row["n_impossible"])
    return non_empty_span and nulls_an_impossible


def threshold_objective(row: Mapping) -> float:
    """D010 selection objective: ``f1_overall`` (the SQuAD2-style overall F1).

    Impossible questions are scored against the empty reference, so a correct null
    contributes 1.0 and abstaining cannot win the metric by itself; the two §10 constraints
    (:func:`acceptance_constraint_met`) keep the degenerate extremes out of the selection.

    D008's ``f1_answerable + no_answer_accuracy`` is abandoned because it is structurally
    abstention-biased: algebraically it equals ``1 + (f1_answerable - false_answer_rate)``,
    so it can only prefer answering by the margin answerable F1 exceeds the false-answer
    rate. On the medium reader's sweep (val: 3107 answerable / 1292 impossible) no answering
    threshold did: every answering row stayed below 1.0 (at 0.5: 0.476 + 0.238 = 0.714)
    while the near-abstain row at -4.5 — which answers 41/3107 answerable questions, F1
    0.010 — scored 1.0008 = 0.010 + 0.991. The rule therefore selected a reader whose forced
    -decode F1 is 0.547, and the same frozen reader scores ``f1_overall`` 0.298 there versus
    0.406 at the row this objective selects.
    """
    return float(row["f1_overall"])


def selection_key(row: Mapping) -> tuple:
    """D010 ordering: feasible thresholds first, then overall F1, overall EM, lower threshold."""
    return (not acceptance_constraint_met(row), -threshold_objective(row),
            -float(row["em_overall"]), float(row["threshold"]))


def choose_best_threshold(sweep: Sequence[Mapping]) -> dict:
    """Best row of a threshold sweep under :func:`selection_key`.

    Raises ``ValueError`` when no threshold satisfies both §10 acceptance constraints: the
    reader answers nothing at every point of the grid, or it never returns null on an
    impossible question. Either extreme would be frozen as the deployed reader, so the run
    must fail instead (D010); the caller keeps the sweep on disk as evidence.
    """
    if not sweep:
        raise ValueError("empty threshold sweep")
    best = dict(min(sweep, key=selection_key))
    if not acceptance_constraint_met(best):
        thresholds = [float(row["threshold"]) for row in sweep]
        raise ValueError(
            f"no threshold in {min(thresholds)}..{max(thresholds)} ({len(sweep)} points) "
            "meets both §10 acceptance constraints on val — at least one non-empty span with "
            "valid provenance, and at least one impossible question returned null — so the "
            f"acceptance check cannot pass and the run must fail (D010; best objective "
            f"{threshold_objective(best):.4f})")
    return best


def selection_report(sweep: Sequence[Mapping]) -> dict:
    """Audit record of the selection decision, stored next to the sweep in ``reader_eval.json``.

    ``constraint_binding`` is true when the row the objective alone would pick fails a §10
    constraint, so feasibility — not the objective — decided the threshold.
    """
    chosen = choose_best_threshold(sweep)
    objective_only = min(sweep, key=lambda row: (-threshold_objective(row),
                                                 -float(row["em_overall"]),
                                                 float(row["threshold"])))
    return {
        "objective": "f1_overall",
        "constraint": "both §10 acceptance constraints at the chosen threshold: at least one "
                      "non-empty span with valid provenance and at least one impossible "
                      "question returned null (03_QA_READER_TRAINING_AND_CACHE.md §10)",
        "chosen_threshold": float(chosen["threshold"]),
        "chosen_objective": threshold_objective(chosen),
        "n_thresholds": len(sweep),
        "n_feasible_thresholds": sum(1 for row in sweep if acceptance_constraint_met(row)),
        "objective_only_threshold": float(objective_only["threshold"]),
        "constraint_binding": not acceptance_constraint_met(objective_only),
    }


def best_prediction_per_question(predictions: Iterable[ReaderPrediction]) -> dict[str, ReaderPrediction]:
    """Highest-scoring non-null prediction per question; NULL_PREDICTION when there is none.

    Score-only selection, exactly like the environment's candidate choice (spec 05 §5).
    """
    out: dict[str, ReaderPrediction] = {}
    for prediction in predictions:
        if prediction.answer_text is None or not prediction.provenance_ok:
            continue
        current = out.get(prediction.qid)
        if current is None or prediction.non_null_score > current.non_null_score:
            out[prediction.qid] = prediction
    return out


__all__ = ["BestSpan", "best_span_in_window", "decode_windows", "apply_threshold",
           "sweep_threshold", "choose_best_threshold", "selection_report",
           "acceptance_constraint_met", "threshold_objective", "selection_key",
           "best_prediction_per_question", "NULL_PREDICTION"]
