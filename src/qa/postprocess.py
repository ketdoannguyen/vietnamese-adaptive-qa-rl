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

    Overall EM/F1 are deliberately **not** the selection objective: they are computed with
    an empty reference for impossible questions, so on a dataset with a large impossible
    share an always-null reader scores ``em_overall`` equal to that share and beats any
    answering threshold (a real pilot sweep was decided by the all-null row at the bottom of
    the grid). Selection follows D008 instead — feasibility first (>=1 non-empty span with
    valid provenance, `02_IMPLEMENTATION/03_QA_READER_TRAINING_AND_CACHE.md` §10), then
    ``f1_answerable + no_answer_accuracy``, then ``em_answerable``, then the lower threshold.
    Per-row metrics are kept as-is so both the constraint and the objective stay auditable.
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
    """§10's acceptance requirement, per threshold: at least one non-empty span with valid
    provenance on the val predictions.

    ``answered_*`` come from :func:`src.qa.reader_eval.evaluate_predictions` (a question
    counts as answered when the threshold left a non-null span), and ``provenance_failures``
    are subtracted so the count stays correct if a caller passes unverified predictions.
    """
    answered = int(row["answered_answerable"]) + int(row["answered_impossible"])
    return answered - int(row.get("provenance_failures", 0)) >= 1


def threshold_objective(row: Mapping) -> float:
    """D008 selection objective: ``f1_answerable + no_answer_accuracy``.

    The two terms live on disjoint question sets (answerable / impossible), so abstaining on
    one side cannot inflate the other; the constraint above is still needed because an
    always-null reader trivially maximizes ``no_answer_accuracy`` at 1.0.
    """
    return float(row["f1_answerable"]) + float(row["no_answer_accuracy"])


def selection_key(row: Mapping) -> tuple:
    """D008 ordering: feasible thresholds first, then objective, EM, lower threshold."""
    return (not acceptance_constraint_met(row), -threshold_objective(row),
            -float(row["em_answerable"]), float(row["threshold"]))


def choose_best_threshold(sweep: Sequence[Mapping]) -> dict:
    """Best row of a threshold sweep under :func:`selection_key`.

    Raises ``ValueError`` when no threshold emits a non-empty span with valid provenance:
    the reader answers nothing at every point of the grid, P3's acceptance check cannot be
    met, and shipping the always-null row would silently freeze a reader that abstains
    always (D008).
    """
    if not sweep:
        raise ValueError("empty threshold sweep")
    best = dict(min(sweep, key=selection_key))
    if not acceptance_constraint_met(best):
        thresholds = [float(row["threshold"]) for row in sweep]
        raise ValueError(
            f"no threshold in {min(thresholds)}..{max(thresholds)} ({len(sweep)} points) emits "
            f"a non-empty span with valid provenance on val: the reader abstains at every "
            f"point (best objective {threshold_objective(best):.4f}), so the §10 acceptance "
            "check cannot pass and the run must fail (D008)")
    return best


def selection_report(sweep: Sequence[Mapping]) -> dict:
    """Audit record of the selection decision, stored next to the sweep in ``reader_eval.json``.

    ``constraint_binding`` is true when the row the objective alone would pick fails the
    acceptance constraint, so the constraint — not the objective — decided the threshold.
    """
    chosen = choose_best_threshold(sweep)
    objective_only = min(sweep, key=lambda row: (-threshold_objective(row),
                                                 -float(row["em_answerable"]),
                                                 float(row["threshold"])))
    return {
        "objective": "f1_answerable + no_answer_accuracy",
        "constraint": "at least one non-empty span with valid provenance at the chosen "
                      "threshold (03_QA_READER_TRAINING_AND_CACHE.md §10)",
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
