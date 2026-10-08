"""Reader evaluation: answerable vs impossible metrics (`03_...md` §4).

Reported separately and never merged into a single flattering number:

* ``em_answerable`` / ``f1_answerable`` — only questions that have an answer; a null
  prediction scores zero here;
* ``no_answer_accuracy`` / ``false_answer_rate`` — only impossible questions; a non-null
  prediction is a false answer;
* ``em_overall`` / ``f1_overall`` — the SQuAD2 convention where an impossible question has
  the empty string as its reference (predicting null is then correct), so abstention and
  answering are both priced in one number.

Raw model scores are *not* probabilities: nothing here calls a score a confidence.
"""

from __future__ import annotations

from typing import Any, Mapping

from src.qa.metrics import exact_match, f1_score, safe_divide
from src.qa.reader import NULL_PREDICTION, ReaderPrediction


def _answers(row: Mapping[str, Any]) -> list[str]:
    return [str(a) for a in (row.get("answer_texts") or [])]


def evaluate_predictions(predictions: Mapping[str, ReaderPrediction],
                         gold: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Metrics for one condition (oracle-context or retrieved-context).

    ``predictions`` maps qid -> prediction (the best candidate for that question under the
    condition being measured). A qid absent from ``predictions`` counts as a null
    prediction — a question the reader refused to answer.
    """
    totals = {k: 0.0 for k in ("em_ans", "f1_ans", "em_all", "f1_all")}
    counts = {"n": 0, "n_answerable": 0, "n_impossible": 0, "answered_answerable": 0,
              "answered_impossible": 0, "provenance_failures": 0}
    for qid, row in gold.items():
        counts["n"] += 1
        prediction = predictions.get(qid, NULL_PREDICTION)
        if prediction.answer_text is not None and not prediction.provenance_ok:
            counts["provenance_failures"] += 1
        answered = prediction.answer_text is not None
        if row.get("is_impossible"):
            counts["n_impossible"] += 1
            counts["answered_impossible"] += int(answered)
            # Reference for an impossible question is the empty answer.
            em = 1.0 if not answered else 0.0
            totals["em_all"] += em
            totals["f1_all"] += em
        else:
            counts["n_answerable"] += 1
            counts["answered_answerable"] += int(answered)
            answers = _answers(row)
            em = exact_match(prediction.answer_text, answers) if answered else 0.0
            f1 = f1_score(prediction.answer_text, answers) if answered else 0.0
            totals["em_ans"] += em
            totals["f1_ans"] += f1
            totals["em_all"] += em
            totals["f1_all"] += f1
    return {
        **counts,
        "em_answerable": safe_divide(totals["em_ans"], counts["n_answerable"]),
        "f1_answerable": safe_divide(totals["f1_ans"], counts["n_answerable"]),
        "no_answer_accuracy": safe_divide(counts["n_impossible"] - counts["answered_impossible"],
                                          counts["n_impossible"]),
        "false_answer_rate": safe_divide(counts["answered_impossible"], counts["n_impossible"]),
        "answer_rate_answerable": safe_divide(counts["answered_answerable"], counts["n_answerable"]),
        "em_overall": safe_divide(totals["em_all"], counts["n"]),
        "f1_overall": safe_divide(totals["f1_all"], counts["n"]),
    }


def gold_index(rows: Mapping[str, Mapping[str, Any]] | list[Mapping[str, Any]]
               ) -> dict[str, Mapping[str, Any]]:
    """qid -> {answer_texts, is_impossible} (the only gold the metrics may see)."""
    if isinstance(rows, Mapping):
        return {qid: {"answer_texts": row.get("answer_texts") or [],
                      "is_impossible": bool(row.get("is_impossible"))}
                for qid, row in rows.items()}
    return {row["qid"]: {"answer_texts": row.get("answer_texts") or [],
                         "is_impossible": bool(row.get("is_impossible"))} for row in rows}


def threshold_grid(start: float, stop: float, step: float) -> list[float]:
    """Inclusive grid, rounded to avoid float dust in the sweep table."""
    count = int(round((stop - start) / step))
    return [round(start + i * step, 6) for i in range(count + 1)]
