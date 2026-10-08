"""QA reader interface and cached prediction lookup.

The reader itself (XLM-R base fine-tuned on train_strict) is trained in P3. This module
defines the *contract* the environment depends on, plus the cache that lets RL rollouts
run without the reader resident on the GPU.

Contract highlights (`02_IMPLEMENTATION/03_QA_READER_TRAINING_AND_CACHE.md` §5):
- ``answer_text`` is either ``None`` or a substring of the cited document, verified
  against the stored offsets. A prediction that fails the substring check is reported
  with ``provenance_ok=False`` and must never be offered as an answer.
- ``non_null_score`` / ``null_score`` come straight from the model. They are **not**
  calibrated correctness probabilities and are never reported as such.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from src.utils.hashing import sha256_json


@dataclass(frozen=True)
class ReaderPrediction:
    qid: str
    doc_id: str | None
    answer_text: str | None
    start_char: int | None
    end_char: int | None
    non_null_score: float
    null_score: float
    null_margin: float
    reader_model_hash: str
    provenance_ok: bool
    #: Feature-only; explicitly NOT a probability that the answer is correct.
    max_context: bool = True
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "qid": self.qid, "doc_id": self.doc_id, "answer_text": self.answer_text,
            "start_char": self.start_char, "end_char": self.end_char,
            "non_null_score": self.non_null_score, "null_score": self.null_score,
            "null_margin": self.null_margin, "reader_model_hash": self.reader_model_hash,
            "provenance_ok": self.provenance_ok, "max_context": self.max_context,
            "error": self.error,
        }


NULL_PREDICTION = ReaderPrediction(
    qid="", doc_id=None, answer_text=None, start_char=None, end_char=None,
    non_null_score=float("-inf"), null_score=float("-inf"), null_margin=0.0,
    reader_model_hash="", provenance_ok=False,
)


def verify_provenance(prediction: ReaderPrediction, doc_text: str) -> ReaderPrediction:
    """Assert the predicted span really is a substring of the cited document.

    Spec: "``answer_text`` phải là substring trong visible doc theo offsets hoặc null.
    Nếu không match, không hiển thị câu trả lời đó và log ``provenance_error``."
    """
    if prediction.answer_text is None or prediction.doc_id is None:
        return prediction
    start, end = prediction.start_char, prediction.end_char
    ok = (
        start is not None and end is not None
        and 0 <= start <= end <= len(doc_text)
        and doc_text[start:end] == prediction.answer_text
    )
    if ok:
        return prediction
    return ReaderPrediction(
        **{**prediction.to_dict(), "provenance_ok": False, "answer_text": None,
           "start_char": None, "end_char": None,
           "error": "provenance_error"}
    )


def reader_cache_key(qid: str, visible_doc_ids: Sequence[str], reader_sha: str,
                     tokenizer_sha: str, inference_config_sha: str) -> str:
    """Cache key contract from `02_IMPLEMENTATION/09_...md` §2.

    Never key a reader prediction on bare ``qid``: different seen-document sets produce
    different candidates, and keying on qid alone silently mixes them.
    """
    return sha256_json({
        "qid": qid,
        "visible_doc_ids_ordered": list(visible_doc_ids),
        "reader_sha": reader_sha,
        "tokenizer_sha": tokenizer_sha,
        "reader_inference_config_sha": inference_config_sha,
    })


class ReaderCache:
    """In-memory view over precomputed reader predictions.

    A miss is a hard error unless a fallback callable is supplied: silently returning an
    empty prediction would let the agent "answer" on fabricated evidence.
    """

    def __init__(self, entries: Mapping[str, ReaderPrediction] | None = None,
                 *, fallback=None) -> None:
        self._entries: dict[str, ReaderPrediction] = dict(entries or {})
        self._fallback = fallback

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: str) -> ReaderPrediction | None:
        return self._entries.get(key)

    def put(self, key: str, prediction: ReaderPrediction) -> None:
        self._entries[key] = prediction

    def lookup(self, key: str) -> ReaderPrediction:
        hit = self._entries.get(key)
        if hit is not None:
            return hit
        if self._fallback is not None:
            pred = self._fallback(key)
            self._entries[key] = pred
            return pred
        raise KeyError(f"reader cache miss for key {key[:16]}...; "
                       "refusing to substitute an empty prediction")


def best_across_docs(predictions: Iterable[ReaderPrediction]) -> ReaderPrediction:
    """Pick the highest-scoring non-null candidate across documents.

    Selection uses the reader's own score only — never the gold answer (spec 05 §5).
    """
    best: ReaderPrediction | None = None
    for pred in predictions:
        if pred.answer_text is None or not pred.provenance_ok:
            continue
        if best is None or pred.non_null_score > best.non_null_score:
            best = pred
    return best if best is not None else NULL_PREDICTION
