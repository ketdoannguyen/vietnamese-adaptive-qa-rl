"""Vietnamese-aware answer normalization and SQuAD-style scoring.

Normalization rules (applied identically in reward, teacher oracle, and reporting):
NFC -> lowercase -> strip punctuation -> collapse whitespace. **Diacritics are never
removed** — "hoà" and "hoa" are different words and must not be conflated.
"""

from __future__ import annotations

import re
import unicodedata

# Vietnamese-aware tokenizer: letters (incl. diacritics), digits, and _ only.
_TOKEN_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)


def normalize_answer(text: str) -> str:
    """NFC, lowercase, punctuation stripped, whitespace collapsed. Diacritics kept."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text)).lower()
    text = _TOKEN_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    normalized = normalize_answer(text)
    return normalized.split() if normalized else []


def exact_match(prediction: str | None, ground_truths: list[str]) -> float:
    if prediction is None:
        return 0.0
    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(gt) for gt in ground_truths))


def f1_score(prediction: str | None, ground_truths: list[str]) -> float:
    """Max token-F1 over provided references (standard SQuAD definition)."""
    if prediction is None:
        return 0.0
    pred_tokens = tokenize(prediction)
    if not pred_tokens:
        return 0.0
    best = 0.0
    for gt in ground_truths:
        gt_tokens = tokenize(gt)
        if not gt_tokens:
            continue
        # Cap overlap at the ground-truth multiplicity of each token.
        pred_counts: dict[str, int] = {}
        for tok in pred_tokens:
            pred_counts[tok] = pred_counts.get(tok, 0) + 1
        gt_counts: dict[str, int] = {}
        for tok in gt_tokens:
            gt_counts[tok] = gt_counts.get(tok, 0) + 1
        num_same = sum(min(count, gt_counts.get(tok, 0)) for tok, count in pred_counts.items())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best
