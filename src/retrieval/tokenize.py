"""Vietnamese tokenization for BM25.

Spec `02_IMPLEMENTATION/02_BM25_DENSE_RETRIEVAL.md` §2 requires one consistent
tokenizer for both documents and queries, verified on known Vietnamese text. Word
segmentation (`underthesea`/`pyvi`) is optional; when absent we fall back to a
diacritic-aware Unicode word regex, which keeps "Phạm" and "Pham" distinct and never
splits a Vietnamese syllable.
"""

from __future__ import annotations

import re
import unicodedata

#: Unicode letter/digit runs. Keeps Vietnamese diacritics intact.
_WORD_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)

#: Vietnamese stopwords that carry no retrieval signal. Kept deliberately small and
#: fixed: the same list is used for indexing and querying, so it cannot bias either side.
STOPWORDS = frozenset({
    "và", "của", "là", "có", "trong", "được", "cho", "với", "các", "một", "này",
    "đã", "để", "khi", "từ", "trên", "về", "như", "theo", "tại", "bị", "bởi",
    "hay", "hoặc", "nhưng", "mà", "thì", "nên", "vì", "do", "ra", "vào", "lên",
    "ở", "và", "ông", "bà", "nó", "họ", "chúng", "ai", "gì", "nào", "sao",
})


def tokenize_vi(text: str, *, remove_stopwords: bool = False) -> list[str]:
    """NFC -> lowercase -> Unicode word tokens (diacritics preserved)."""
    if not text:
        return []
    text = unicodedata.normalize("NFC", str(text)).lower()
    tokens = _WORD_RE.findall(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def tokenizer_name() -> str:
    return "regex_fallback"


#: Sanity anchors from the spec: these strings must tokenize sensibly.
_SELF_TEST = {
    "Thành phố Hồ Chí Minh": ["thành", "phố", "hồ", "chí", "minh"],
    "Phạm Văn Đồng": ["phạm", "văn", "đồng"],
}


def self_test() -> dict[str, bool]:
    """Returns {input: tokens == expected}. Used by the retrieval tests."""
    return {text: tokenize_vi(text) == expected for text, expected in _SELF_TEST.items()}
