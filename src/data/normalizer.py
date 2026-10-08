"""Normalize raw ViQuAD2 rows into ``QuestionRecord`` / ``DocumentRecord`` pairs.

Rules taken from ``01_SPECIFICATIONS/04_DATASET_VALIDITY_AND_LEAKAGE.md`` and
``02_IMPLEMENTATION/01_DOWNLOAD_VALIDATE_PREPROCESS.md``:

* ``qid = "<source-split>:<id>"``; a duplicate id gets a checksum suffix instead of
  silently overwriting the earlier row.
* ``context_raw`` is preserved exactly (Unicode and offsets). ``context_display`` is
  NFC with whitespace collapsed; the display offsets are derived through an explicit
  index map and verified, never by blind re-search.
* Gold answers live only in ``QuestionRecord.private``; ``DocumentRecord`` is content only.
* ``is_impossible=True`` must come with empty gold answers. This dataset's convention is
  verified at run time and logged, not assumed.
* A broken ``answer_start`` is repaired only when the repair is unambiguous; every repair
  and every failure is recorded in ``offset_issue``.
"""

from __future__ import annotations

import re
import unicodedata
from array import array
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

from src.data.schemas import DocumentRecord, QuestionPrivate, QuestionRecord
from src.utils.hashing import canonical_json, content_doc_id, normalize_text, normalize_title, sha256_text

_WORD_RE = re.compile(r"\w+", re.UNICODE)

#: Small Vietnamese function-word list used by the cheap lexical heuristics
#: (negative audit, difficulty features). Not a linguistic claim — a heuristic.
STOPWORDS = frozenset("""
và của là các có trong một những được cho với không người này đã trên khi từ đến về như
tại theo để do cũng đó thì ở ra vào sau trước giữa nhiều ít hơn nhất rất đang sẽ bởi vì
nếu nhưng mà hay hoặc nên phải cần bị cùng cả cái chiếc con ông bà anh chị em nó họ
chúng tôi ta mình ai gì nào đâu sao thế bao nhiêu mấy lúc nơi đây kia ấy vậy ạ à ừ
một_hai tuy dù song còn lại rồi chưa vẫn cứ mãi luôn thường đôi ba bốn năm sáu bảy tám
chín mười phần trăm triệu tỷ thứ tự cuối đầu tiên tiếp theo
""".split())

#: question -> expected answer-type keyword families (used by the reader proxy only)
_QTYPE_HINTS = {
    "who": ("ai", "danh xưng", "tên gọi"),
    "when": ("khi nào", "năm nào", "thời gian nào", "bao giờ"),
    "where": ("ở đâu", "nơi nào", "địa điểm"),
    "howmany": ("bao nhiêu", "số lượng", "mấy"),
    "what": ("gì", "nào", "loại nào", "kiểu gì"),
}


def tokenize_words(text: str) -> list[str]:
    """Whitespace/punctuation-insensitive word tokens (Unicode aware)."""
    return _WORD_RE.findall(text)


def content_words(text: str) -> list[str]:
    """Lower-cased tokens minus stopwords and single characters."""
    return [w for w in (t.casefold() for t in tokenize_words(text))
            if w not in STOPWORDS and len(w) > 1]


def question_type(question: str) -> str:
    q = question.casefold()
    for kind, hints in _QTYPE_HINTS.items():
        if any(h in q for h in hints):
            return kind
    return "other"


# --------------------------------------------------------------------------------------
# display mapping: raw -> NFC + whitespace-collapsed, with a verified index map
# --------------------------------------------------------------------------------------

def _segments(raw: str) -> Iterator[tuple[int, int]]:
    """Yield (start, end) spans that NFC can compose: starter + combining marks."""
    start = 0
    for i, ch in enumerate(raw):
        if unicodedata.combining(ch) == 0 and i > start:
            yield start, i
            start = i
    if start < len(raw):
        yield start, len(raw)


def _nfc_index_map(raw: str) -> tuple[str, array | None]:
    """NFC text plus raw-index -> nfc-index map (None = identity)."""
    if unicodedata.is_normalized("NFC", raw):
        return raw, None
    out: list[str] = []
    idx = array("i", [-1]) * 0 if False else array("i", bytes(4 * len(raw)))
    for i in range(len(raw)):
        idx[i] = -1
    for start, end in _segments(raw):
        base = len(out)
        out.extend(unicodedata.normalize("NFC", raw[start:end]))
        for i in range(start, end):
            idx[i] = base
    text = "".join(out)
    if text != unicodedata.normalize("NFC", raw):  # composition crossed a segment boundary
        return text, None
    return text, idx


def build_display_map(raw: str) -> tuple[str, array | None]:
    """Return (display_text, index_map).

    ``index_map`` maps a raw character index to its position in ``display_text``
    (``-1`` when the character was dropped). ``None`` means ``display_text == raw``.
    """
    display = normalize_text(raw)
    if display == raw:
        return display, None
    nfc_text, nfc_map = _nfc_index_map(raw)
    collapsed: list[str] = []
    cmap = array("i", bytes(4 * len(nfc_text)))
    for i in range(len(nfc_text)):
        cmap[i] = -1
    prev_space = False
    for i, ch in enumerate(nfc_text):
        if ch.isspace():
            if not collapsed or prev_space:
                continue
            collapsed.append(" ")
            cmap[i] = len(collapsed) - 1
            prev_space = True
        else:
            collapsed.append(ch)
            cmap[i] = len(collapsed) - 1
            prev_space = False
    if collapsed and collapsed[-1] == " ":
        collapsed.pop()
        for i in range(len(cmap)):
            if cmap[i] >= len(collapsed):
                cmap[i] = -1
    text = "".join(collapsed)
    if text != display:  # should be unreachable; keeps the map honest if it is not
        return display, None
    idx = array("i", bytes(4 * len(raw)))
    for i in range(len(raw)):
        j = nfc_map[i] if nfc_map is not None else i
        idx[i] = cmap[j] if j >= 0 else -1
    return text, idx


class TextNormalizer:
    """Builds and caches display/index maps per unique raw context.

    A context is repeated by ~7 questions on average, so caching turns an O(rows)
    cost into an O(unique contexts) one.
    """

    def __init__(self) -> None:
        self._cache: dict[str, tuple[str, array | None]] = {}

    def display(self, raw: str) -> str:
        return self._entry(raw)[0]

    def _entry(self, raw: str) -> tuple[str, array | None]:
        key = sha256_text(raw)
        entry = self._cache.get(key)
        if entry is None:
            entry = build_display_map(raw)
            self._cache[key] = entry
        return entry

    def span(self, raw: str, start: int, length: int) -> tuple[int, int] | None:
        """Map a verified raw span into display coordinates, or None if unmappable."""
        display, idx = self._entry(raw)
        if idx is None:
            ds, de = start, start + length
        else:
            if start < 0 or start + length > len(idx):
                return None
            ds = idx[start]
            if ds < 0:
                return None
            de = -1
            for j in range(start + length - 1, start - 1, -1):
                if idx[j] >= 0:
                    de = idx[j] + 1
                    break
            if de <= ds:
                return None
        span_text = display[ds:de]
        if span_text != normalize_text(raw[start:start + length]):
            return None
        return ds, de


# --------------------------------------------------------------------------------------
# field extraction
# --------------------------------------------------------------------------------------

def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _as_start_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (int,)):
        return [int(value)]
    return [int(v) for v in value]


def extract_answers(answers_field: Any) -> tuple[tuple[str, ...], tuple[int, ...], bool]:
    """Normalize ``answers`` in any of its observed shapes.

    Returns ``(texts, starts, field_missing)``. Raises ``ValueError`` when the two
    lists disagree in length — that contradicts the declared feature schema and must
    never be silently truncated.
    """
    if answers_field is None:
        return (), (), True
    if isinstance(answers_field, Mapping):
        texts = _as_text_list(answers_field.get("text"))
        starts = _as_start_list(answers_field.get("answer_start"))
    elif isinstance(answers_field, (list, tuple)):
        texts = [str(item.get("text", "")) for item in answers_field]
        starts = [int(item.get("answer_start", -1)) for item in answers_field]
    else:
        raise ValueError(f"unsupported answers field type: {type(answers_field).__name__}")
    if len(texts) != len(starts):
        raise ValueError(f"answers text/answer_start length mismatch: {len(texts)} != {len(starts)}")
    if any(s < 0 for s in starts):
        raise ValueError("negative answer_start")
    return tuple(texts), tuple(starts), False


def verify_offsets(
    context_raw: str, texts: Sequence[str], starts: Sequence[int]
) -> tuple[tuple[int, ...], tuple[str, ...], tuple[str, ...]]:
    """Check every gold span against ``context_raw``.

    Returns ``(starts_out, recovered, unrecoverable)`` where ``recovered`` /
    ``unrecoverable`` hold human-readable issue strings. A repair is applied only when
    the position is unambiguous:

    1. exact match at the recorded offset            -> no issue;
    2. recorded offset sits on the space before the answer -> shift +1;
    3. the answer sits exactly one character earlier    -> shift -1;
    4. the answer occurs exactly once in the context    -> use that occurrence;
    5. anything else                                    -> keep the original offset and
       report ``offset_unrecoverable`` (never a silent fix).
    """
    out: list[int] = []
    recovered: list[str] = []
    unrecoverable: list[str] = []
    for text, start in zip(texts, starts):
        if start + len(text) <= len(context_raw) and context_raw[start:start + len(text)] == text:
            out.append(start)
            continue
        occurrences = [m.start() for m in re.finditer(re.escape(text), context_raw)] if text else []
        if len(occurrences) > 1:
            if start + 1 in occurrences and start < len(context_raw) and context_raw[start].isspace():
                out.append(start + 1)
                recovered.append(f"offset_recovered_space_prefix:{start}->{start + 1}")
                continue
            if start - 1 in occurrences:
                out.append(start - 1)
                recovered.append(f"offset_recovered_neighbor:-1:{start}->{start - 1}")
                continue
            out.append(start)
            unrecoverable.append(f"offset_unrecoverable:multiple_matches:{start}:{len(occurrences)}")
            continue
        if len(occurrences) == 1:
            out.append(occurrences[0])
            recovered.append(f"offset_recovered_unique_match:{start}->{occurrences[0]}")
            continue
        out.append(start)
        unrecoverable.append(f"offset_unrecoverable:no_match:{start}")
    return tuple(out), tuple(recovered), tuple(unrecoverable)


# --------------------------------------------------------------------------------------
# row normalization
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class NormalizedRow:
    """One normalized question plus every quality signal the reports need."""

    question: QuestionRecord
    document: DocumentRecord
    context_raw: str
    title_raw: str
    context_sha_raw: str
    answers_field_missing: bool
    n_gold_answers: int
    offset_exact: int
    offset_recovered: tuple[str, ...] = ()
    offset_unrecoverable: tuple[str, ...] = ()
    answer_starts_display: tuple[int, ...] = ()
    answer_texts_display: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def display_offsets_ok(self) -> bool:
        return len(self.answer_starts_display) == self.n_gold_answers


class RecordNormalizer:
    """Normalizes rows within one source split, tracking qid uniqueness."""

    def __init__(self) -> None:
        self._seen_qids: set[str] = set()
        self.text = TextNormalizer()

    def _unique_qid(self, raw_id: str | None, example: Mapping[str, Any], split: str) -> tuple[str, str | None]:
        base = str(raw_id) if raw_id not in (None, "") else \
            f"row-{sha256_text(canonical_json(_jsonable(example)))[:12]}"
        qid = f"{split}:{base}"
        if qid not in self._seen_qids:
            self._seen_qids.add(qid)
            return qid, None
        note = f"duplicate_id:{base}"
        suffix = sha256_text(canonical_json(_jsonable(example)))[:8]
        candidate = f"{qid}#{suffix}"
        k = 2
        while candidate in self._seen_qids:
            candidate = f"{qid}#{suffix}-{k}"
            k += 1
        self._seen_qids.add(candidate)
        return candidate, note

    def normalize(self, example: Mapping[str, Any], split: str) -> NormalizedRow:
        """Normalize one raw example. Raises ``ValueError`` for malformed rows."""
        notes: list[str] = []
        qid, dup_note = self._unique_qid(example.get("id"), example, split)
        if dup_note:
            notes.append(dup_note)

        title_raw = example.get("title") or ""
        context_raw = example.get("context")
        if context_raw is None:
            raise ValueError("missing context")
        if not isinstance(context_raw, str):
            context_raw = str(context_raw)
        question_raw = example.get("question")
        if question_raw is None:
            raise ValueError("missing question")
        question = normalize_text(str(question_raw))
        if not question:
            raise ValueError("empty question")
        if not context_raw.strip():
            raise ValueError("empty context")

        texts, starts, answers_missing = extract_answers(example.get("answers"))
        if answers_missing:
            notes.append("answers_field_none")
        is_impossible = bool(example.get("is_impossible"))

        if is_impossible and texts:
            raise ValueError("is_impossible=True with non-empty gold answers")
        if not is_impossible and not texts:
            raise ValueError("answerable row without gold answers")

        starts, recovered, unrecoverable = verify_offsets(context_raw, texts, starts)
        if recovered:
            notes.extend(recovered)
        if unrecoverable:
            notes.extend(unrecoverable)

        plausible_texts: tuple[str, ...] = ()
        if is_impossible:
            pa_texts, pa_starts, pa_missing = extract_answers(example.get("plausible_answers"))
            plausible_texts = tuple(t for t, _ in zip(pa_texts, pa_starts))
            if pa_missing:
                notes.append("plausible_answers_field_none")
        else:
            pa_texts, _, _ = extract_answers(example.get("plausible_answers"))
            if pa_texts:
                notes.append("plausible_answers_present_on_answerable_row")

        display = self.text.display(context_raw)
        doc_id = content_doc_id(context_raw)
        title_group = normalize_title(title_raw)
        if not title_group:
            title_group = f"ctx:{doc_id[:16]}"
            notes.append("title_group_fallback_to_context_hash")

        starts_display: list[int] = []
        texts_display: list[str] = []
        for text, start in zip(texts, starts):
            span = self.text.span(context_raw, start, len(text))
            if span is None:
                notes.append(f"display_offset_unmappable:{start}")
                continue
            starts_display.append(span[0])
            texts_display.append(display[span[0]:span[1]])

        offset_issue = "; ".join(recovered + unrecoverable) or None
        if texts and len(starts_display) != len(texts):
            offset_issue = "; ".join(filter(None, [offset_issue, "display_offsets_incomplete"]))

        private = QuestionPrivate(
            answer_texts=texts,
            answer_starts=starts,
            is_impossible=is_impossible,
            plausible_texts=plausible_texts,
            gold_doc_id=doc_id,
            offset_issue=offset_issue,
        )
        question_record = QuestionRecord(
            qid=qid,
            source_split=split,
            title_group=title_group,
            question=question,
            private=private,
        )
        document = DocumentRecord(doc_id=doc_id, title_group=title_group, text=display,
                                  source_split=split)
        return NormalizedRow(
            question=question_record,
            document=document,
            context_raw=context_raw,
            title_raw=title_raw,
            context_sha_raw=sha256_text(context_raw),
            answers_field_missing=answers_missing,
            n_gold_answers=len(texts),
            offset_exact=len(texts) - len(recovered) - len(unrecoverable),
            offset_recovered=recovered,
            offset_unrecoverable=unrecoverable,
            answer_starts_display=tuple(starts_display),
            answer_texts_display=tuple(texts_display),
            notes=tuple(notes),
        )


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def normalize_record(example: Mapping[str, Any], split: str) -> NormalizedRow:
    """Stateless single-row convenience wrapper (no cross-row duplicate detection)."""
    return RecordNormalizer().normalize(example, split)


@dataclass
class RowIssues:
    """Aggregated quality counters for one source split (see ``auditor``)."""

    counts: dict[str, int] = field(default_factory=dict)

    def bump(self, key: str, n: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + n
