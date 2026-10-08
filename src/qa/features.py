"""SQuAD-style feature construction for the Vietnamese extractive reader.

Spec `02_IMPLEMENTATION/03_QA_READER_TRAINING_AND_CACHE.md` §3. One question+context pair
becomes one feature per sliding window. Three rules carry the correctness weight:

* the gold span is selected **per window** — the first answer variant fully inside that
  window's context span; a window that does not contain any gold span is trained as
  no-answer (CLS/CLS), which is exactly how the reader learns to stay silent on a
  distractor passage;
* an impossible question is CLS/CLS in every window;
* an answerable question with **no** window containing a gold span has no usable
  supervision, so it is skipped and counted — never silently turned into a no-answer
  example, which would teach the model to abstain on answerable questions.

Windows are cut from the **full-context token sequence** and re-assembled with the pair's
own special tokens, rather than by asking the tokenizer for overflow windows. In
``tokenizers`` 0.23 ``Tokenizer.encode`` returns at most one overflow chunk, and the new
``Encoding.truncate`` drops the pair's sequence ids — so slicing tokens here is both the
only reliable route and the more explicit one: the document is tokenized once, and every
window carries absolute character offsets into that document. Alignment otherwise follows
the canonical Hugging Face SQuAD conversion, including its
``offsets[start][0] <= start_char < offsets[start][1]`` walk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

CLS_INDEX = 0

#: Texts for the probe pair that reveals a tokenizer's special-token layout. Content is
#: irrelevant — only how many specials sit before, between and after the two sequences.
_PROBE_QUESTION = "a"
_PROBE_CONTEXT = "a"


@dataclass
class Feature:
    """One training window plus the metadata postprocessing needs."""

    input_ids: list[int]
    attention_mask: list[int]
    start_position: int
    end_position: int
    token_type_ids: list[int] | None = None

    def to_model_inputs(self) -> dict[str, Any]:
        row: dict[str, Any] = {"input_ids": self.input_ids,
                               "attention_mask": self.attention_mask,
                               "start_positions": self.start_position,
                               "end_positions": self.end_position}
        if self.token_type_ids is not None:
            row["token_type_ids"] = self.token_type_ids
        return row


@dataclass
class FeatureMeta:
    """Window bookkeeping kept out of the tensors."""

    qid: str
    doc_id: str
    sequence_ids: list[int | None]
    offset_mapping: list[tuple[int, int]]
    answerable: bool
    has_gold_span: bool
    window_index: int

    def context_span(self) -> tuple[int, int] | None:
        """First and last token index belonging to the context (sequence id 1).

        ``None`` when the window holds no context tokens at all — the case for a document
        with no tokenizable content.
        """
        if 1 not in self.sequence_ids:
            return None
        start = self.sequence_ids.index(1)
        end = len(self.sequence_ids) - 1 - self.sequence_ids[::-1].index(1)
        return start, end


@dataclass
class FeatureSet:
    features: list[Feature] = field(default_factory=list)
    metas: list[FeatureMeta] = field(default_factory=list)
    skipped_no_visible_span: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.features)

    def rows(self) -> list[dict[str, Any]]:
        return [f.to_model_inputs() for f in self.features]


def pair_special_layout(tokenizer: Any) -> tuple[int, int, int]:
    """``(prefix, middle, suffix)`` special-token counts of a question/context pair.

    Read off a tiny probe pair so the layout comes from the tokenizer itself instead of a
    hard-coded per-family assumption (``<s> q </s></s> c </s>`` for XLM-R, ``[CLS] q [SEP]
    c [SEP]`` for BERT).
    """
    encoded = tokenizer(_PROBE_QUESTION, _PROBE_CONTEXT, add_special_tokens=True, verbose=False)
    sequence_ids = list(encoded.sequence_ids(0))
    question = [i for i, s in enumerate(sequence_ids) if s == 0]
    context = [i for i, s in enumerate(sequence_ids) if s == 1]
    if not question or not context:
        raise ValueError(f"cannot read the pair layout of {type(tokenizer).__name__}")
    prefix = question[0]
    middle = context[0] - question[-1] - 1
    suffix = len(sequence_ids) - context[-1] - 1
    expected = tokenizer.num_special_tokens_to_add(pair=True)
    if prefix + middle + suffix != expected:
        raise ValueError(f"{type(tokenizer).__name__}: {prefix}+{middle}+{suffix} specials "
                         f"around the pair, expected {expected}")
    return prefix, middle, suffix


def window_bounds(n_tokens: int, *, budget: int, stride: int) -> list[tuple[int, int]]:
    """Sliding-window ``[start, end)`` bounds over ``n_tokens`` context tokens.

    The final window is aligned to the end of the context (the canonical SQuAD convention)
    so the tail is never left as a short stub.
    """
    if budget <= 0:
        raise ValueError(f"context budget {budget} must be positive")
    if not 0 <= stride < budget:
        raise ValueError(f"doc_stride {stride} must be in [0, {budget})")
    if n_tokens <= 0:
        return [(0, 0)]
    starts = []
    start = 0
    while start + budget < n_tokens:
        starts.append(start)
        start += budget - stride
    last = max(0, n_tokens - budget)
    if not starts or starts[-1] != last:
        starts.append(last)
    return [(s, min(s + budget, n_tokens)) for s in starts]


def build_pair_features(
    question: str,
    context: str,
    tokenizer: Any,
    *,
    qid: str,
    doc_id: str,
    answer_texts: Sequence[str] = (),
    answer_starts: Sequence[int] = (),
    is_impossible: bool = False,
    max_length: int = 384,
    doc_stride: int = 128,
) -> tuple[list[Feature], list[FeatureMeta]]:
    """Tokenize one (question, context) pair into sliding-window features."""
    if len(answer_texts) != len(answer_starts):
        raise ValueError(f"{qid}: {len(answer_texts)} answer texts but {len(answer_starts)} starts")
    encoded = tokenizer(question, context, add_special_tokens=True, truncation=False,
                        return_offsets_mapping=True, verbose=False)
    ids = list(encoded["input_ids"])
    sequence_ids = list(encoded.sequence_ids(0))
    offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
    token_type_ids = list(encoded["token_type_ids"]) if "token_type_ids" in encoded else None
    prefix_n, middle_n, suffix_n = pair_special_layout(tokenizer)

    question_run = [i for i, s in enumerate(sequence_ids) if s == 0]
    context_run = [i for i, s in enumerate(sequence_ids) if s == 1]
    if not question_run:
        raise ValueError(f"{qid}: the tokenizer produced no question tokens")
    if context_run and len(context_run) != context_run[-1] - context_run[0] + 1:
        raise ValueError(f"{qid}: the context tokens are not contiguous")
    q_first, q_last = question_run[0], question_run[-1]
    context_ids = [ids[i] for i in context_run]
    head = ids[:q_first]
    question_ids = ids[q_first:q_last + 1]
    between = ids[q_last + 1:q_last + 1 + middle_n]
    closing = ids[q_last + 1 + middle_n + len(context_ids):]
    if len(head) != prefix_n or len(between) != middle_n or len(closing) != suffix_n:
        raise ValueError(f"{qid}: unexpected pair layout {len(head)}/{len(between)}/"
                         f"{len(closing)} specials, expected {prefix_n}/{middle_n}/{suffix_n}")

    budget = max_length - len(head) - len(question_ids) - len(between) - len(closing)
    spans = [(int(s), int(s) + len(t)) for s, t in zip(answer_starts, answer_texts)]
    features: list[Feature] = []
    metas: list[FeatureMeta] = []
    for window_index, (start, end) in enumerate(
            window_bounds(len(context_ids), budget=budget, stride=doc_stride)):
        context_window = context_ids[start:end]
        window_offsets = [(0, 0)] * prefix_n + offsets[q_first:q_last + 1] \
            + [(0, 0)] * middle_n + [offsets[i] for i in context_run[start:end]] \
            + [(0, 0)] * suffix_n
        window_sequence_ids: list[int | None] = [None] * prefix_n + [0] * len(question_ids) \
            + [None] * middle_n + [1] * len(context_window) + [None] * suffix_n
        window_ids = head + question_ids + between + context_window + closing
        if not is_impossible:
            token_span = _locate_span(window_offsets, window_sequence_ids, spans)
        else:
            token_span = None
        start_position, end_position = token_span or (CLS_INDEX, CLS_INDEX)
        pad_n = max_length - len(window_ids)
        if pad_n < 0:
            raise ValueError(f"{qid}: window of {len(window_ids)} tokens exceeds max_length={max_length}")
        row: dict[str, Any] = {
            "input_ids": window_ids + [tokenizer.pad_token_id] * pad_n,
            "attention_mask": [1] * len(window_ids) + [0] * pad_n,
            "start_position": start_position,
            "end_position": end_position,
        }
        if token_type_ids is not None:
            row["token_type_ids"] = (token_type_ids[:q_first] + token_type_ids[q_first:q_last + 1]
                                     + token_type_ids[q_last + 1:q_last + 1 + middle_n]
                                     + [token_type_ids[i] for i in context_run[start:end]]
                                     + token_type_ids[q_last + 1 + middle_n + len(context_ids):]
                                     + [0] * pad_n)
        features.append(Feature(**row))
        metas.append(FeatureMeta(qid=qid, doc_id=doc_id, sequence_ids=window_sequence_ids
                                 + [None] * pad_n, offset_mapping=window_offsets + [(0, 0)] * pad_n,
                                 answerable=not is_impossible,
                                 has_gold_span=token_span is not None, window_index=window_index))
    return features, metas


def _locate_span(offsets: Sequence[tuple[int, int]], sequence_ids: Sequence[int | None],
                 spans: Sequence[tuple[int, int]]) -> tuple[int, int] | None:
    """Token (start, end) for the first gold span fully inside this window, else None."""
    context_start = next((i for i, s in enumerate(sequence_ids) if s == 1), None)
    if context_start is None:
        return None
    context_end = len(sequence_ids) - 1 - next(
        i for i, s in enumerate(reversed(sequence_ids)) if s == 1)
    window_start, window_end = offsets[context_start][0], offsets[context_end][1]

    for start_char, end_char in spans:
        if end_char <= start_char:
            continue
        if start_char < window_start or end_char > window_end:
            continue
        start_token = context_start
        while start_token <= context_end and offsets[start_token][0] <= start_char:
            start_token += 1
        end_token = context_end
        while end_token >= context_start and offsets[end_token][1] >= end_char:
            end_token -= 1
        start_token -= 1
        end_token += 1
        # The token span must cover the character span. Exact character equality is *not*
        # required: a gold answer can begin mid-token, and dropping those examples would
        # remove exactly the hard alignment cases from training.
        if start_token <= end_token and offsets[start_token][1] > start_char \
                and offsets[end_token][0] < end_char:
            return start_token, end_token
    return None


def build_feature_set(rows: Sequence[Mapping[str, Any]], docs: Mapping[str, str], tokenizer: Any,
                      *, max_length: int, doc_stride: int) -> FeatureSet:
    """Build features for normalized question rows (``QuestionRecord.to_row``).

    The training context is ``gold_doc_id`` — for an impossible question that is the source
    passage in which the answer is *absent*, which is precisely the signal a reader needs in
    order to answer null on a plausible distractor.
    """
    out = FeatureSet()
    for row in rows:
        qid, doc_id = row["qid"], row.get("gold_doc_id")
        answerable = not bool(row.get("is_impossible"))
        if doc_id is None or doc_id not in docs:
            out.skipped_no_visible_span.append(qid)
            continue
        features, metas = build_pair_features(
            row["question"], docs[doc_id], tokenizer, qid=qid, doc_id=doc_id,
            answer_texts=tuple(row.get("answer_texts") or ()),
            answer_starts=tuple(row.get("answer_starts") or ()),
            is_impossible=not answerable, max_length=max_length, doc_stride=doc_stride)
        if answerable and not any(meta.has_gold_span for meta in metas):
            out.skipped_no_visible_span.append(qid)
            continue
        out.features.extend(features)
        out.metas.extend(metas)
    return out
