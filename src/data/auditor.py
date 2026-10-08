"""Diagnostics: schema report, leakage report, negative audit.

Everything in this module is computed from the artifacts of the current run — no number
is typed by hand and nothing is reported that was not measured. The negative audit is an
*automated heuristic*; the CSV says so explicitly and never claims human review.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.data.normalizer import (NormalizedRow, content_words, question_type, tokenize_words)
from src.data.schemas import DocumentRecord, ScenarioRecord
from src.utils.hashing import stable_seed

#: A distractor that contains >= this fraction of the question's content words is treated
#: as a possible answer source for a negative question.
LEXICAL_COVERAGE_THRESHOLD = 0.70
#: Cheap reader proxy: max over sliding windows of ``coverage * (0.5 + 0.5 * answer_type)``.
READER_PROXY_THRESHOLD = 0.80
#: A window contains an answer-type token (number for when/how-many, capitalised token
#: for who/where/what).
_WINDOW_STRIDE = 4


# --------------------------------------------------------------------------------------
# schema statistics
# --------------------------------------------------------------------------------------

class SchemaStats:
    """Streaming quality counters for one scope (a raw split or a strict split)."""

    def __init__(self, scope: str) -> None:
        self.scope = scope
        self.rows = 0
        self.malformed = 0
        self.malformed_reasons: Counter[str] = Counter()
        self.answerable = 0
        self.impossible = 0
        self.answers_field_missing = 0
        self.duplicate_qids = 0
        self.title_group_fallback = 0
        self.plausible_missing_on_impossible = 0
        self.plausible_present_on_answerable = 0
        self.offset_exact = 0
        self.offset_recovered = 0
        self.offset_unrecoverable = 0
        self.display_offsets_incomplete = 0
        self.display_differs = 0
        self.multi_answer_rows = 0
        self.unique_title_raw: set[str] = set()
        self.unique_title_group: set[str] = set()
        self.unique_context_raw: set[str] = set()
        self.unique_doc_id: set[str] = set()
        self._doc_raw_variants: dict[str, set[str]] = defaultdict(set)
        self._doc_groups: dict[str, set[str]] = defaultdict(set)
        self.answer_token_lengths: list[int] = []

    def add(self, row: NormalizedRow) -> None:
        self.rows += 1
        private = row.question.private
        if private.is_impossible:
            self.impossible += 1
        else:
            self.answerable += 1
        if row.answers_field_missing:
            self.answers_field_missing += 1
        if any(n.startswith("duplicate_id:") for n in row.notes):
            self.duplicate_qids += 1
        if "title_group_fallback_to_context_hash" in row.notes:
            self.title_group_fallback += 1
        if "plausible_answers_field_none" in row.notes:
            self.plausible_missing_on_impossible += 1
        if "plausible_answers_present_on_answerable_row" in row.notes:
            self.plausible_present_on_answerable += 1
        self.offset_exact += row.offset_exact
        self.offset_recovered += len(row.offset_recovered)
        self.offset_unrecoverable += len(row.offset_unrecoverable)
        if row.document.text != row.context_raw:
            self.display_differs += 1
        if private.answer_texts and len(row.answer_starts_display) != len(private.answer_texts):
            self.display_offsets_incomplete += 1
        if len(private.answer_texts) > 1:
            self.multi_answer_rows += 1
        self.unique_title_raw.add(row.title_raw)
        self.unique_title_group.add(row.question.title_group)
        self.unique_context_raw.add(row.context_sha_raw)
        self.unique_doc_id.add(row.document.doc_id)
        self._doc_raw_variants[row.document.doc_id].add(row.context_sha_raw)
        self._doc_groups[row.document.doc_id].add(row.question.title_group)
        for text in private.answer_texts:
            self.answer_token_lengths.append(len(tokenize_words(text)))

    def add_malformed(self, reason: str) -> None:
        self.rows += 1
        self.malformed += 1
        self.malformed_reasons[reason[:80]] += 1

    def to_dict(self) -> dict[str, Any]:
        lengths = np.asarray(self.answer_token_lengths, dtype=np.float64)
        collisions_raw = sum(1 for v in self._doc_raw_variants.values() if len(v) > 1)
        collisions_group = sum(1 for v in self._doc_groups.values() if len(v) > 1)
        return {
            "scope": self.scope,
            "rows": self.rows,
            "answerable": self.answerable,
            "impossible": self.impossible,
            "answerable_ratio": round(self.answerable / self.rows, 6) if self.rows else None,
            "unique_title_raw": len(self.unique_title_raw),
            "unique_title_group": len(self.unique_title_group),
            "unique_context_raw": len(self.unique_context_raw),
            "unique_doc_id": len(self.unique_doc_id),
            "duplicate_qids": self.duplicate_qids,
            "malformed_rows": self.malformed,
            "malformed_reasons": dict(self.malformed_reasons),
            "answers_field_missing": self.answers_field_missing,
            "plausible_answers_missing_on_impossible": self.plausible_missing_on_impossible,
            "plausible_answers_present_on_answerable": self.plausible_present_on_answerable,
            "offset_exact": self.offset_exact,
            "offset_recovered": self.offset_recovered,
            "offset_unrecoverable": self.offset_unrecoverable,
            "display_offsets_incomplete_rows": self.display_offsets_incomplete,
            "display_differs_from_raw_rows": self.display_differs,
            "multi_answer_rows": self.multi_answer_rows,
            "doc_ids_with_multiple_raw_variants": collisions_raw,
            "doc_ids_in_multiple_title_groups": collisions_group,
            "answer_tokens": {
                "n": int(lengths.size),
                "mean": round(float(lengths.mean()), 4) if lengths.size else None,
                "median": round(float(np.median(lengths)), 4) if lengths.size else None,
                "p95": round(float(np.percentile(lengths, 95)), 4) if lengths.size else None,
            },
        }


# --------------------------------------------------------------------------------------
# overlap / leakage checks
# --------------------------------------------------------------------------------------

def row_sets(rows: Iterable[NormalizedRow]) -> dict[str, set[str]]:
    """The identity sets of a row collection that must not cross a strict-split boundary."""
    out: dict[str, set[str]] = {k: set() for k in OVERLAP_KEYS}
    for row in rows:
        out["qid"].add(row.question.qid)
        out["title_group"].add(row.question.title_group)
        out["doc_id"].add(row.document.doc_id)
        out["raw_context_sha256"].add(row.context_sha_raw)
        out["row_id"].add(row.question.qid.split(":", 1)[1])
    return out


OVERLAP_KEYS = ("qid", "title_group", "doc_id", "raw_context_sha256", "row_id")


def overlap_sets(sets_by_split: Mapping[str, set[str]]) -> dict[str, Any]:
    """Pairwise intersections of one identity set across splits."""
    splits = list(sets_by_split)
    block: dict[str, Any] = {}
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            shared = sets_by_split[a] & sets_by_split[b]
            block[f"{a}~{b}"] = {
                "count": len(shared),
                "examples": sorted(shared)[:3],
                "sizes": {a: len(sets_by_split[a]), b: len(sets_by_split[b])},
            }
    return block


def overlap_block(rows_by_split: Mapping[str, Sequence[NormalizedRow]]) -> dict[str, Any]:
    """Pairwise intersections of qid / title_group / doc_id / raw-context-sha per split."""
    per_split = {name: row_sets(rows) for name, rows in rows_by_split.items()}
    return {key: overlap_sets({name: s[key] for name, s in per_split.items()})
            for key in OVERLAP_KEYS}


def near_duplicate_block(
    docs_by_split: Mapping[str, Mapping[str, DocumentRecord]], *, threshold: float,
    limit: int = 30,
) -> dict[str, Any]:
    splits = list(docs_by_split)
    block: dict[str, Any] = {}
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            docs_a, docs_b = docs_by_split[a], docs_by_split[b]
            report = cross_similarity(sorted(docs_a), [docs_a[d].text for d in sorted(docs_a)],
                                      sorted(docs_b), [docs_b[d].text for d in sorted(docs_b)],
                                      threshold=threshold)
            block[f"{a}~{b}"] = near_duplicate_report(report, limit=limit)
    return block


# --------------------------------------------------------------------------------------
# negative audit (automated heuristic — NOT human review)
# --------------------------------------------------------------------------------------

def lexical_coverage(question: str, text: str) -> float:
    """Fraction of the question's content words present in ``text``."""
    words = set(content_words(question))
    if not words:
        return 0.0
    return len(words & set(content_words(text))) / len(words)


def _answer_type_match(kind: str, tokens: Sequence[str]) -> bool:
    if kind in ("when", "howmany"):
        return any(t.isdigit() for t in tokens)
    if kind in ("who", "where", "what"):
        return any(t[:1].isupper() for t in tokens)
    return False


def reader_proxy(question: str, text: str) -> float:
    """Cheap span reader: max over sliding windows of ``coverage * (0.5 + 0.5 * type)``.

    Not a trained reader — a lexical proxy that flags "a short window of this passage
    contains the question's content words and looks like the answer type asked for".
    """
    words = set(content_words(question))
    if not words:
        return 0.0
    tokens = tokenize_words(text)
    lowered = [t.casefold() for t in tokens]
    window = max(12, 2 * len(words))
    kind = question_type(question)
    best = 0.0
    starts = range(0, max(1, len(lowered) - window + 1), _WINDOW_STRIDE)
    for start in starts:
        chunk = lowered[start:start + window]
        if not chunk:
            continue
        coverage = len(words & set(chunk)) / len(words)
        matches = 1.0 if _answer_type_match(kind, tokens[start:start + window]) else 0.0
        score = coverage * (0.5 + 0.5 * matches)
        if score > best:
            best = score
    return best


def plausible_hit(plausible_texts: Sequence[str], text: str) -> str | None:
    """Return a plausible answer (from the dataset's own annotation) found in ``text``."""
    haystack = text.casefold()
    for candidate in plausible_texts:
        needle = candidate.strip().casefold()
        if len(needle) >= 2 and needle in haystack:
            return candidate
    return None


def select_negatives(scenarios: Sequence[ScenarioRecord], size: int, *, seed: object) -> list[ScenarioRecord]:
    """Deterministic, split-proportional selection of negatives to audit."""
    negatives = [s for s in scenarios if not s.difficulty.get("answerable", True)]
    if not negatives:
        return []
    by_split: dict[str, list[ScenarioRecord]] = defaultdict(list)
    for s in negatives:
        by_split[s.split].append(s)
    for bucket in by_split.values():
        bucket.sort(key=lambda s: (stable_seed(seed, s.qid), s.qid))
    total = len(negatives)
    budget = min(size, total)
    splits = sorted(by_split)
    quota = {s: min(len(by_split[s]), int(budget * len(by_split[s]) / total)) for s in splits}
    remaining = budget - sum(quota.values())
    while remaining > 0:
        progressed = False
        for s in splits:
            if remaining and quota[s] < len(by_split[s]):
                quota[s] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    selected: list[ScenarioRecord] = []
    for s in splits:
        selected.extend(by_split[s][:quota[s]])
    return selected


def audit_negatives(
    scenarios: Sequence[ScenarioRecord],
    rows_by_qid: Mapping[str, NormalizedRow],
    docs_by_id: Mapping[str, DocumentRecord],
    *,
    size: int,
    seed: object,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Audit negatives with documented lexical heuristics; returns (csv rows, summary).

    Columns: ``qid, split, docs, verdict, ambiguous_flag, reviewer_note``. The note always
    states that the verdict is automated and was not verified by a human.
    """
    selected = select_negatives(scenarios, size, seed=seed)
    rows_out: list[dict[str, str]] = []
    n_ambiguous = 0
    reasons: Counter[str] = Counter()
    for scenario in selected:
        row = rows_by_qid[scenario.qid]
        question = row.question.question
        private = row.question.private
        corpus = list(scenario.corpus_doc_ids)
        signals: list[str] = []
        for doc_id in corpus:
            doc = docs_by_id[doc_id]
            coverage = lexical_coverage(question, doc.text)
            if coverage >= LEXICAL_COVERAGE_THRESHOLD:
                signals.append(f"lexical_coverage={coverage:.2f}@doc={doc_id[:12]}")
                reasons["lexical_coverage"] += 1
            score = reader_proxy(question, doc.text)
            if score >= READER_PROXY_THRESHOLD:
                signals.append(f"reader_proxy={score:.2f}@doc={doc_id[:12]}")
                reasons["reader_proxy"] += 1
            hit = plausible_hit(private.plausible_texts, doc.text)
            if hit:
                signals.append(f"plausible_answer_in_doc={hit!r}@doc={doc_id[:12]}")
                reasons["plausible_answer_in_doc"] += 1
        ambiguous = bool(signals)
        n_ambiguous += int(ambiguous)
        note = ("automated heuristic only; NO human review performed. "
                "thresholds lexical_coverage>=0.70, reader_proxy>=0.80, plausible-answer substring.")
        if signals:
            note += " signals: " + "; ".join(signals[:6])
            if len(signals) > 6:
                note += f" (+{len(signals) - 6} more)"
        else:
            note += " signals: none"
        rows_out.append({
            "qid": scenario.qid,
            "split": scenario.split,
            "docs": ";".join(corpus),
            "verdict": "ambiguous" if ambiguous else "clean",
            "ambiguous_flag": "true" if ambiguous else "false",
            "reviewer_note": note,
        })
    summary = {
        "requested": size,
        "negatives_available": sum(1 for s in scenarios if not s.difficulty.get("answerable", True)),
        "audited": len(rows_out),
        "ambiguous": n_ambiguous,
        "clean": len(rows_out) - n_ambiguous,
        "ambiguous_rate": round(n_ambiguous / len(rows_out), 6) if rows_out else None,
        "signal_counts": dict(reasons),
        "human_review": False,
        "method": "automated lexical heuristics (lexical_coverage, reader_proxy, plausible-answer substring)",
    }
    return rows_out, summary


# --------------------------------------------------------------------------------------
# writers
# --------------------------------------------------------------------------------------

def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(_fmt(c) for c in row) + " |")
    return out


def write_schema_report(
    stats: Mapping[str, Mapping[str, Any]],
    path: str | Path,
    *,
    context: Mapping[str, Any],
    exclusions: Sequence[Mapping[str, str]] = (),
    negative_summary: Mapping[str, Any] | None = None,
    notes: Sequence[str] = (),
) -> None:
    lines = ["# P1 data schema report", "",
             "Every number below was computed from the artifacts of the run described in the",
             "context table; nothing is typed by hand.", ""]
    lines += ["## Run context", ""]
    lines += _table(["key", "value"], [[k, v] for k, v in context.items()])
    lines += ["", "## Splits", ""]
    headers = ["scope", "rows", "answerable", "impossible", "answerable ratio", "unique titles",
               "unique title_groups", "unique contexts", "unique doc_ids", "dup qids",
               "malformed", "offset recovered", "offset unrecoverable"]
    table = []
    for scope, s in stats.items():
        table.append([scope, s["rows"], s["answerable"], s["impossible"], s["answerable_ratio"],
                      s["unique_title_raw"], s["unique_title_group"], s["unique_context_raw"],
                      s["unique_doc_id"], s["duplicate_qids"], s["malformed_rows"],
                      s["offset_recovered"], s["offset_unrecoverable"]])
    lines += _table(headers, table)
    lines += ["", "## Gold answer length (tokens)", ""]
    lines += _table(["scope", "n answers", "mean", "median", "p95"],
                    [[scope, s["answer_tokens"]["n"], s["answer_tokens"]["mean"],
                      s["answer_tokens"]["median"], s["answer_tokens"]["p95"]]
                     for scope, s in stats.items()])
    lines += ["", "## Offset quality", "",
              f"Exact = `context_raw[start:start+len(text)] == text`; recovered = unambiguous repair, "
              f"logged per row in `offset_issue`; unrecoverable = kept as-is and reported (rows are "
              f"dropped from scenarios with a logged reason, never silently).", ""]
    lines += _table(["scope", "exact", "recovered", "unrecoverable", "display offsets incomplete (rows)",
                     "rows where display != raw"],
                    [[scope, s["offset_exact"], s["offset_recovered"], s["offset_unrecoverable"],
                      s["display_offsets_incomplete_rows"],
                      s["display_differs_from_raw_rows"]] for scope, s in stats.items()])
    lines += ["", "## Answers / malformed detail", ""]
    lines += _table(["scope", "answers field None", "plausible missing (impossible)",
                     "plausible present (answerable)", "multi-answer rows", "malformed rows",
                     "malformed reasons"],
                    [[scope, s["answers_field_missing"], s["plausible_answers_missing_on_impossible"],
                      s["plausible_answers_present_on_answerable"], s["multi_answer_rows"],
                      s["malformed_rows"], s["malformed_reasons"] or "—"]
                     for scope, s in stats.items()])
    lines += ["", "## Document hash collisions", "",
              "`doc_id` is sha256 of the NFC + whitespace-collapsed context. A collision means two "
              "distinct raw contexts normalise to the same document, which would silently merge "
              "documents (and can break strict-split isolation).", ""]
    lines += _table(["scope", "doc_ids with >1 raw variant", "doc_ids in >1 title_group"],
                    [[scope, s["doc_ids_with_multiple_raw_variants"],
                      s["doc_ids_in_multiple_title_groups"]] for scope, s in stats.items()])
    lines += ["", "## Exclusions", ""]
    if exclusions:
        counter = Counter(e["reason"].split(":", 1)[0] for e in exclusions)
        lines += _table(["reason", "count"], sorted(counter.items()))
        lines += ["", "Full list: `data/diagnostics/exclusions.csv`.", ""]
    else:
        lines += ["None.", ""]
    if negative_summary is not None:
        lines += ["## Negative audit", "",
                  f"Method: {negative_summary.get('method')}. Human review performed: "
                  f"`{negative_summary.get('human_review')}`.", ""]
        lines += _table(["audited", "ambiguous", "clean", "ambiguous rate", "available", "requested"],
                        [[negative_summary.get("audited"), negative_summary.get("ambiguous"),
                          negative_summary.get("clean"), negative_summary.get("ambiguous_rate"),
                          negative_summary.get("negatives_available"),
                          negative_summary.get("requested")]])
        lines += ["", f"Signal counts: `{negative_summary.get('signal_counts')}`.", "",
                  "Negative rows flagged `ambiguous` must not be used for reward in the primary "
                  "analysis until a human resolves them (spec §6); the flag is available per row in "
                  "`data/diagnostics/negative_audit.csv`.", ""]
    if notes:
        lines += ["## Notes and honest limitations", ""]
        lines += [f"- {n}" for n in notes] + [""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def doc_sha_rows(docs_by_split: Mapping[str, Mapping[str, DocumentRecord]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, docs in docs_by_split.items():
        for doc_id in sorted(docs):
            doc = docs[doc_id]
            rows.append({"doc_id": doc.doc_id, "source_split": doc.source_split,
                         "strict_split": split, "title_group": doc.title_group,
                         "n_chars": len(doc.text)})
    return rows
