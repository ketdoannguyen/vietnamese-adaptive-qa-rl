"""Diagnostics: schema report, leakage report, negative audit.

Everything in this module is computed from the artifacts of the current run — no number
is typed by hand and nothing is reported that was not measured. The negative audit is an
*automated heuristic*; the CSV says so explicitly and never claims human review, and
ambiguity is counted on distractor documents only — never on the question's own gold doc
(see :func:`negative_signals`).
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
#: Description of the negative-ambiguity heuristic, used verbatim in the CSV, the report
#: and the machine-readable flag file so all three always tell the same story.
NEGATIVE_METHOD = (
    f"automated lexical heuristics on DISTRACTOR documents only (a corpus document other "
    f"than the question's own gold doc): plausible-answer token-sequence match in a "
    f"distractor, lexical_coverage>={LEXICAL_COVERAGE_THRESHOLD:.2f}, "
    f"reader_proxy>={READER_PROXY_THRESHOLD:.2f}; gold-doc signals are informational, "
    f"never ambiguity")


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
    """Return a plausible answer (from the dataset's own annotation) found in ``text``.

    Matching is token-sequence based, not a raw substring: a short plausible answer such as
    ``"ba"`` must occur as its own word, not inside another word (``"bàn"``). Measured on the
    pilot audit, substring matching produced 16 of 30 distractor hits that were fragment
    matches of this kind.
    """
    tokens = [t.casefold() for t in tokenize_words(text)]
    for candidate in plausible_texts:
        needle = [t.casefold() for t in tokenize_words(candidate)]
        if len(candidate.strip()) < 2 or not needle or len(needle) > len(tokens):
            continue
        if any(tokens[i:i + len(needle)] == needle for i in range(len(tokens) - len(needle) + 1)):
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


def negative_signals(
    question: str,
    plausible_texts: Sequence[str],
    doc_ids: Sequence[str],
    docs_by_id: Mapping[str, DocumentRecord],
    *,
    gold_doc_id: str | None,
) -> tuple[list[str], list[str]]:
    """Return ``(distractor_signals, gold_doc_signals)`` for one negative question.

    The question's own gold document is *expected* to overlap the question and to contain
    its plausible answers — that is what ``is_impossible`` means for this dataset — so
    signals on it are informational only and never count as ambiguity. The real risk is a
    *distractor* document (same episodic corpus, different doc) that could plausibly
    answer the question, which would invert the reward of `ABSTAIN`.
    """
    distractor: list[str] = []
    gold: list[str] = []
    for doc_id in doc_ids:
        doc = docs_by_id[doc_id]
        bucket = gold if doc_id == gold_doc_id else distractor
        coverage = lexical_coverage(question, doc.text)
        if coverage >= LEXICAL_COVERAGE_THRESHOLD:
            bucket.append(f"lexical_coverage={coverage:.2f}@doc={doc_id[:12]}")
        score = reader_proxy(question, doc.text)
        if score >= READER_PROXY_THRESHOLD:
            bucket.append(f"reader_proxy={score:.2f}@doc={doc_id[:12]}")
        hit = plausible_hit(plausible_texts, doc.text)
        if hit:
            bucket.append(f"plausible_answer_in_doc={hit!r}@doc={doc_id[:12]}")
    return distractor, gold


def audit_negatives(
    scenarios: Sequence[ScenarioRecord],
    rows_by_qid: Mapping[str, NormalizedRow],
    docs_by_id: Mapping[str, DocumentRecord],
    *,
    size: int,
    seed: object,
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, bool]]:
    """Audit negatives with documented distractor-only heuristics.

    Returns ``(csv_rows, summary, ambiguous_by_qid)``. The CSV covers a deterministic
    sample of at most ``size`` negatives; ``ambiguous_by_qid`` carries the verdict for
    *every* negative, because the flag is stamped into the label store and the scenarios
    for all rows — the sample and the full pass use the same heuristic, so a flag is
    never extrapolated from a subsample.

    Columns: ``qid, split, docs, verdict, ambiguous_flag, reviewer_note``. The note always
    states that the verdict is automated and was not verified by a human.
    """
    selected = select_negatives(scenarios, size, seed=seed)
    selected_qids = {s.qid for s in selected}
    rows_out: list[dict[str, str]] = []
    flags: dict[str, bool] = {}
    reasons: Counter[str] = Counter()
    gold_reasons: Counter[str] = Counter()
    per_split: dict[str, dict[str, Any]] = {}
    sample_ambiguous = 0
    for scenario in scenarios:
        if scenario.difficulty.get("answerable", True):
            continue
        row = rows_by_qid[scenario.qid]
        private = row.question.private
        distractor_signals, gold_signals = negative_signals(
            row.question.question, private.plausible_texts, scenario.corpus_doc_ids,
            docs_by_id, gold_doc_id=private.gold_doc_id)
        ambiguous = bool(distractor_signals)
        flags[scenario.qid] = ambiguous
        for signal in distractor_signals:
            reasons[signal.split("=", 1)[0]] += 1
        for signal in gold_signals:
            gold_reasons[signal.split("=", 1)[0]] += 1
        bucket = per_split.setdefault(scenario.split, {"negatives": 0, "flagged": 0,
                                                       "gold_doc_signal_negatives": 0})
        bucket["negatives"] += 1
        bucket["flagged"] += int(ambiguous)
        bucket["gold_doc_signal_negatives"] += int(bool(gold_signals))
        if scenario.qid not in selected_qids:
            continue
        sample_ambiguous += int(ambiguous)
        note = f"automated heuristic only; NO human review performed. ambiguity = {NEGATIVE_METHOD}."
        if gold_signals:
            note += (f" own gold doc carries {len(gold_signals)} signal(s) — expected for a"
                     f" negative question, informational only, never counted as ambiguity.")
        if distractor_signals:
            note += " signals: " + "; ".join(distractor_signals[:6])
            if len(distractor_signals) > 6:
                note += f" (+{len(distractor_signals) - 6} more)"
        else:
            note += " signals: none"
        rows_out.append({
            "qid": scenario.qid,
            "split": scenario.split,
            "docs": ";".join(scenario.corpus_doc_ids),
            "verdict": "ambiguous" if ambiguous else "clean",
            "ambiguous_flag": "true" if ambiguous else "false",
            "reviewer_note": note,
        })
    for split, bucket in per_split.items():
        bucket["clean"] = bucket["negatives"] - bucket["flagged"]
        bucket["flagged_rate"] = (round(bucket["flagged"] / bucket["negatives"], 6)
                                  if bucket["negatives"] else None)
    n_negatives = sum(b["negatives"] for b in per_split.values())
    n_flagged = sum(b["flagged"] for b in per_split.values())
    summary = {
        "requested": size,
        "negatives_available": n_negatives,
        "audited": len(rows_out),
        "ambiguous": sample_ambiguous,
        "clean": len(rows_out) - sample_ambiguous,
        "ambiguous_rate": round(sample_ambiguous / len(rows_out), 6) if rows_out else None,
        "flagged_negatives_all": n_flagged,
        "flagged_rate_all": round(n_flagged / n_negatives, 6) if n_negatives else None,
        "gold_doc_signal_negatives": sum(b["gold_doc_signal_negatives"]
                                         for b in per_split.values()),
        "signal_counts": dict(reasons),
        "gold_doc_signal_counts": dict(gold_reasons),
        "human_review": False,
        "method": NEGATIVE_METHOD,
        "per_split": per_split,
    }
    return rows_out, summary, flags


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
        lines += _table(["audited (sample)", "ambiguous", "clean", "ambiguous rate", "available",
                         "requested"],
                        [[negative_summary.get("audited"), negative_summary.get("ambiguous"),
                          negative_summary.get("clean"), negative_summary.get("ambiguous_rate"),
                          negative_summary.get("negatives_available"),
                          negative_summary.get("requested")]])
        per_split = negative_summary.get("per_split") or {}
        if per_split:
            lines += ["", "Full pass — every negative scenario was evaluated (the sample above is "
                      "the CSV subset):", ""]
            lines += _table(["split", "negatives", "flagged ambiguous", "clean", "flagged rate",
                             "with gold-doc signals (informational)"],
                            [[split, b["negatives"], b["flagged"], b["clean"], b["flagged_rate"],
                              b["gold_doc_signal_negatives"]]
                             for split, b in sorted(per_split.items())])
        lines += ["", f"Signal counts (distractor docs only): "
                  f"`{negative_summary.get('signal_counts')}`. Informational gold-doc signal "
                  f"counts (never ambiguity): `{negative_summary.get('gold_doc_signal_counts')}`.",
                  "",
                  "Policy (spec §6): a negative flagged `ambiguous_negative` stays in the dataset "
                  "but the primary RL trainer skips flagged TRAIN negatives from reward-bearing "
                  "updates; `val_strict`/`test_strict` keep their flags so clean vs ambiguous "
                  "negatives can be compared. The flag is automated-heuristic only — no human "
                  "review was performed — and is carried per row in "
                  "`questions_*.parquet.ambiguous_negative`, per scenario in "
                  "`difficulty.ambiguous_negative`, and listed in "
                  "`data/processed/ambiguous_negatives.json`. Flagged counts are reported here "
                  "and not hidden.", ""]
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
