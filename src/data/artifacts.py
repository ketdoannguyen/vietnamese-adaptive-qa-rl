"""Readers for the P1 artifacts — the only supported way to consume them downstream.

P1 (`src/pipeline/audit_split.py`) writes, per strict split:

* ``data/processed/docs_{split}.parquet``      — ``doc_id``, ``text`` (normalized), ``text_raw``
* ``data/processed/questions_{split}.parquet`` — the **label store**: gold answers,
  ``is_impossible``, ``gold_doc_id``. Never an observation, never a prompt.
* ``data/scenarios/scenarios_{short}.jsonl``   — ``ScenarioRecord.to_row()``

Split keys are ``train_strict`` / ``val_strict`` / ``test_strict``; the file names use that
key directly for docs/questions and its short form (``train``) for scenarios. Answer offsets
come in two flavours and they are **not** interchangeable: ``answer_starts_raw`` indexes
``text_raw`` while ``answer_starts_display`` indexes the normalized ``text``. The reader works
on ``text``, so it must use the display pair.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.data.schemas import ScenarioRecord
from src.utils.io import read_jsonl

STRICT_SPLITS = ("train_strict", "val_strict", "test_strict")


def short_split(split: str) -> str:
    """``train_strict`` -> ``train`` (the name used in the scenario file)."""
    if split not in STRICT_SPLITS:
        raise ValueError(f"unknown strict split {split!r}; known: {list(STRICT_SPLITS)}")
    return split.removesuffix("_strict")


def docs_path(processed_dir: str | Path, split: str) -> Path:
    return Path(processed_dir) / f"docs_{split}.parquet"


def questions_path(processed_dir: str | Path, split: str) -> Path:
    return Path(processed_dir) / f"questions_{split}.parquet"


def scenarios_path(scenarios_dir: str | Path, split: str) -> Path:
    return Path(scenarios_dir) / f"scenarios_{short_split(split)}.jsonl"


def require(path: Path, hint: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing — {hint}")
    return path


def read_doc_rows(processed_dir: str | Path, split: str) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    path = require(docs_path(processed_dir, split), "run the P1 download + audit_split steps")
    return pq.read_table(str(path)).to_pylist()


def read_docs(processed_dir: str | Path, split: str) -> dict[str, str]:
    """``doc_id -> normalized text`` for one split."""
    rows = read_doc_rows(processed_dir, split)
    docs = {row["doc_id"]: row["text"] for row in rows}
    if len(docs) != len(rows):
        raise ValueError(f"{docs_path(processed_dir, split)}: duplicate doc_id")
    return docs


def read_docs_all(processed_dir: str | Path, splits: Iterable[str] = STRICT_SPLITS
                  ) -> dict[str, str]:
    """Merge the splits' documents; a doc_id may belong to exactly one strict split."""
    docs: dict[str, str] = {}
    for split in splits:
        for doc_id, text in read_docs(processed_dir, split).items():
            if doc_id in docs:
                raise ValueError(f"doc_id {doc_id!r} appears in more than one strict split")
            docs[doc_id] = text
    return docs


def read_questions(processed_dir: str | Path, split: str) -> list[dict[str, Any]]:
    """Label-store rows for one split (see the module docstring: private data)."""
    import pyarrow.parquet as pq
    path = require(questions_path(processed_dir, split),
                   "run the P1 download + audit_split steps")
    return pq.read_table(str(path)).to_pylist()


def question_texts(processed_dir: str | Path, split: str) -> dict[str, str]:
    """``qid -> question`` — the only label-store column an observation may carry."""
    texts: dict[str, str] = {}
    for row in read_questions(processed_dir, split):
        qid = row["qid"]
        if qid in texts:
            raise ValueError(f"{questions_path(processed_dir, split)}: duplicate qid {qid!r}")
        texts[qid] = row["question"]
    return texts


def read_scenarios(scenarios_dir: str | Path, split: str) -> list[ScenarioRecord]:
    rows = read_jsonl(require(scenarios_path(scenarios_dir, split),
                              "run the P1 download + audit_split steps"))
    scenarios = []
    for row in rows:
        if row.get("split") != split:
            raise ValueError(f"{scenarios_path(scenarios_dir, split)}: row for split "
                             f"{row.get('split')!r} in the {split} file")
        scenarios.append(ScenarioRecord(
            scenario_id=row["scenario_id"], qid=row["qid"], split=row["split"],
            corpus_doc_ids=tuple(row["corpus_doc_ids"]), scenario_seed=int(row["scenario_seed"]),
            gold_doc_id=row.get("gold_doc_id"), reachable_bm25=row.get("reachable_bm25"),
            reachable_dense=row.get("reachable_dense"), difficulty=row.get("difficulty") or {}))
    return scenarios


def questions_by_qid(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        qid = row["qid"]
        if qid in index:
            raise ValueError(f"duplicate qid {qid!r}")
        index[qid] = row
    return index


def reader_rows(question_rows: Sequence[Mapping[str, Any]], *, limit: int | None = None,
                docs: Mapping[str, str] | None = None) -> tuple[list[dict[str, Any]],
                                                                list[dict[str, str]]]:
    """Project label-store rows into reader training/eval rows, dropping unusable ones.

    Returns ``(rows, dropped)``. A row is dropped — and reported, never silently — when its
    display answer offsets do not actually reproduce the display answer text in the
    normalized document, or when the gold document is missing from ``docs``. Rows keep the
    display pair (``answer_texts_display`` / ``answer_starts_display``) because the reader
    works on the normalized text.
    """
    ordered = sorted(question_rows, key=lambda row: row["qid"])
    if limit is not None:
        ordered = ordered[:int(limit)]
    rows: list[dict[str, Any]] = []
    dropped: list[dict[str, str]] = []
    for row in ordered:
        qid = row["qid"]
        doc_id = row.get("gold_doc_id")
        if doc_id is None or (docs is not None and doc_id not in docs):
            dropped.append({"qid": qid, "reason": "gold_doc_missing"})
            continue
        texts = [str(t) for t in (row.get("answer_texts_display") or [])]
        starts = [int(s) for s in (row.get("answer_starts_display") or [])]
        if len(texts) != len(starts):
            dropped.append({"qid": qid, "reason": "display_offsets_length_mismatch"})
            continue
        if docs is not None:
            text = docs[doc_id]
            bad = [i for i, (answer, start) in enumerate(zip(texts, starts))
                   if text[start:start + len(answer)] != answer]
            if bad:
                dropped.append({"qid": qid, "reason": f"display_offset_mismatch:{bad[0]}"})
                continue
        rows.append({"qid": qid, "question": row["question"], "gold_doc_id": doc_id,
                     "is_impossible": bool(row.get("is_impossible")),
                     "answer_texts": texts, "answer_starts": starts})
    return rows, dropped
