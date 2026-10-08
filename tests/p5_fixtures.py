"""Synthetic P1/P2/P3 artifacts for the P5 tests (no network, no GPU, no real checkpoints).

Writes the on-disk layout the real pipeline produces inside ``tmp_path``, so
:func:`src.pipeline.worlds.build_worlds` and the P5 CLIs run end to end over it:

* ``data/processed/docs_{split}.parquet`` / ``questions_{split}.parquet``
* ``data/scenarios/scenarios_{short}.jsonl``
* ``data/cache/retrieval/{split}_rankings_{version}.jsonl`` (the P2 cache)
* the P3 reader state cache parquet + meta, built with
  :func:`src.qa.state_cache.build_state_cache` and a deterministic stub inference

Each question's corpus holds three documents: its own (the stub reader answers only when it
is visible) and two distractors. Even-indexed questions bootstrap their own document;
odd-indexed ones bootstrap a distractor, so the answer needs a search — and both tools expose
it, so two different histories reach it. ``test_strict`` gets an empty docs parquet because
``artifacts.read_docs_all`` reads every strict split.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.data import artifacts
from src.pipeline import build_retrieval, train_reader
from src.qa import state_cache
from src.qa.postprocess import NO_SPAN_SCORE, BestSpan
from src.utils.config import load_config
from src.utils.hashing import sha256_text
from src.utils.io import atomic_write_json, write_jsonl, write_parquet

READER_MODEL_HASH = "r" * 64
TOKENIZER_SHA = "t" * 64
INFERENCE_CONFIG_SHA = "i" * 64
NULL_SCORE_DIFF_THRESHOLD = 0.0

_DOC_TEXT = ("Đoạn văn {} kể về một sự kiện lịch sử diễn ra vào năm {} "
             "tại một vùng đất cụ thể của Việt Nam.")
_YEARS = (1010, 1945, 1976, 1288)


def p5_config(tmp_path: Path) -> dict:
    """The real config with every path under ``tmp_path`` and the smoke profile uncapped."""
    conf = load_config(None)
    keys = ("data_raw", "data_processed", "data_scenarios", "data_indexes", "data_cache",
            "data_diagnostics", "outputs", "models", "runs", "predictions", "manifests",
            "diagnostics", "reports", "plots", "tables")
    conf["paths"] = {**conf["paths"], **{key: str(tmp_path / key) for key in keys}}
    conf["data"]["profile"] = "smoke"
    conf["data"]["caps"]["smoke"] = None
    return conf


def doc_ids(split: str, index: int) -> list[str]:
    """The corpus of question ``index``: own document first, then two distractors.

    Ids stay under ``state_text.DOC_ID_CHARS`` so the rendered state keeps them distinct.
    """
    short = split.removesuffix("_strict")[:3]
    return [f"d:{short}:{index:03d}:{slot}" for slot in range(3)]


def ranking(split: str, index: int) -> dict[str, list[str]]:
    """BM25/Dense rankings over the question's corpus (own document at a known position)."""
    own, other_a, other_b = doc_ids(split, index)
    if index % 2 == 0:
        return {"bm25": [own, other_a, other_b], "dense": [other_b, other_a, own]}
    return {"bm25": [other_a, own, other_b], "dense": [own, other_b, other_a]}


def _split_rows(split: str, n: int) -> tuple[list[dict], list[dict], list[dict]]:
    docs, questions, scenarios = [], [], []
    for index in range(n):
        own, other_a, other_b = doc_ids(split, index)
        own_year = _YEARS[index % len(_YEARS)]
        own_text = _DOC_TEXT.format(own, own_year)
        texts = [own_text]
        for slot, doc_id in enumerate((other_a, other_b), start=1):
            texts.append(_DOC_TEXT.format(doc_id, _YEARS[(index + 3 * slot) % len(_YEARS)]))
        for doc_id, text in zip((own, other_a, other_b), texts):
            docs.append({"doc_id": doc_id, "text": text, "text_raw": text})
        qid = f"{split}:q{index:03d}"
        questions.append({
            "qid": qid, "question": f"Đoạn văn {own} nhắc tới sự kiện năm nào?",
            "gold_doc_id": own, "is_impossible": False,
            "answer_texts": [str(own_year)], "answer_texts_display": [str(own_year)],
            "answer_starts_display": [own_text.index(str(own_year))]})
        scenarios.append({
            "scenario_id": f"sc:{split}:{qid}:v1", "qid": qid, "split": split,
            "corpus_doc_ids": [own, other_a, other_b], "scenario_seed": 42,
            "gold_doc_id": own, "reachable_bm25": None, "reachable_dense": None,
            "difficulty": {}})
    return docs, questions, scenarios


class _StubInference:
    """Deterministic reader: answers with the own document's year when it is visible."""

    def predict(self, pairs) -> dict[tuple[str, str], BestSpan]:
        out = {}
        for pair in pairs:
            parts = pair.doc_id.split(":")
            same_doc = parts[-1] == "0" and parts[-2] == pair.qid.rsplit(":q", 1)[-1]
            number = re.search(r"\d{4}", pair.context) if same_doc else None
            if number:
                out[(pair.qid, pair.doc_id)] = BestSpan(
                    qid=pair.qid, doc_id=pair.doc_id, answer_text=number.group(),
                    start_char=number.start(), end_char=number.end(),
                    non_null_score=5.0, null_score=1.0)
            else:
                out[(pair.qid, pair.doc_id)] = BestSpan(
                    qid=pair.qid, doc_id=pair.doc_id, answer_text="", start_char=0, end_char=0,
                    non_null_score=NO_SPAN_SCORE, null_score=1.0)
        return out


def state_cache_paths(cfg: Mapping[str, Any], split: str) -> tuple[Path, Path]:
    """The P3 cache layout (identical to ``cache_reader.state_cache_paths``)."""
    base = train_reader.reader_dir(cfg) / f"reader_qa_state_cache_{split}"
    return Path(f"{base}.parquet"), Path(f"{base}.json")


def write_split(cfg: Mapping[str, Any], split: str, n: int) -> dict[str, Any]:
    """Write P1/P2/P3 artifacts for ``n`` synthetic questions; returns the state cache meta."""
    processed = Path(cfg["paths"]["data_processed"])
    scenarios_dir = Path(cfg["paths"]["data_scenarios"])
    docs, questions, scenarios = _split_rows(split, n)
    write_parquet(artifacts.docs_path(processed, split), docs)
    write_parquet(artifacts.questions_path(processed, split), questions)
    write_jsonl(artifacts.scenarios_path(scenarios_dir, split), scenarios)

    rankings = {row["qid"]: ranking(split, index) for index, row in enumerate(questions)}
    write_jsonl(build_retrieval.rankings_path(cfg, split),
                [{"scenario_id": f"sc:{split}:{qid}:v1", "qid": qid, "rankings": value,
                  "scores": {}} for qid, value in sorted(rankings.items())])

    rows, stats = state_cache.build_state_cache(
        scenarios=artifacts.read_scenarios(scenarios_dir, split),
        questions=artifacts.question_texts(processed, split),
        docs={row["doc_id"]: row["text"] for row in docs},
        rankings=rankings, inference=_StubInference(),
        threshold=NULL_SCORE_DIFF_THRESHOLD, reader_model_hash=READER_MODEL_HASH,
        tokenizer_sha=TOKENIZER_SHA, inference_config_sha=INFERENCE_CONFIG_SHA,
        retrieval=cfg["retrieval"])
    parquet_path, meta_path = state_cache_paths(cfg, split)
    write_parquet(parquet_path, rows)
    meta = {
        "split": split, "schema_version": state_cache.SCHEMA_VERSION, "n_states": len(rows),
        "reader_model_hash": READER_MODEL_HASH, "tokenizer_sha": TOKENIZER_SHA,
        "inference_config_sha": INFERENCE_CONFIG_SHA,
        "null_score_diff_threshold": NULL_SCORE_DIFF_THRESHOLD,
        "retrieval_budget": {
            "bootstrap_tool": cfg["retrieval"]["initial_bootstrap"],
            "bootstrap_passages": int(cfg["retrieval"]["bootstrap_passages"]),
            "passage_per_search_action": int(cfg["retrieval"]["passage_per_search_action"]),
            "optional_searches": int(cfg["retrieval"]["optional_searches"])},
        "stats": stats,
    }
    atomic_write_json(meta_path, meta)
    return meta


def write_all(cfg: Mapping[str, Any], *, n_train: int = 3, n_val: int = 2) -> dict[str, dict]:
    """``train_strict`` / ``val_strict`` fixtures plus the empty ``test_strict`` docs file."""
    write_parquet(artifacts.docs_path(cfg["paths"]["data_processed"], "test_strict"), [])
    return {"train_strict": write_split(cfg, "train_strict", n_train),
            "val_strict": write_split(cfg, "val_strict", n_val)}


class StubAdapter:
    """Deterministic :class:`~src.decision.laya_adapter.LayaAdapter` stand-in.

    Probabilities are a fixed function of the state text (its digest), so two adapters with
    the same ``model_hash`` agree row for row and a different hash means different rows.
    """

    def __init__(self, *, model_hash: str = "a" * 64, n_actions: int = 4) -> None:
        self._model_hash = model_hash
        self.n_actions = n_actions
        self.calls = 0
        self.texts: list[str] = []
        self.closed = False

    def model_hash(self) -> str:
        return self._model_hash

    def predict_probs(self, states) -> np.ndarray:
        self.calls += 1
        out = np.zeros((len(states), self.n_actions), dtype=np.float64)
        for row, text in enumerate(states):
            self.texts.append(str(text))
            seed = int(sha256_text(str(text))[:16], 16)
            values = np.array([((seed >> (3 * k)) & 0x7) + 1 for k in range(self.n_actions)],
                              dtype=np.float64)
            out[row] = values / values.sum()
        return out

    def close(self) -> None:
        self.closed = True
