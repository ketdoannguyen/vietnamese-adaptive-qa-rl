"""VixSTORY V2 worlds loader for retrieval-only episodes.

``load_worlds(run_dir, split)`` turns one prepared run directory into a list of
``VixStoryWorldV2`` (from ``src.rl.vixstory_env_v2``), in sorted QID order. Before any
world is built it re-verifies the full provenance chain:

* every artifact listed in ``manifest.json`` exists and matches its recorded sha256;
* the dataset sources are the pinned ``corpus.parquet``/``train.parquet``/
  ``validation.parquet`` files — ``test.parquet``, csv or any other source is refused;
* all three source files are re-hashed without decoding QA columns; only the corpus
  parquet is re-read (projected ``context_id``/``text``);
* the public question artifacts carry sorted, unique, non-empty questions and the split
  hashes match; train and validation QIDs are disjoint;
* the private gold is non-empty, duplicate-free and fully inside the pinned corpus;
* the unified ``rankings_top12.json`` covers every QID with ``TOP_N`` unique in-corpus doc
  ids per tool; the required ``rankings_manifest.json`` binds its artifact hash and model
  revision to the preflight manifest. An incomplete ranking stage is rejected.

The private gold files never leave this loader except as the ``gold_context_ids`` of the
worlds handed to the environment; the loader reads no answer column and no raw QA parquet
except the pinned corpus. The environment module is imported lazily so this loader (and its
tests) work before ``src/rl/vixstory_env_v2.py`` exists.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pyarrow.parquet as pq

from src.utils.hashing import sha256_file, sha256_json
from src.utils.io import read_json

MANIFEST_NAME = "manifest.json"
RANKINGS_NAME = "rankings_top12.json"
RANKINGS_MANIFEST_NAME = "rankings_manifest.json"
DUP_REPORT_NAME = "dup_report.json"
SPLITS = ("train", "validation")
TOP_N = 12

_ENV_MODULE = "src.rl.vixstory_env_v2"
_PINNED_INPUT_NAMES = {"corpus": "corpus.parquet", "train": "train.parquet",
                       "validation": "validation.parquet"}


def public_questions_name(split: str) -> str:
    return f"public_questions_{split}.json"


def private_gold_name(split: str) -> str:
    return f"private_gold_{split}.json"


def read_manifest(run_dir: str | Path) -> dict:
    return read_json(Path(run_dir) / MANIFEST_NAME)


def _world_class():
    try:
        module = importlib.import_module(_ENV_MODULE)
    except ImportError as exc:
        raise ImportError(
            f"{_ENV_MODULE} is required for V2 worlds but is unavailable; implement the "
            "V2 environment (VixStoryWorldV2 / VixStoryEnvV2) before loading worlds") from exc
    return module.VixStoryWorldV2


def _validate_public(payload: dict, split: str) -> None:
    qids = payload.get("qids")
    if not isinstance(qids, list) or not qids:
        raise ValueError(f"{split}: public question artifact has no QIDs")
    if qids != sorted(qids):
        raise ValueError(f"{split}: public QIDs are not sorted")
    if len(set(qids)) != len(qids):
        raise ValueError(f"{split}: duplicate QIDs in the public question artifact")
    if payload.get("n") != len(qids) or payload.get("qids_sha256") != sha256_json(qids):
        raise ValueError(f"{split}: public QID count/sha256 mismatch")
    questions = payload.get("questions", {})
    if set(questions) != set(qids):
        raise ValueError(f"{split}: public questions do not match the QID list")
    for qid in qids:
        question = questions[qid]
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{split}: empty question for {qid!r}")
    if payload.get("questions_sha256") != sha256_json([questions[qid] for qid in qids]):
        raise ValueError(f"{split}: public question sha256 mismatch")


def _validate_private(payload: dict, split: str, qids: list[str], corpus_ids: set[str]) -> None:
    if payload.get("qids") != qids or payload.get("n") != len(qids):
        raise ValueError(f"{split}: private gold QID list does not match the public artifact")
    gold = payload.get("gold", {})
    if set(gold) != set(qids):
        raise ValueError(f"{split}: private gold QIDs do not match the public artifact")
    for qid in qids:
        ids = gold[qid]
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"{split}: empty gold_context_ids for {qid!r}")
        if len(set(ids)) != len(ids):
            raise ValueError(f"{split}: duplicate context id in gold for {qid!r}")
        outside = [cid for cid in ids if cid not in corpus_ids]
        if outside:
            raise ValueError(
                f"{split}: context id {outside[0]!r} for {qid!r} is not in the corpus")
    if payload.get("gold_sha256") != sha256_json(gold):
        raise ValueError(f"{split}: private gold sha256 mismatch")


def load_worlds(run_dir: str | Path, split: str) -> list:
    """Build the ordered ``VixStoryWorldV2`` list for one split after full chain checks."""
    run_dir = Path(run_dir)
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    manifest = read_manifest(run_dir)
    if manifest.get("stage") != "preflight":
        raise ValueError(f"{run_dir / MANIFEST_NAME}: not a preflight manifest")
    if manifest["inputs"]["dataset"].get("official_test_loaded") is not False:
        raise ValueError("the manifest does not certify that the official test split is unused")
    for name, entry in manifest["outputs"].items():
        path = run_dir / name
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"{name}: sha256 mismatch against the preflight manifest")
    for role, expected in _PINNED_INPUT_NAMES.items():
        source = Path(manifest["inputs"][role]["path"])
        if source.name != expected:
            raise ValueError(
                f"refusing pinned source {source.name!r} for {role}: expected {expected} "
                "(official test, csv and other sources are never used)")
        if sha256_file(source) != manifest["inputs"][role]["sha256"]:
            raise ValueError(f"{role}: sha256 mismatch against the preflight manifest")

    corpus_path = Path(manifest["inputs"]["corpus"]["path"])
    table = pq.read_table(corpus_path, columns=["context_id", "text"])
    corpus = {}
    for row in table.to_pylist():
        cid = row["context_id"]
        if cid in corpus:
            raise ValueError(f"corpus: duplicate context id {cid!r}")
        corpus[cid] = row["text"]
    if len(corpus) != manifest["inputs"]["corpus"]["rows"]:
        raise ValueError("corpus: row count mismatch against the preflight manifest")
    corpus_ids = set(corpus)

    publics = {}
    for other in SPLITS:
        payload = read_json(run_dir / public_questions_name(other))
        _validate_public(payload, other)
        split_record = manifest["splits"][other]
        if any(payload[key] != split_record[key] for key in ("n", "qids", "qids_sha256")):
            raise ValueError(f"{other}: public split differs from the preflight manifest")
        if payload["n"] != manifest["inputs"][other]["rows"]:
            raise ValueError(f"{other}: public split count differs from the source manifest")
        publics[other] = payload
    overlap = set(publics["train"]["qids"]) & set(publics["validation"]["qids"])
    if overlap:
        raise ValueError(f"train and validation QIDs are not disjoint: {sorted(overlap)[:5]}")
    public = publics[split]
    qids = list(public["qids"])

    private = read_json(run_dir / private_gold_name(split))
    _validate_private(private, split, qids, corpus_ids)
    gold = private["gold"]

    rankings = read_json(run_dir / RANKINGS_NAME)
    for qid in qids:
        if qid not in rankings:
            raise ValueError(f"rankings missing qid {qid!r} for split {split}")
        entry = rankings[qid]
        for tool in ("bm25", "dense"):
            ranking = entry.get(tool) if isinstance(entry, dict) else None
            if not isinstance(ranking, list) or len(ranking) != TOP_N:
                raise ValueError(
                    f"rankings for {qid} {tool}: expected {TOP_N} doc ids, "
                    f"got {len(ranking) if isinstance(ranking, list) else ranking!r}")
            if len(set(ranking)) != len(ranking):
                raise ValueError(f"rankings for {qid} {tool}: duplicate doc id")
            outside = [doc_id for doc_id in ranking if doc_id not in corpus_ids]
            if outside:
                raise ValueError(
                    f"rankings for {qid} {tool}: {outside[0]!r} is not in the corpus")

    rankings_manifest_path = run_dir / RANKINGS_MANIFEST_NAME
    if not rankings_manifest_path.is_file():
        raise ValueError("required rankings manifest is missing; ranking stage is incomplete")
    rankings_manifest = read_json(rankings_manifest_path)
    recorded = rankings_manifest["outputs"][RANKINGS_NAME]["sha256"]
    if recorded != sha256_file(run_dir / RANKINGS_NAME):
        raise ValueError(f"{RANKINGS_NAME}: sha256 mismatch against the rankings manifest")
    if rankings_manifest["preflight_manifest"]["sha256"] != sha256_file(run_dir / MANIFEST_NAME):
        raise ValueError("preflight manifest sha256 mismatch against the rankings manifest")
    model = rankings_manifest["model"]
    if model["repo"] != manifest["model_pin"]["repo"]:
        raise ValueError(
            f"rankings model repo {model['repo']!r} != pinned {manifest['model_pin']['repo']!r}")
    if model["resolved_revision"] != manifest["model_pin"]["revision"]:
        raise ValueError(
            f"rankings model revision {model['resolved_revision']!r} != pinned "
            f"{manifest['model_pin']['revision']!r}")

    world_class = _world_class()
    worlds = []
    for qid in qids:
        worlds.append(world_class(qid=qid, question=public["questions"][qid], corpus=corpus,
                                  rankings={"bm25": list(rankings[qid]["bm25"]),
                                            "dense": list(rankings[qid]["dense"])},
                                  gold_context_ids=tuple(gold[qid])))
    return worlds
