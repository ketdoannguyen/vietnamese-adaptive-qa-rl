"""VixSTORY V2 prepare: privacy-split artifacts (preflight) + unified top-12 rankings.

Stage ``preflight`` writes the public question view and the private gold for train and
validation from the pinned HF snapshot (``ura-hcmut/VixSTORY`` at the pinned revision,
sha256-pinned files), plus a content-duplicate report:

* public reads project ``id``/``question``; private reads project ``id``/``context_ids``;
  the ``answer``/``contexts`` columns are never requested;
* non-pinned sources (csv, ``test.parquet``) are refused by filename before any read;
* counts, QID uniqueness, non-empty questions, non-empty/unique/in-corpus gold and
  train/validation disjointness fail loudly, and a failure leaves no partial artifacts;
* the private gold files are written with 0600 permissions;
* every artifact (and the manifest) is written once — a second run raises instead of
  overwriting.

Stage ``rankings`` reads the preflight manifest, re-verifies the complete artifact hash
chain, scores BM25 (shared ``BM25Index``) and the pinned dense encoder over the pinned
corpus, and writes the unified ``rankings_top12.json`` ``{qid: {bm25: [...], dense: [...]}}``
for train + validation. The private gold is never read by this stage.

CLI: ``.venv/bin/python scripts/vixstory_v2_prepare.py --run-dir DIR --stage preflight``.
"""

from __future__ import annotations

# ruff: noqa: E402 -- direct CLI invocation needs the repository path before local imports.
import argparse
import json
import os
import random
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:                     # direct ``python scripts/...`` invocation
    sys.path.insert(0, str(ROOT))

from src.pipeline.vixstory_worlds_v2 import (  # noqa: E402  (path fix must run first)
    DUP_REPORT_NAME,
    MANIFEST_NAME,
    RANKINGS_MANIFEST_NAME,
    RANKINGS_NAME,
    SPLITS,
    TOP_N,
    private_gold_name,
    public_questions_name,
    read_manifest,
)
from src.retrieval.bm25 import BM25Index
from src.retrieval.dense import rank_by_inner_product
from src.retrieval.dense_v2 import (
    DEFAULT_BATCH_SIZE,
    MAX_SEQ_LENGTH,
    MODEL_REPO,
    MODEL_REVISION,
    VietnameseEmbeddingEncoder,
    download_model,
)
from src.utils.hashing import sha256_file, sha256_json, sha256_text
from src.utils.io import read_json, write_json

DATASET_REPO = "ura-hcmut/VixSTORY"
DATASET_REVISION = "19613082c7ef128647d561472d1e150ca878fb22"
SOURCE_FILES = {"corpus": "corpus/corpus.parquet", "train": "data/train.parquet",
                "validation": "data/validation.parquet"}
PINNED_SHA256 = {
    "corpus": "5b1222dcdb0807544c9634b4f3816adc8eea8992ab0b83dffae08dc0db9bbf0a",
    "train": "7e0f41ce2896a66bc5ea4e4150303508fe8aa2787db6800c118e8830eb307dad",
    "validation": "db356de42e18b1b4f3a9150852702c406188bbf95a2b1524b52084f310ca6ab7",
}
PINNED_ROWS = {"corpus": 3462, "train": 2553, "validation": 246}
PINNED_INPUT_NAMES = {"corpus": "corpus.parquet", "train": "train.parquet",
                      "validation": "validation.parquet"}

MINHASH_PERMUTATIONS = 32
MINHASH_BANDS = 8
MINHASH_SEED = 42
NEAR_DUP_THRESHOLD = 0.9
_MOD = (1 << 61) - 1


# --- pinned sources ----------------------------------------------------------------------

def pinned_source(role: str, *, cache_dir: str | Path | None = None) -> Path:
    """Local path of one pinned dataset file (HF cache; downloads only if absent)."""
    if role not in SOURCE_FILES:
        raise ValueError(f"unknown pinned source {role!r}; expected one of {sorted(SOURCE_FILES)}")
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(DATASET_REPO, SOURCE_FILES[role], repo_type="dataset",
                                revision=DATASET_REVISION,
                                cache_dir=str(cache_dir) if cache_dir else None))


def _refuse_unpinned(path: str | Path) -> None:
    name = Path(path).name
    if name == "test.parquet":
        raise ValueError(f"refusing {name!r}: the official test split is never read")
    if Path(path).suffix != ".parquet":
        raise ValueError(f"refusing {name!r}: only pinned parquet sources are accepted")


def _assert_pinned_filename(path: str | Path, role: str) -> None:
    name = Path(path).name
    expected = PINNED_INPUT_NAMES[role]
    if name != expected:
        raise ValueError(
            f"refusing pinned source {name!r} for {role}: only {expected} from "
            f"{DATASET_REPO}@{DATASET_REVISION} is accepted "
            "(official test, csv and other sources are never used)")


# --- projected reads ---------------------------------------------------------------------

def read_public_rows(path: str | Path) -> list[dict]:
    """Rows with only ``id``/``question`` — the answer columns are never requested."""
    _refuse_unpinned(path)
    return pq.read_table(path, columns=["id", "question"]).to_pylist()


def read_private_rows(path: str | Path) -> list[dict]:
    """Rows with only ``id``/``context_ids`` — never answers, never question text."""
    _refuse_unpinned(path)
    rows = []
    for row in pq.read_table(path, columns=["id", "context_ids"]).to_pylist():
        rows.append({"id": row["id"], "context_ids": _context_ids(row["context_ids"])})
    return rows


def read_corpus_rows(path: str | Path) -> list[dict]:
    _refuse_unpinned(path)
    return pq.read_table(path, columns=["context_id", "text"]).to_pylist()


def _context_ids(raw) -> list:
    if not isinstance(raw, list):
        raise ValueError(f"context_ids must be a list, got {type(raw).__name__}")
    return raw


# --- validation --------------------------------------------------------------------------

def _check_count(role: str, actual: int, expected_rows: dict) -> None:
    if actual != expected_rows[role]:
        raise ValueError(f"{role}: {actual} rows, expected {expected_rows[role]} rows")


def _corpus_map(rows: list[dict]) -> dict:
    corpus = {}
    for row in rows:
        cid, text = row["context_id"], row["text"]
        if not isinstance(cid, str) or not cid:
            raise ValueError(f"corpus: invalid context id {cid!r}")
        if cid in corpus:
            raise ValueError(f"corpus: duplicate context id {cid!r}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"corpus: empty text for {cid!r}")
        corpus[cid] = text
    return corpus


def _validated_public(split: str, rows: list[dict]) -> dict:
    questions = {}
    for row in rows:
        qid, question = row["id"], row["question"]
        if not isinstance(qid, str) or not qid:
            raise ValueError(f"{split}: invalid question id {qid!r}")
        if qid in questions:
            raise ValueError(f"{split}: duplicate QID {qid!r}")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{split}: empty question for {qid!r}")
        questions[qid] = question
    return questions


def _validated_gold(split: str, rows: list[dict], public_qids: list[str],
                    corpus_ids: set[str]) -> dict:
    gold = {}
    for row in rows:
        qid, ids = row["id"], row["context_ids"]
        if qid in gold:
            raise ValueError(f"{split}: duplicate private QID {qid!r}")
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"{split}: empty gold_context_ids for {qid!r}")
        if not all(isinstance(cid, str) and cid for cid in ids):
            raise ValueError(f"{split}: invalid context id in gold for {qid!r}")
        if len(set(ids)) != len(ids):
            raise ValueError(f"{split}: duplicate context id in gold for {qid!r}")
        outside = [cid for cid in ids if cid not in corpus_ids]
        if outside:
            raise ValueError(
                f"{split}: context id {outside[0]!r} for {qid!r} is not in the corpus")
        gold[qid] = ids
    missing = [qid for qid in public_qids if qid not in gold]
    extra = [qid for qid in gold if qid not in set(public_qids)]
    if missing or extra:
        raise ValueError(f"{split}: private/public QID mismatch "
                         f"missing={missing[:5]} extra={extra[:5]}")
    return gold


# --- artifacts ---------------------------------------------------------------------------

def build_public_questions(split: str, qids: list[str], questions: dict) -> dict:
    qids = list(qids)
    return {"split": split, "n": len(qids), "qids": qids,
            "qids_sha256": sha256_json(qids),
            "questions": {qid: questions[qid] for qid in qids},
            "questions_sha256": sha256_json([questions[qid] for qid in qids])}


def build_private_gold(split: str, qids: list[str], gold: dict) -> dict:
    qids = list(qids)
    payload = {qid: sorted(gold[qid]) for qid in qids}
    return {"split": split, "n": len(qids), "qids": qids,
            "qids_sha256": sha256_json(qids), "gold": payload,
            "gold_sha256": sha256_json(payload)}


# --- duplicate report (report only — splits are never changed) ---------------------------

def _exact_key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _permutations() -> list[tuple[int, int]]:
    rng = random.Random(MINHASH_SEED)
    return [(rng.randrange(1, _MOD), rng.randrange(_MOD)) for _ in range(MINHASH_PERMUTATIONS)]


def _minhash(tokens: list[str], permutations: list[tuple[int, int]]) -> list[int]:
    signature: list[int] | None = None
    for token in tokens:
        h = int(sha256_text(token)[:16], 16)
        values = [(a * h + b) % _MOD for a, b in permutations]
        signature = values if signature is None else [min(x, y) for x, y in zip(signature, values)]
    return signature or []


def _jaccard(a: list[str], b: list[str]) -> float:
    ta, tb = set(a), set(b)
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / len(ta | tb)


def duplicate_report(questions_by_split: dict, *, threshold: float = NEAR_DUP_THRESHOLD) -> dict:
    """Exact (NFC+casefold) and near (MinHash candidate + exact Jaccard) duplicate report."""
    entries = []
    for split in SPLITS:
        for qid in sorted(questions_by_split[split]):
            text = questions_by_split[split][qid]
            entries.append((split, qid, _exact_key(text), _exact_key(text).split()))

    groups: dict[str, list[dict]] = {}
    for split, qid, key, _ in entries:
        groups.setdefault(key, []).append({"split": split, "qid": qid})
    exact_groups = [{"key": key, "members": members}
                    for key, members in sorted(groups.items()) if len(members) > 1]

    permutations = _permutations()
    width = MINHASH_PERMUTATIONS // MINHASH_BANDS
    signatures = [_minhash(tokens, permutations) for *_, tokens in entries]
    buckets: dict[tuple, list[int]] = {}
    for index, signature in enumerate(signatures):
        for band in range(MINHASH_BANDS):
            key = tuple(signature[band * width:(band + 1) * width])
            buckets.setdefault(key, []).append(index)
    candidates = set()
    for bucket in buckets.values():
        for position, first in enumerate(bucket):
            for second in bucket[position + 1:]:
                candidates.add((min(first, second), max(first, second)))

    split_order = {split: index for index, split in enumerate(SPLITS)}
    near_pairs = []
    for first, second in sorted(candidates):
        split_a, qid_a, key_a, tokens_a = entries[first]
        split_b, qid_b, key_b, tokens_b = entries[second]
        if key_a == key_b:                       # already reported as an exact duplicate
            continue
        if (split_order[split_b], qid_b) < (split_order[split_a], qid_a):
            first, second = second, first
            split_a, qid_a, split_b, qid_b = split_b, qid_b, split_a, qid_a
        score = _jaccard(tokens_a, tokens_b)
        if score >= threshold:
            near_pairs.append({
                "a": {"split": split_a, "qid": qid_a},
                "b": {"split": split_b, "qid": qid_b},
                "jaccard": round(score, 4),
                "scope": split_a if split_a == split_b else "train_vs_validation"})
    near_pairs.sort(key=lambda pair: (pair["a"]["split"], pair["a"]["qid"],
                                      pair["b"]["split"], pair["b"]["qid"]))
    return {"schema_version": 1, "splits_unchanged": True,
            "counts": {"exact_duplicate_groups": len(exact_groups),
                       "exact_duplicate_questions": sum(len(g["members"]) for g in exact_groups),
                       "near_duplicate_pairs": len(near_pairs),
                       "questions_total": len(entries)},
            "exact_duplicate_groups": exact_groups,
            "near_duplicate_pairs": near_pairs,
            "parameters": {"near_dup_threshold": threshold,
                           "minhash_permutations": MINHASH_PERMUTATIONS,
                           "minhash_bands": MINHASH_BANDS, "minhash_seed": MINHASH_SEED},
            "notes": "content duplicates are reported only; split membership is never changed"}


# --- shared plumbing ---------------------------------------------------------------------

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _file_entry(path: Path) -> dict:
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size}


def _write_new_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"{path} exists; refusing to overwrite")
    write_json(path, payload)


def _verify_source_digest(role: str, path: Path, expected_sha256: dict) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256[role]:
        raise ValueError(f"{role}: sha256 {actual} != pinned {expected_sha256[role]}")


# --- stage 1: preflight ------------------------------------------------------------------

def run_preflight(run_dir: str | Path, *, corpus_path: str | Path | None = None,
                  train_path: str | Path | None = None,
                  validation_path: str | Path | None = None,
                  expected_sha256: dict | None = None, expected_rows: dict | None = None,
                  seed: int = 42, cache_dir: str | Path | None = None,
                  near_dup_threshold: float = NEAR_DUP_THRESHOLD) -> dict:
    """Validate the pinned inputs and write public/private artifacts + manifest."""
    started_utc, started = _utcnow(), time.time()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_names = ([public_questions_name(s) for s in SPLITS]
                      + [private_gold_name(s) for s in SPLITS] + [DUP_REPORT_NAME])
    for name in artifact_names + [MANIFEST_NAME]:
        if (run_dir / name).exists():
            raise FileExistsError(
                f"{run_dir / name} exists; refusing to overwrite preflight artifacts")
    expected_sha256 = dict(PINNED_SHA256 if expected_sha256 is None else expected_sha256)
    expected_rows = dict(PINNED_ROWS if expected_rows is None else expected_rows)

    corpus_path = Path(corpus_path) if corpus_path else pinned_source("corpus", cache_dir=cache_dir)
    train_path = Path(train_path) if train_path else pinned_source("train", cache_dir=cache_dir)
    validation_path = (Path(validation_path) if validation_path
                       else pinned_source("validation", cache_dir=cache_dir))
    sources = {"corpus": corpus_path, "train": train_path, "validation": validation_path}
    for role, path in sources.items():
        _assert_pinned_filename(path, role)
    for role, path in sources.items():
        _verify_source_digest(role, path, expected_sha256)

    corpus_rows = read_corpus_rows(corpus_path)
    _check_count("corpus", len(corpus_rows), expected_rows)
    corpus = _corpus_map(corpus_rows)
    corpus_ids = set(corpus)

    questions_by_split, gold_by_split = {}, {}
    for split, path in (("train", train_path), ("validation", validation_path)):
        public_rows = read_public_rows(path)
        _check_count(split, len(public_rows), expected_rows)
        questions = _validated_public(split, public_rows)
        questions_by_split[split] = questions
        private_rows = read_private_rows(path)
        _check_count(split, len(private_rows), expected_rows)
        gold_by_split[split] = _validated_gold(split, private_rows, sorted(questions), corpus_ids)

    overlap = set(questions_by_split["train"]) & set(questions_by_split["validation"])
    if overlap:
        raise ValueError(f"train and validation QIDs are not disjoint: {sorted(overlap)[:5]}")

    report = duplicate_report(questions_by_split, threshold=near_dup_threshold)
    split_qids = {split: sorted(questions_by_split[split]) for split in SPLITS}

    written = {}
    for split in SPLITS:
        public_path = run_dir / public_questions_name(split)
        _write_new_json(public_path, build_public_questions(
            split, split_qids[split], questions_by_split[split]))
        written[public_path.name] = _file_entry(public_path)
        gold_path = run_dir / private_gold_name(split)
        _write_new_json(gold_path, build_private_gold(split, split_qids[split], gold_by_split[split]))
        os.chmod(gold_path, 0o600)
        written[gold_path.name] = _file_entry(gold_path)
    report_path = run_dir / DUP_REPORT_NAME
    _write_new_json(report_path, report)
    written[report_path.name] = _file_entry(report_path)

    manifest = {
        "schema_version": 1, "stage": "preflight", "run_id": run_dir.name,
        "inputs": {
            "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION,
                        "official_test_loaded": False},
            "corpus": {"path": str(corpus_path), "sha256": expected_sha256["corpus"],
                       "rows": len(corpus)},
            "train": {"path": str(train_path), "sha256": expected_sha256["train"],
                      "rows": len(questions_by_split["train"])},
            "validation": {"path": str(validation_path),
                           "sha256": expected_sha256["validation"],
                           "rows": len(questions_by_split["validation"])}},
        "splits": {split: {"n": len(split_qids[split]), "qids": split_qids[split],
                           "qids_sha256": sha256_json(split_qids[split])} for split in SPLITS},
        "disjointness": {"train_validation": 0, "all_disjoint": True},
        "duplicates": {"report": DUP_REPORT_NAME, "counts": report["counts"],
                       "splits_unchanged": True},
        "model_pin": {"repo": MODEL_REPO, "revision": MODEL_REVISION},
        "parameters": {"seed": seed, "top_n": TOP_N, "near_dup_threshold": near_dup_threshold,
                       "public_columns": ["id", "question"],
                       "private_columns": ["id", "context_ids"]},
        "outputs": written,
        "time": {"started_utc": started_utc, "finished_utc": _utcnow(),
                 "seconds": round(time.time() - started, 3)},
    }
    _write_new_json(run_dir / MANIFEST_NAME, manifest)
    return manifest


# --- stage 2: rankings -------------------------------------------------------------------

def run_rankings(run_dir: str | Path, *, encoder=None, batch_size: int = DEFAULT_BATCH_SIZE,
                 device: str | None = "cuda", top_n: int = TOP_N,
                 cache_dir: str | Path | None = None) -> dict:
    """BM25 + pinned dense top-``top_n`` for train and validation into one artifact."""
    started_utc, started = _utcnow(), time.time()
    run_dir = Path(run_dir)
    manifest_path = run_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{manifest_path} is missing; run the preflight stage first")
    for name in (RANKINGS_NAME, RANKINGS_MANIFEST_NAME):
        if (run_dir / name).exists():
            raise FileExistsError(
                f"{run_dir / name} exists; refusing to overwrite rankings artifacts")

    manifest = read_manifest(run_dir)
    if manifest.get("stage") != "preflight":
        raise ValueError(f"{manifest_path}: unexpected stage {manifest.get('stage')!r}")
    for name, entry in manifest["outputs"].items():
        path = run_dir / name
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"{name}: sha256 mismatch against the preflight manifest")

    corpus_entry = manifest["inputs"]["corpus"]
    corpus_path = Path(corpus_entry["path"])
    if sha256_file(corpus_path) != corpus_entry["sha256"]:
        raise ValueError("corpus: sha256 mismatch against the preflight manifest")
    corpus_rows = read_corpus_rows(corpus_path)
    if len(corpus_rows) != corpus_entry["rows"]:
        raise ValueError("corpus: row count mismatch against the preflight manifest")
    corpus = _corpus_map(corpus_rows)

    questions = {}
    for split in SPLITS:
        payload = read_json(run_dir / public_questions_name(split))
        for qid in payload["qids"]:
            questions[qid] = payload["questions"][qid]
    qids = sorted(questions)

    doc_ids = sorted(corpus)
    bm25_index = BM25Index.build({doc_id: corpus[doc_id] for doc_id in doc_ids})
    owns_encoder = encoder is None
    if owns_encoder:
        model_path, model_files = download_model(cache_dir=cache_dir)
        encoder = VietnameseEmbeddingEncoder(model_path, batch_size=batch_size, device=device)
        encoder.load()
        model_block = {"source": "pinned", "repo": MODEL_REPO, "revision": MODEL_REVISION,
                       "resolved_revision": MODEL_REVISION, "dimension": encoder.dimension(),
                       "max_seq_length": MAX_SEQ_LENGTH, "pooling": "cls",
                       "batch_size": encoder.effective_batch_size or batch_size,
                       "files": model_files}
    else:
        model_block = {"source": "injected", "repo": MODEL_REPO, "revision": MODEL_REVISION,
                       "resolved_revision": MODEL_REVISION,
                       "dimension": int(encoder.dimension())}

    doc_vectors = encoder.encode_passages([corpus[doc_id] for doc_id in doc_ids])
    query_vectors = encoder.encode_queries([questions[qid] for qid in qids])

    rankings = {}
    for index, qid in enumerate(qids):
        dense = [doc_id for doc_id, _ in
                 rank_by_inner_product(query_vectors[index], doc_vectors, doc_ids)[:top_n]]
        sparse = [doc_id for doc_id, _ in bm25_index.rank(questions[qid], top_n=top_n)]
        for tool, ranking in (("bm25", sparse), ("dense", dense)):
            if len(ranking) != top_n:
                raise ValueError(
                    f"rankings for {qid} {tool}: {len(ranking)} docs, expected {top_n}")
            if len(set(ranking)) != len(ranking):
                raise ValueError(f"rankings for {qid} {tool}: duplicate doc id")
            outside = [doc_id for doc_id in ranking if doc_id not in corpus]
            if outside:
                raise ValueError(
                    f"rankings for {qid} {tool}: {outside[0]!r} is not in the corpus")
        rankings[qid] = {"bm25": sparse, "dense": dense}

    rankings_path = run_dir / RANKINGS_NAME
    _write_new_json(rankings_path, rankings)
    provenance = {
        "schema_version": 1, "stage": "rankings", "run_id": run_dir.name,
        "preflight_manifest": {"path": MANIFEST_NAME, "sha256": sha256_file(manifest_path)},
        "inputs": {"corpus": {"path": str(corpus_path), "sha256": corpus_entry["sha256"],
                              "rows": len(doc_ids)},
                   "splits": {split: manifest["splits"][split] for split in SPLITS}},
        "model": model_block,
        "parameters": {"top_n": top_n, "splits": list(SPLITS),
                       "seed": manifest["parameters"]["seed"]},
        "outputs": {RANKINGS_NAME: _file_entry(rankings_path)},
        "time": {"started_utc": started_utc, "finished_utc": _utcnow(),
                 "seconds": round(time.time() - started, 3)},
    }
    _write_new_json(run_dir / RANKINGS_MANIFEST_NAME, provenance)
    if owns_encoder:
        encoder.unload()
    return {"stage": "rankings", "run_id": run_dir.name, "n_qids": len(qids), "top_n": top_n,
            "corpus_docs": len(doc_ids),
            "rankings_sha256": provenance["outputs"][RANKINGS_NAME]["sha256"],
            "time": provenance["time"]}


# --- CLI ---------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VixSTORY V2 prepare stages")
    parser.add_argument("--run-dir", required=True, help="experiment run directory")
    parser.add_argument("--stage", required=True, choices=("preflight", "rankings"))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args(argv)
    try:
        if args.stage == "preflight":
            manifest = run_preflight(Path(args.run_dir), seed=args.seed,
                                     cache_dir=args.cache_dir)
            summary = {"stage": "preflight", "run_id": manifest["run_id"],
                       "splits": {split: manifest["splits"][split]["n"] for split in SPLITS},
                       "outputs": sorted(manifest["outputs"])}
        else:
            summary = run_rankings(Path(args.run_dir), batch_size=args.batch_size,
                                   device=args.device, cache_dir=args.cache_dir)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
