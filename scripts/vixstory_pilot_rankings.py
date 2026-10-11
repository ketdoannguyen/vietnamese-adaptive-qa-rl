"""Build the frozen VixSTORY pilot retrieval rankings (issue #30) before any GPU run.

Independently scored BM25 (pure corpus statistics, no pretrained weights) and
multilingual-E5-small dense rankings over the full pinned 3,462-passage VixSTORY corpus, for
the 300 ``pilot_300`` + 200 ``train_dev_200`` QIDs by default. The 246 ``validation_246``
QIDs require explicit ``--include-validation``; train/dev ranking does not read or hash their QA.
Retrieval never reads a reference answer or a gold passage id: BM25 sees
corpus text + question text, and E5 sees ``passage:``/``query:`` prefixed text only, with the
pinned model revision. The 246 validation questions are ranked from their question text alone.

Reuse decision. The earlier ``train500_rankings_top300.json`` is a bare ``{qid: {bm25, dense}}``
dict with no corpus or model provenance, and 434 of the 746 pilot QIDs (all 246 validation
QIDs plus 188 ``train_dev_200`` QIDs) are absent from it. It cannot be verified against the
pinned corpus or the E5 revision, and it cannot reduce the dominant cost — the one-time
encoding of all 3,462 passages — which the absent QIDs require regardless. Every QID is
therefore scored fresh here and the legacy artifact is left untouched.

Outputs (new files; an existing path is refused, never overwritten):

* ``pilot_rankings_top{N}.json`` -- bare ``{qid: {"bm25": [...], "dense": [...]}}`` mapping,
  directly consumable by ``scripts/vixstory_reader_eval.py --rankings``.
* ``pilot_rankings_manifest.json`` -- corpus/model fingerprints, per-split QID hashes and
  ``top_n`` for provenance.

No GPU call is made at import or in the tests; the model is only touched when ``run`` executes
the dense stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import pyarrow.parquet as pq

from src.retrieval.bm25 import BM25Index
from src.retrieval.dense import (
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    DenseBank,
    DenseEncoder,
    rank_by_inner_product,
)
from src.utils.hashing import sha256_file, sha256_json
from src.utils.io import atomic_write_json

REVISION = "19613082c7ef128647d561472d1e150ca878fb22"
REPO = "ura-hcmut/VixSTORY"
PILOT_DIR = Path("outputs/experiments/vixstory-pilot-300-2026-10-11")
MANIFEST_PATH = PILOT_DIR / "manifest.json"
SNAP = (Path.home() / ".cache/huggingface/hub/datasets--ura-hcmut--VixSTORY"
        / "snapshots" / REVISION)
CORPUS_PATH = SNAP / "corpus/corpus.parquet"
TRAIN_PATH = SNAP / "data/train.parquet"
VALIDATION_PATH = SNAP / "data/validation.parquet"

CORPUS_SHA256 = "5b1222dcdb0807544c9634b4f3816adc8eea8992ab0b83dffae08dc0db9bbf0a"
E5_MODEL = "intfloat/multilingual-e5-small"
E5_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"

SPLITS = ("pilot_300", "train_dev_200", "validation_246")

#: Retrieval reads only these two columns; the label columns below are never looked up.
QID_FIELD = "id"
QUESTION_FIELD = "question"
LABEL_FIELDS = ("answer", "context_ids", "contexts")

#: A ranking has to cover the deepest reachable state the repo can build: bootstrap passages
#: plus ``max_optional_search`` searches of ``passage_per_search_action`` passages each. The
#: largest shipped config is 3 bootstraps + 3x3 searches, i.e. 12 documents from one tool.
MAX_BOOTSTRAP_PASSAGES = 3
MAX_PASSAGES_PER_SEARCH = 3
MAX_OPTIONAL_SEARCH = 3
MIN_TOP_N = MAX_BOOTSTRAP_PASSAGES + MAX_PASSAGES_PER_SEARCH * MAX_OPTIONAL_SEARCH
DEFAULT_TOP_N = MIN_TOP_N

LEGACY_RANKINGS = (Path("outputs/experiments/2026-10-11-vixstory-diagnosis")
                   / "train500_rankings_top300.json")
LEGACY_REUSE_REASON = (
    "not reused: the legacy artifact is a bare {qid: {bm25, dense}} dict with no corpus or "
    "model provenance, it omits all 246 validation QIDs and 188 train_dev_200 QIDs, and it "
    "cannot avoid the one-time 3,462-passage encoding the missing QIDs require. Every pilot "
    "QID is scored fresh against the pinned corpus and the pinned E5 revision."
)

PROVENANCE_NAME = "pilot_rankings_manifest.json"


def rankings_name(top_n: int) -> str:
    return f"pilot_rankings_top{top_n}.json"


def load_manifest(path: str | Path = MANIFEST_PATH) -> dict:
    """Load the pilot manifest and re-check every split's sorted QIDs against its hash."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    for name in manifest["splits"]:
        split = manifest["splits"][name]
        qids = split["qids"]
        if split["n"] != len(qids) or qids != sorted(qids):
            raise ValueError(f"manifest split {name!r} is not a sorted list of {split['n']} QIDs")
        if split["qids_sha256"] != sha256_json(qids):
            raise ValueError(f"manifest split {name!r} QID hash does not match its QIDs")
    return manifest


def select_questions(rows: Sequence[Mapping], qids: Sequence[str], *,
                     qid_field: str = QID_FIELD, question_field: str = QUESTION_FIELD) -> dict[str, str]:
    """``{qid: question}`` for ``qids``, reading only the id and question columns.

    The reference answer and the gold passage ids are never accessed, so a ranking built from
    this mapping cannot depend on a label. Any manifest QID missing from ``rows`` is an error
    rather than a silent omission.
    """
    by_id = {}
    for row in rows:
        if qid_field in row:
            by_id[row[qid_field]] = row[question_field]
    missing = [qid for qid in qids if qid not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} manifest QIDs absent from the source rows: {missing[:5]}")
    return {qid: by_id[qid] for qid in qids}


def build_rankings(corpus: Mapping[str, str], questions: Mapping[str, str], *,
                   top_n: int = DEFAULT_TOP_N, encoder) -> dict[str, dict[str, list[str]]]:
    """Independently scored BM25 and dense rankings over the whole corpus.

    BM25 is rebuilt from corpus statistics alone; dense uses the supplied E5 ``encoder`` with
    its asymmetric prefixes, encoding every query in one batch. Corpus documents are visited in
    sorted id order and both rankers break score ties by document id, so the output is
    deterministic. Only corpus text and question text enter either ranker.
    """
    if top_n < MIN_TOP_N:
        raise ValueError(f"top_n {top_n} < {MIN_TOP_N}: too short for the deepest reachable state")
    doc_ids = sorted(corpus)
    if len(doc_ids) < top_n:
        raise ValueError(f"corpus has {len(doc_ids)} passages, fewer than top_n={top_n}")
    texts = {doc_id: corpus[doc_id] for doc_id in doc_ids}
    bm25 = BM25Index.build(texts)
    dense = DenseBank.build(texts, encoder)
    query_ids = sorted(questions)
    query_vecs = encoder.encode_queries([questions[qid] for qid in query_ids])
    rankings: dict[str, dict[str, list[str]]] = {}
    for qid, query_vec in zip(query_ids, query_vecs):
        rankings[qid] = {
            "bm25": [doc_id for doc_id, _ in bm25.rank(questions[qid], top_n=top_n)],
            "dense": [doc_id for doc_id, _ in
                      rank_by_inner_product(query_vec, dense.vectors, dense.doc_ids)[:top_n]],
        }
    return rankings


def build_provenance(*, manifest: Mapping, manifest_path: str | Path, corpus_path: str | Path,
                     train_path: str | Path, validation_path: str | Path, corpus_ids: Sequence[str],
                     top_n: int, model_name: str, model_revision: str,
                     rankings: Mapping, include_validation: bool = False) -> dict:
    """Provenance envelope binding the rankings to their corpus, model and manifest."""
    return {
        "artifact": "vixstory_pilot_rankings",
        "schema_version": 1,
        "issue": 30,
        "run_id": manifest["run_id"],
        "purpose": "Independently scored BM25 and multilingual-E5-small rankings over the full "
                   "pinned VixSTORY corpus for selected pilot QIDs; retrieval reads no answer "
                   "or gold passage.",
        "source": {
            "repo_id": REPO,
            "resolved_revision": manifest["source"]["resolved_revision"],
            "corpus_sha256": sha256_file(corpus_path),
            "train_sha256": sha256_file(train_path),
            **({"validation_sha256": sha256_file(validation_path)} if include_validation else {}),
        },
        "corpus": {"n_docs": len(corpus_ids), "doc_ids_sha256": sha256_json(sorted(corpus_ids))},
        "retrieval": {
            "bm25": {"impl": "src.retrieval.bm25.BM25Index", "k1": 1.5, "b": 0.75},
            "dense": {"model": model_name, "revision": model_revision,
                      "query_prefix": QUERY_PREFIX, "passage_prefix": PASSAGE_PREFIX,
                      "normalized": True, "impl": "src.retrieval.dense.rank_by_inner_product"},
        },
        "top_n": top_n,
        "min_top_n": MIN_TOP_N,
        "n_qids": len(rankings),
        "splits": {name: {"n": manifest["splits"][name]["n"],
                          "qids_sha256": manifest["splits"][name]["qids_sha256"]}
                   for name in SPLITS if include_validation or name != "validation_246"},
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "no_gold_leakage": "rankings derive from corpus text and question text only; the answer, "
                           "context_ids and contexts columns are not loaded from parquet at all "
                           "for any selected split",
        "legacy_train500_reuse": {"path": str(LEGACY_RANKINGS), "reused": False,
                                  "reason": LEGACY_REUSE_REASON},
        "rankings_sha256": sha256_json(rankings),
    }


def write_outputs(out_dir: str | Path, rankings: Mapping, provenance: Mapping, *, top_n: int
                  ) -> tuple[Path, Path]:
    """Write both artifacts, refusing to overwrite an existing one."""
    out_dir = Path(out_dir)
    rankings_path = out_dir / rankings_name(top_n)
    provenance_path = out_dir / PROVENANCE_NAME
    for path in (rankings_path, provenance_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing pilot ranking artifact {path}")
    atomic_write_json(rankings_path, rankings)
    atomic_write_json(provenance_path, provenance)
    return rankings_path, provenance_path


def run(*, manifest_path: str | Path = MANIFEST_PATH, corpus_path: str | Path = CORPUS_PATH,
        train_path: str | Path = TRAIN_PATH, validation_path: str | Path = VALIDATION_PATH,
        out_dir: str | Path = PILOT_DIR, top_n: int = DEFAULT_TOP_N, encoder=None,
        model_name: str = E5_MODEL, model_revision: str = E5_REVISION,
        expected_corpus_sha256: str | None = CORPUS_SHA256,
        include_validation: bool = False) -> dict:
    """Score train/dev QIDs by default; validation only with explicit opt-in.

    ``encoder`` is injectable so the CPU tests can supply a deterministic stand-in; when it is
    omitted a real :class:`DenseEncoder` is built with the pinned revision and unloaded again.
    """
    manifest = load_manifest(manifest_path)
    if include_validation and "validation_246" not in manifest["splits"]:
        raise ValueError("validation rankings require a manifest with validation_246 QIDs")
    digest = sha256_file(corpus_path)
    if expected_corpus_sha256 is not None and digest != expected_corpus_sha256:
        raise ValueError(f"corpus sha256 {digest} != pinned {expected_corpus_sha256}")

    # Column projection: the answer / context_ids columns are never even loaded, so retrieval
    # cannot read a label even by accident.
    corpus: dict[str, str] = {}
    for row in pq.read_table(corpus_path, columns=["context_id", "text"]).to_pylist():
        context_id = row["context_id"]
        if context_id in corpus:
            raise ValueError(f"duplicate corpus context_id {context_id}")
        corpus[context_id] = row["text"]

    train_rows = pq.read_table(train_path, columns=[QID_FIELD, QUESTION_FIELD]).to_pylist()
    pilot_qids = manifest["splits"]["pilot_300"]["qids"]
    dev_qids = manifest["splits"]["train_dev_200"]["qids"]
    questions = select_questions(train_rows, pilot_qids + dev_qids)
    expected = set(pilot_qids) | set(dev_qids)
    if include_validation:
        val_qids = manifest["splits"]["validation_246"]["qids"]
        validation_rows = pq.read_table(validation_path,
                                        columns=[QID_FIELD, QUESTION_FIELD]).to_pylist()
        questions.update(select_questions(validation_rows, val_qids))
        expected.update(val_qids)
    if set(questions) != expected:
        raise AssertionError("question coverage drifted from the manifest")

    owns = encoder is None
    if owns:
        encoder = DenseEncoder(model_name=model_name, revision=model_revision,
                               batch_size=32, max_length=512)
    try:
        rankings = build_rankings(corpus, questions, top_n=top_n, encoder=encoder)
    finally:
        if owns:
            encoder.unload()

    provenance = build_provenance(
        manifest=manifest, manifest_path=manifest_path, corpus_path=corpus_path,
        train_path=train_path, validation_path=validation_path, corpus_ids=list(corpus),
        top_n=top_n, model_name=model_name, model_revision=model_revision, rankings=rankings,
        include_validation=include_validation)
    rankings_path, provenance_path = write_outputs(out_dir, rankings, provenance, top_n=top_n)
    return {"rankings_path": str(rankings_path), "provenance_path": str(provenance_path),
            "n_qids": len(rankings), "top_n": top_n, "corpus_docs": len(corpus)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--validation", type=Path, default=VALIDATION_PATH)
    parser.add_argument("--include-validation", action="store_true",
                        help="explicitly rank official-validation questions")
    parser.add_argument("--out-dir", type=Path, default=PILOT_DIR)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                        help=f"ranking length; must be >= {MIN_TOP_N}")
    args = parser.parse_args()
    result = run(manifest_path=args.manifest, corpus_path=args.corpus, train_path=args.train,
                 validation_path=args.validation, out_dir=args.out_dir, top_n=args.top_n,
                 include_validation=args.include_validation)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
