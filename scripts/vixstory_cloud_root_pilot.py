"""Cloud reader pilot: root-only BM25 on the *fixed* dev40 of the pinned VixSTORY train split.

One bounded experiment, official-train only:

* ``dev40`` — the fixed 40-question subset of the manifest's ``train_dev_200`` split, selected
  deterministically by the reader's own ``select_dev_sample(.., 40, seed=42)`` rule. That rule
  is copied verbatim from the local reader-training module so the id set matches every earlier
  dev40 probe *without importing* that module (this pilot never loads a local QA reader).
* ``root`` — for each question the reader is shown a *single* passage: the BM25 top-1
  retrieval result. No gold passage, no reference answer and no validation/test row ever
  enters the prompt; only ``corpus.parquet`` and ``data/train.parquet`` are opened.

Generation runs through :class:`scripts.vixstory_reader_eval.CloudReader` with the shared
``reader_prompt`` and an identity bound to the corpus and rankings it read. The returned text is
scored locally (token-F1 against the gold answer), reported together with citation *ID*
validity and the citation-gated success rate.

The endpoint is required: ``AIBOX_API_KEY`` is checked before any QA row is read or any request
is attempted. The whole run — a synthetic connectivity smoke followed by the 40 questions —
shares one hard request cap (default 50) enforced by ``CloudReader``. The report is written to a
dedicated namespace and the run refuses to clobber an existing report.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Mapping, Sequence

import pyarrow.parquet as pq

from scripts.vixstory_reader_eval import (
    MAX_NEW_TOKENS,
    MAX_REQUESTS,
    MODEL,
    MODEL_REVISION,
    PROMPT_VERSION,
    CloudReader,
    answer_reward,
    build_identity,
    parse_reader_output,
    reader_prompt,
    refuse_overwrite,
)
from src.qa.metrics import f1_score
from src.utils.io import atomic_write_json

# --- pinned dataset (same revision as vixstory_reader_eval / pilot preflight) ---------------
DATASET_REVISION = "19613082c7ef128647d561472d1e150ca878fb22"
SNAP = (Path.home() / ".cache/huggingface/hub/datasets--ura-hcmut--VixSTORY"
        / "snapshots" / DATASET_REVISION)
CORPUS_PATH = SNAP / "corpus/corpus.parquet"
TRAIN_PATH = SNAP / "data/train.parquet"
MANIFEST_PATH = Path("outputs/experiments/vixstory-pilot-300-2026-10-11/manifest.json")
#: Covers all 746 pilot/dev/validation QIDs, so every dev40 question has a BM25 ranking.
RANKINGS_PATH = (Path("outputs/experiments/vixstory-pilot-300-2026-10-11")
                 / "pilot_rankings_top12.json")

OUT = Path("outputs/experiments/vixstory-cloud-root-dev40-2026-10-11")
REPORT_NAME = "report.json"
CACHE_NAME = "reader-cache.json"

DEV_SPLIT_ALIASES = ("dev", "train_dev", "train_dev_200")
DEV_N = 40
DEV_SEED = 42

#: One synthetic connectivity prompt. It shares the reader's cache and request budget but is
#: never scored; it only proves the endpoint and cache path work before the scored batch.
SMOKE_QUESTION = "Mã kiểm tra là gì?"
SMOKE_PASSAGE = {"context_id": "SMOKE", "text": "Mã kiểm tra là Xanh."}


# --------------------------------------------------------------- selection (copied, not imported)

def select_dev_sample(rows: Sequence[Mapping], n: int | None, *, seed: int) -> list[Mapping]:
    """The deterministic dev-subset rule copied from the local reader-training module.

    Same rows and seed give the same id set, so this pilot reproduces the historical dev40
    selection without importing the (GPU) reader-training module.
    """
    ordered = sorted(rows, key=lambda row: row["id"])
    if n is None:
        return ordered
    if n > len(ordered) or n < 1:
        raise ValueError(f"dev sample size {n} outside [1, {len(ordered)}]")
    return [ordered[i] for i in sorted(random.Random(seed).sample(range(len(ordered)), n))]


def split_qids(manifest: Mapping, aliases: Sequence[str]) -> list[str]:
    """Resolve one split's QID list from a manifest by any of its accepted record names."""
    source = manifest.get("splits")
    if not isinstance(source, Mapping):
        raise ValueError("manifest carries no 'splits' mapping")
    record = next((source[name] for name in aliases if name in source), None)
    if record is None:
        raise ValueError(f"manifest has no split under any of {tuple(aliases)}")
    qids = record.get("qids") if isinstance(record, Mapping) else record
    if not isinstance(qids, (list, tuple)) or not all(isinstance(q, str) for q in qids):
        raise ValueError(f"split {tuple(aliases)} must carry a list of string QIDs")
    return list(qids)


# -------------------------------------------------------------------------------- data loading

def require_api_key() -> None:
    """Fail before any QA read or network call when the endpoint credential is absent."""
    if not os.environ.get("AIBOX_API_KEY"):
        raise RuntimeError("AIBOX_API_KEY is not set; refusing to start the cloud root dev40 pilot")


def load_dev40(manifest_path: Path, corpus_path: Path, train_path: Path, *,
               n: int, seed: int) -> tuple[dict[str, Mapping], list[Mapping]]:
    """Load the corpus and the fixed dev40 rows from *official train only*.

    Refuses when the manifest does not certify the official test split as unread, when a dev
    QID is absent from official train, or when a required local artifact is missing. The
    validation table is never opened.
    """
    for path in (Path(corpus_path), Path(train_path), Path(manifest_path)):
        if not path.exists():
            raise FileNotFoundError(f"required local artifact is missing: {path}")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("source", {}).get("official_test_loaded") is not False:
        raise ValueError("manifest does not certify that the official test split was unread")
    docs = {row["context_id"]: row for row in pq.read_table(corpus_path).to_pylist()}
    train = {row["id"]: row for row in pq.read_table(train_path).to_pylist()}
    dev_qids = split_qids(manifest, DEV_SPLIT_ALIASES)
    missing = [qid for qid in dev_qids if qid not in train]
    if missing:
        raise ValueError(f"{len(missing)} dev QIDs are absent from official train")
    return docs, select_dev_sample([train[qid] for qid in dev_qids], n, seed=seed)


# ------------------------------------------------------------------------- prompt construction

def root_passage(row: Mapping, docs: Mapping[str, Mapping], rankings: Mapping
                 ) -> tuple[str, list[dict]]:
    """The single visible passage for a question: its BM25 top-1 retrieval result."""
    entry = rankings.get(row["id"])
    if not isinstance(entry, Mapping) or not entry.get("bm25"):
        raise KeyError(f"{row['id']}: no BM25 ranking available")
    root_id = entry["bm25"][0]
    if root_id not in docs:
        raise KeyError(f"{row['id']}: root passage {root_id} is absent from the corpus")
    return root_id, [{"context_id": root_id, "text": docs[root_id]["text"]}]


def build_prompts(rows: Sequence[Mapping], docs: Mapping[str, Mapping], rankings: Mapping
                  ) -> tuple[list[list[dict]], list[str]]:
    """Render one reader prompt per row, each showing only that row's BM25 top-1 passage."""
    visible, prompts = [], []
    for row in rows:
        _root_id, passages = root_passage(row, docs, rankings)
        visible.append(passages)
        prompts.append(reader_prompt(row["question"], passages))
    return visible, prompts


def smoke_prompt() -> str:
    """The synthetic connectivity prompt: one fabricated passage, no reference answer."""
    return reader_prompt(SMOKE_QUESTION, [SMOKE_PASSAGE])


# -------------------------------------------------------------------------------- local scoring

def score(rows: Sequence[Mapping], visible: Sequence[Sequence[Mapping]],
          responses: Sequence[str]) -> list[dict]:
    """Score each returned text locally: token-F1, citation-ID validity and gated success."""
    records = []
    for row, passages, raw in zip(rows, visible, responses):
        seen = [p["context_id"] for p in passages]
        text, _cited, valid = parse_reader_output(raw, seen)
        f1 = f1_score(text, [row["answer"]]) if text else 0.0
        records.append({"qid": row["id"], "root_context_id": seen[0],
                        "answered": text is not None, "citation_valid": valid,
                        "f1": round(f1, 4), "success": bool(answer_reward(f1, valid))})
    return records


def metrics(records: Sequence[Mapping]) -> dict:
    """Aggregate the dev40 records into the report's concrete numbers."""
    n = len(records)
    if not n:
        return {"n": 0}

    def rate(key: str) -> float:
        return round(sum(bool(r[key]) for r in records) / n, 4)

    return {"n": n, "mean_f1": round(sum(r["f1"] for r in records) / n, 4),
            "answer_rate": rate("answered"), "citation_valid_rate": rate("citation_valid"),
            "gated_success_rate": rate("success")}


# ------------------------------------------------------------------------------------ run

def run(*, out: Path = OUT, manifest: Path = MANIFEST_PATH, rankings: Path = RANKINGS_PATH,
        corpus: Path = CORPUS_PATH, train: Path = TRAIN_PATH, dev_n: int = DEV_N,
        dev_seed: int = DEV_SEED, max_requests: int = MAX_REQUESTS, smoke: bool = True,
        reader: CloudReader | None = None) -> dict:
    """Run the bounded cloud root dev40 pilot and write its report.

    ``reader`` may be supplied for a synthetic/offline smoke; otherwise one is built from
    :func:`build_identity`. The API-key check runs first, so a missing key fails before any QA
    row is read or any request is attempted. The smoke prompt (when enabled) shares the reader
    and its request budget with the 40 scored questions.
    """
    require_api_key()
    out = Path(out)
    report_path = out / REPORT_NAME
    # Refuse before loading data or spending any generation: the report is a measurement.
    refuse_overwrite((report_path,))
    docs, rows = load_dev40(manifest, corpus, train, n=dev_n, seed=dev_seed)
    ranking_data = json.loads(Path(rankings).read_text(encoding="utf-8"))
    if reader is None:
        identity = build_identity(corpus_path=Path(corpus), rankings_path=Path(rankings))
        reader = CloudReader(out / CACHE_NAME, identity=identity, max_requests=max_requests)
    if max_requests != reader.max_requests:
        raise ValueError("cloud pilot request cap differs from the reader limit")

    visible, prompts = build_prompts(rows, docs, ranking_data)
    planned = len({p for p in prompts if p not in reader.cache})
    if smoke and smoke_prompt() not in reader.cache:
        planned += 1
    if reader.cache.attempts + planned > max_requests:
        raise RuntimeError("cloud pilot cannot fit all 40 dev questions within the request cap")

    requests_before = reader.requests
    if smoke:
        smoke_response = reader.generate([smoke_prompt()])[0]
        smoke_answer, _cited, smoke_valid = parse_reader_output(smoke_response, ["SMOKE"])
        if not smoke_answer or not smoke_valid:
            raise RuntimeError("cloud compatibility smoke returned no valid JSON answer/citation")
        reader.flush()
    smoke_requests = reader.requests - requests_before
    responses = []
    provider_failures = []
    for row, prompt in zip(rows, prompts):
        try:
            responses.extend(reader.generate([prompt]))
        except RuntimeError as exc:
            if "empty cloud reader response" not in str(exc) and \
                    "cloud response did not complete normally" not in str(exc):
                raise
            responses.append("")
            provider_failures.append(row["id"])
    records = score(rows, visible, responses)

    report = {
        "scope": "cloud reader on official-train dev40 (reader seed42 selection); root BM25 "
                 "top-1 passage only",
        "config": {"model": MODEL, "model_revision": MODEL_REVISION,
                   "reader_prompt_version": PROMPT_VERSION, "max_new_tokens": MAX_NEW_TOKENS,
                   "decoding": "temperature=0", "evidence": "BM25 top-1 passage only",
                   "dev_n": dev_n, "dev_seed": dev_seed, "max_requests": max_requests,
                   "smoke": smoke},
        "data": {"manifest": str(manifest), "rankings": str(rankings), "corpus": str(corpus),
                 "train": str(train), "dataset_revision": DATASET_REVISION,
                 "official_validation_used": False, "official_test_loaded": False,
                 "gold_evidence_in_prompt": False},
        "dev_qids": [row["id"] for row in rows],
        "metrics": metrics(records),
        "requests_total": reader.cache.attempts,
        "smoke_requests": smoke_requests,
        "provider_failures": provider_failures,
        "generated_this_run": reader.generated,
        "cache_entries": len(reader.cache),
        "usage": reader.usage,
        "per_qid": records,
    }
    atomic_write_json(report_path, report)
    return report


# ------------------------------------------------------------------------------------- CLI

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--rankings", type=Path, default=RANKINGS_PATH)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--dev-n", type=int, default=DEV_N)
    parser.add_argument("--dev-seed", type=int, default=DEV_SEED)
    parser.add_argument("--max-requests", type=int, default=MAX_REQUESTS)
    parser.add_argument("--no-smoke", action="store_true",
                        help="skip the synthetic connectivity prompt")
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    args = parser.parse_args(argv)
    report = run(out=args.out, manifest=args.manifest, rankings=args.rankings, corpus=args.corpus,
                 train=args.train, dev_n=args.dev_n, dev_seed=args.dev_seed,
                 max_requests=args.max_requests, smoke=not args.no_smoke)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    m = report["metrics"]
    print(f"cloud root dev40: n={m['n']} mean_f1={m['mean_f1']} "
          f"citation_valid_rate={m['citation_valid_rate']} "
          f"gated_success_rate={m['gated_success_rate']} "
          f"requests={report['requests_total']}/{args.max_requests}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
