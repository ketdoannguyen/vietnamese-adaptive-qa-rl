"""Audit V2 ranking reachability and actual Laya evidence retention before training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.decision.vixstory_state_text_v2 import (
    HEAD_MAX_LEN,
    MAX_LEN,
    STATE_TEXT_VERSION,
    render_with_retention,
)

LAYA_REVISION = "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67"


def summarize_retention(rows: list[dict]) -> dict:
    """No invented pass percentage: flag any evidence loss for substantive review."""
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["n_seen"]].extend(row["passages"])
    summary = {}
    for count in range(1, 6):
        passages = buckets[count]
        if not passages:
            summary[str(count)] = {"passages": 0, "note": "no states observed with this document count"}
            continue
        ratios = [p["retention"] for p in passages]
        tokens = [p["retained_tokens"] for p in passages]
        summary[str(count)] = {"passages": len(passages),
                               "retained_tokens_p50": float(np.percentile(tokens, 50)),
                               "retained_tokens_p95": float(np.percentile(tokens, 95)),
                               "retention_p05": float(np.percentile(ratios, 5)),
                               "retention_p50": float(np.percentile(ratios, 50)),
                               "retention_p95": float(np.percentile(ratios, 95)),
                               "cut_rate": float(np.mean([p["cut"] for p in passages])),
                               "empty_rate": float(np.mean([p["retained_tokens"] == 0 for p in passages]))}
    return summary


def ranking_audit(worlds, teacher_class) -> dict:
    output = {"n_qids": len(worlds)}
    for tool in ("bm25", "dense"):
        for k in (1, 3, 5):
            coverage, full = [], []
            for world in worlds:
                gold = set(world.gold_context_ids)
                seen = set(world.rankings[tool][:k])
                coverage.append(len(gold & seen) / len(gold))
                full.append(gold <= seen)
            output[f"{tool}_recall_at_{k}"] = float(np.mean(coverage))
            output[f"{tool}_full_gold_at_{k}"] = float(np.mean(full))
    teachers = [teacher_class(world) for world in worlds]
    output["oracle_mean_return"] = float(np.mean([t.oracle_return() for t in teachers]))
    output["full_gold_reachable"] = float(np.mean([t.full_gold_reachable() for t in teachers]))
    output["full_gold_root"] = output["bm25_full_gold_at_1"]
    return output


def audit_split(worlds, tokenizer, teacher_class, *, split: str) -> tuple[dict, list[dict], list[dict]]:
    """Gold annotations are added only to the private audit, after public serialization."""
    rows, failures, samples = [], [], []
    for world in worlds:
        for obs in teacher_class(world).enumerate_states():
            try:
                _, audit = render_with_retention(obs, tokenizer=tokenizer)
            except ValueError as exc:
                failures.append({"qid": world.qid, "history": obs["history"], "error": str(exc)})
                continue
            passages = []
            for passage in audit["passages"]:
                passage = dict(passage)
                snippet = passage.pop("snippet")
                passage["is_gold"] = passage["doc_id"] in world.gold_context_ids
                passages.append(passage)
                if passage["cut"] and passage["is_gold"]:
                    samples.append({"split": split, "qid": world.qid, "question": world.question,
                                    "history": obs["history"], "n_seen": len(obs["seen_doc_ids"]),
                                    **passage, "retained_prefix": snippet,
                                    "full_passage": world.corpus[passage["doc_id"]]})
            rows.append({"qid": world.qid, "history": obs["history"],
                         "n_seen": len(obs["seen_doc_ids"]), "passages": passages,
                         "state_tokens": audit["state_tokens"], "truncated": audit["truncated"],
                         "essential_fields_present": audit["essential_fields_present"]})
    summary = {"n_qids": len(worlds), "n_states": len(rows), "failures": failures,
               "by_seen_passages": summarize_retention(rows),
               "final_clamp_truncations": sum(row["truncated"] for row in rows),
               "essential_field_failures": sum(not row["essential_fields_present"] for row in rows),
               "gold_passage_present_occurrences": sum(p["is_gold"] for row in rows for p in row["passages"]),
               "gold_passage_cut_occurrences": len(samples)}
    samples.sort(key=lambda row: (row["retention"], row["qid"], row["doc_id"], row["history"]))
    # A bounded review sample is explicitly not an exhaustive semantic sufficiency test.
    distinct = {}
    for sample in samples:
        distinct.setdefault((sample["qid"], sample["doc_id"]), sample)
    summary["review_sample_count"] = min(12, len(distinct))
    summary["distinct_cut_gold_question_passages"] = len(distinct)
    return summary, rows, list(distinct.values())[:12]


def write_new_json(path: Path, payload, *, private: bool = False) -> None:
    if private:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        handle = os.fdopen(fd, "w", encoding="utf-8")
    else:
        handle = path.open("x", encoding="utf-8")
    with handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def run(run_dir: Path, tokenizer_dir: Path | None = None) -> dict:
    from transformers import AutoTokenizer

    from src.pipeline.vixstory_teacher_v2 import VixStoryTeacherV2
    from src.pipeline.vixstory_worlds_v2 import load_worlds

    out = run_dir / "audit"
    out.mkdir(exist_ok=False)
    if tokenizer_dir is None:
        tokenizer_dir = (Path.home() / ".cache/huggingface/hub/models--convaiinnovations--laya-multilingual"
                         / "snapshots" / LAYA_REVISION / "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    audit = {"state_text_version": STATE_TEXT_VERSION, "max_len": MAX_LEN,
             "head_max_len": HEAD_MAX_LEN, "laya_revision": LAYA_REVISION,
             "tokenizer_sha256": hashlib.sha256((tokenizer_dir / "tokenizer.json").read_bytes()).hexdigest(),
             "serializer_sha256": hashlib.sha256(
                 (Path(__file__).resolve().parents[1] / "src/decision/vixstory_state_text_v2.py").read_bytes()).hexdigest(),
             "splits": {}}
    rankings = {}
    samples = []
    for split in ("train", "validation"):
        worlds = load_worlds(run_dir, split)
        rankings[split] = ranking_audit(worlds, VixStoryTeacherV2)
        summary, rows, review = audit_split(worlds, tokenizer, VixStoryTeacherV2, split=split)
        audit["splits"][split] = summary
        samples.extend(review)
        write_new_json(out / f"retention_{split}_private.json", rows, private=True)
        print(f"{split}: {summary['n_qids']} QIDs, {summary['n_states']} states, "
              f"{summary['gold_passage_cut_occurrences']} gold-passage cut occurrences", flush=True)
    hard_failures = any(s["failures"] or s["final_clamp_truncations"] or s["essential_field_failures"]
                        for s in audit["splits"].values())
    audit["status"] = ("FAILED_STRUCTURAL" if hard_failures else
                       "REQUIRES_EVIDENCE_REVIEW" if samples else "PASS_NO_EVIDENCE_CUT")
    audit["training_allowed"] = not hard_failures and not samples
    audit["gate_note"] = ("Final clamp false is not a semantic retention pass. Cut gold passages require "
                          "review of actual prefixes/full text before authorizing full training; "
                          "no arbitrary numerical retention threshold was added to the approved plan.")
    audit["review_sampling"] = "Up to 12 distinct lowest-retention gold QID/passage pairs per split; not exhaustive."
    write_new_json(out / "ranking_metrics.json", rankings)
    write_new_json(out / "retention_summary.json", audit)
    write_new_json(out / "retention_review_samples_private.json", samples, private=True)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--tokenizer-dir", type=Path)
    args = parser.parse_args()
    result = run(args.run_dir, args.tokenizer_dir)
    print(json.dumps({"status": result["status"], "training_allowed": result["training_allowed"]}))
    return 0 if result["training_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
