"""Compare reachable and gold evidence on fixed official-train dev40, without training."""

from __future__ import annotations

import json
import os
from pathlib import Path

from scripts.vixstory_cloud_root_pilot import (
    CORPUS_PATH, DEV_N, DEV_SEED, MANIFEST_PATH, RANKINGS_PATH, TRAIN_PATH,
    load_dev40, metrics, score,
)
from scripts.vixstory_reader_eval import CloudReader, build_identity, reader_prompt
from src.retrieval.states import reachable_states
from src.utils.io import atomic_write_json

OUT = Path("outputs/experiments/vixstory-cloud-diagnosis-v2-2026-10-11")


def main() -> None:
    if not os.environ.get("AIBOX_API_KEY"):
        raise RuntimeError("AIBOX_API_KEY is not set; no cloud diagnostic request sent")
    report_path = OUT / "report.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic report: {report_path}")
    docs, rows = load_dev40(MANIFEST_PATH, CORPUS_PATH, TRAIN_PATH, n=DEV_N, seed=DEV_SEED)
    rankings = json.loads(RANKINGS_PATH.read_text(encoding="utf-8"))
    reader = CloudReader(OUT / "reader-cache.json",
                         identity=build_identity(corpus_path=CORPUS_PATH,
                                                 rankings_path=RANKINGS_PATH),
                         max_requests=130)
    cases = {"root": [], "reachable": [], "gold": []}
    for row in rows:
        qid = row["id"]
        rank = rankings[qid]
        states = reachable_states(rank, bootstrap_tool="bm25", bootstrap_passages=1,
                                  passage_per_search_action=1, max_optional_search=2)
        root = states[0]
        deep = next((state for state in states if len(state) == 3), states[-1])
        gold = tuple(row["context_ids"])
        for kind, state in (("root", root), ("reachable", deep), ("gold", gold)):
            if not state or any(doc_id not in docs for doc_id in state):
                raise ValueError(f"{qid}: missing diagnostic evidence for {kind}")
            prompt = reader_prompt(row["question"],
                                   [{"context_id": doc_id, "text": docs[doc_id]["text"]}
                                    for doc_id in state])
            cases[kind].append((row, state, prompt))
    planned = {prompt for group in cases.values() for _, _, prompt in group}
    if reader.cache.attempts + sum(p not in reader.cache for p in planned) > reader.max_requests:
        raise RuntimeError("diagnostic request cap cannot cover all missing prompts")

    results = {}
    for kind, group in cases.items():
        selected_rows = []
        visible = []
        responses = []
        failures = []
        for row, state, prompt in group:
            try:
                response = reader.generate([prompt])[0]
            except RuntimeError as exc:
                if "empty cloud reader response" not in str(exc) and \
                        "cloud response did not complete normally" not in str(exc) and \
                        "recorded failure" not in str(exc):
                    raise
                response = ""
                failures.append(row["id"])
            selected_rows.append(row)
            visible.append([{"context_id": doc_id} for doc_id in state])
            responses.append(response)
        records = score(selected_rows, visible, responses)
        results[kind] = {"metrics": metrics(records), "failed_qids": failures,
                         "gold_visible": sum(set(row["context_ids"]) <= set(state)
                                             for row, state, _ in group),
                         "per_qid": records}
        print(f"{kind}: {results[kind]['metrics']}; "
              f"attempts={reader.cache.attempts}/{reader.max_requests}", flush=True)
    report = {"scope": "fixed official-train dev40; root, reachable BM25 depth2, "
                        "and gold-evidence diagnostic only; no validation/test or RL training",
              "cache_identity": reader.identity.as_dict(), "attempts": reader.cache.attempts,
              "usage_this_run": reader.usage, "results": results}
    atomic_write_json(report_path, report)
    print(f"diagnostic report: {report_path}")


if __name__ == "__main__":
    main()
