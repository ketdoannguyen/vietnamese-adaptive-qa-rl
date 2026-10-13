"""Probe focused cloud answers on gold and root evidence in fixed official-train dev40."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

from scripts.vixstory_cloud_root_pilot import (
    CORPUS_PATH, DEV_N, DEV_SEED, MANIFEST_PATH, RANKINGS_PATH, TRAIN_PATH,
    load_dev40, metrics, score,
)
from scripts.vixstory_reader_eval import CloudReader, build_identity
from src.retrieval.states import reachable_states
from src.utils.io import atomic_write_json

OUT = Path("outputs/experiments/vixstory-cloud-diagnosis-v4-2026-10-11")
PROMPT_VERSION = "v4-focused-answer"


def reader_prompt(question: str, passages) -> str:
    """Isolated answer-contract probe; not the production RL reader prompt."""
    rendered = "\n\n".join(f"[{p['context_id']}] {p['text']}" for p in passages)
    return ("Trả lời câu hỏi bằng tiếng Việt theo đúng các đoạn văn. Trong answer, đi thẳng "
            "vào từng ý được hỏi; nêu đủ tên, số liệu, thời gian, quan hệ và điều kiện cần thiết "
            "khi đoạn văn có căn cứ. Chỉ giữ thông tin cần để trả lời, không chép nguyên đoạn, "
            "không lặp ý hoặc kể bối cảnh không được hỏi. Không suy đoán khi thiếu chứng cứ; "
            "chỉ nêu phần có căn cứ hoặc để answer rỗng nếu không có. "
            'Trả về đúng JSON: {"answer":"<câu trả lời>","citations":["<ID đoạn đã dùng>"]}. '
            "Không viết gì ngoài JSON.\n"
            f"Câu hỏi: {question}\nĐoạn văn:\n{rendered}")


def main() -> None:
    if not os.environ.get("AIBOX_API_KEY"):
        raise RuntimeError("AIBOX_API_KEY is not set; no cloud diagnostic request sent")
    report_path = OUT / "report.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic report: {report_path}")
    docs, rows = load_dev40(MANIFEST_PATH, CORPUS_PATH, TRAIN_PATH, n=DEV_N, seed=DEV_SEED)
    identity = replace(build_identity(corpus_path=CORPUS_PATH,
                                      rankings_path=RANKINGS_PATH),
                       prompt_version=PROMPT_VERSION)
    reader = CloudReader(OUT / "reader-cache.json", identity=identity, max_requests=84)
    rankings = json.loads(RANKINGS_PATH.read_text(encoding="utf-8"))
    cases = {"gold": [], "root": []}
    for row in rows:
        gold = tuple(row["context_ids"])
        states = reachable_states(rankings[row["id"]], bootstrap_tool="bm25",
                                  bootstrap_passages=1, passage_per_search_action=1,
                                  max_optional_search=2)
        root = states[0]
        for kind, state in (("gold", gold), ("root", root)):
            if not state or any(doc_id not in docs for doc_id in state):
                raise ValueError(f"{row['id']}: missing {kind} diagnostic evidence")
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
                         "metrics_without_provider_failures": metrics(
                             [record for record in records if record["qid"] not in failures]),
                         "gold_visible": sum(set(row["context_ids"]) <= set(state)
                                             for row, state, _ in group),
                         "per_qid": records}
        print(f"{kind}: {results[kind]['metrics']}; "
              f"attempts={reader.cache.attempts}/{reader.max_requests}", flush=True)
    report = {"scope": "fixed official-train dev40; v4 gold/root prompt-only diagnostic; "
                        "no validation/test or RL training",
              "cache_identity": reader.identity.as_dict(), "attempts": reader.cache.attempts,
              "usage_this_run": reader.usage, "results": results}
    atomic_write_json(report_path, report)
    print(f"diagnostic report: {report_path}")


if __name__ == "__main__":
    main()
