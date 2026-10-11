"""Pre-training VixSTORY suitability audit; never reads its official test split.

Search evidence is measured on real BM25/E5 rankings over the shared 3,462-passage
corpus. Oracle answer advantages require a separately verified reader and are *not*
computed from gold passage IDs. No existing ViQuAD artifacts are written.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Mapping, Sequence

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from src.retrieval.bm25 import BM25Index
from src.retrieval.dense import DenseBank, DenseEncoder, rank_by_inner_product
from src.retrieval.states import next_unseen, reachable_states
from src.utils.hashing import sha256_file, sha256_json, stable_seed
from src.utils.io import atomic_write_json

REPO = "ura-hcmut/VixSTORY"
FILES = ("corpus/corpus.parquet", "data/train.parquet", "data/validation.parquet")
DEFAULT_ROOT = Path("outputs/experiments/2026-10-11-vixstory-suitability")
REQUIRED_QA = frozenset({"id", "question", "answer", "context_ids", "hop_level", "question_type"})
REQUIRED_DOC = frozenset({"context_id", "text", "school_id"})
TOOLS = ("bm25", "dense")


def audit_record(row: Mapping, docs: Mapping[str, Mapping]) -> dict:
    if not REQUIRED_QA <= row.keys():
        raise ValueError(f"VixSTORY QA schema lacks {sorted(REQUIRED_QA - row.keys())}")
    if not row["id"] or not row["question"] or not row["answer"]:
        raise ValueError("VixSTORY QA row has blank id, question or answer")
    gold = tuple(row["context_ids"] or ())
    if not gold or len(set(gold)) != len(gold):
        raise ValueError(f"{row['id']}: missing or duplicate gold passage IDs")
    missing = set(gold) - docs.keys()
    if missing:
        raise ValueError(f"{row['id']}: missing gold passage(s): {len(missing)}")
    for inline in row.get("contexts") or []:
        doc = docs.get(inline.get("context_id"))
        if doc is None or inline.get("text") != doc["text"]:
            raise ValueError(f"{row['id']}: inline context differs from corpus")
    return {"qid": row["id"], "hop_level": int(row["hop_level"]),
            "question_type": row["question_type"], "n_gold_passages": len(gold),
            "hop_mismatch": row["hop_level"] != len(gold),
            "answer_exact_span_in_gold": any(row["answer"] in docs[d]["text"] for d in gold)}


def sample_train(rows: Sequence[Mapping], *, n: int = 500, seed: int = 42) -> list[Mapping]:
    """Deterministic hop/type stratification, independent of answers and context IDs."""
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate QA IDs in train")
    if n > len(rows) or n <= 0:
        raise ValueError(f"not enough train rows for requested sample n={n}")
    buckets: dict[tuple[int, str], list[Mapping]] = defaultdict(list)
    for row in rows:
        buckets[(int(row["hop_level"]), str(row["question_type"]))].append(row)
    for key, bucket in buckets.items():
        bucket.sort(key=lambda r: (stable_seed(seed, *key, r["id"]), r["id"]))
    selected: list[Mapping] = []
    # Proportionally track the natural train distribution; deterministic residual ties.
    keys = sorted(buckets)
    quota = {key: n * len(buckets[key]) // len(rows) for key in keys}
    for key in sorted(keys, key=lambda k: (-(n * len(buckets[k]) % len(rows)), k)):
        if sum(quota.values()) >= n:
            break
        quota[key] += 1
    for key in keys:
        selected.extend(buckets[key][:quota[key]])
    if len(selected) != n:
        raise AssertionError("stratified sample size drifted")
    return sorted(selected, key=lambda r: r["id"])


def evidence_metrics(gold: Sequence[str], visible: Sequence[str]) -> dict:
    selected = set(visible)
    target = set(gold)
    if not target:
        raise ValueError("gold passage set must be nonempty")
    return {"full": target <= selected, "recall": len(target & selected) / len(target)}


def search_paths(rankings: Mapping[str, Sequence[str]], *, gold: Sequence[str],
                 budget: int = 2) -> dict:
    """Measure reachable complete evidence in *one path*, never a union of branches."""
    states = reachable_states(rankings, max_optional_search=budget)
    first = states[0]
    complete = [len(state) - len(first) for state in states if set(gold) <= set(state)]
    best = min(complete) if complete else None
    return {"initial_full": set(gold) <= set(first),
            "initial_gold_count": len(set(gold) & set(first)),
            "can_complete_within": {k: best is not None and best <= k for k in range(budget + 1)},
            "best_first_full_step": best,
            "states_count": len(states),
            "max_evidence_recall": max(evidence_metrics(gold, state)["recall"] for state in states)}


def controlled_initial(rankings: Mapping[str, Sequence[str]], gold: Sequence[str],
                       *, mode: str) -> tuple[str, ...] | None:
    """A *natural* bootstrap classified offline; never inject a gold passage."""
    start = tuple(rankings["bm25"][:1])
    missing = len(set(gold) - set(start))
    if mode == "full" and missing == 0:
        return start
    if mode == "missing_one" and missing == 1:
        return start
    if mode == "missing_all" and missing == len(set(gold)):
        return start
    if mode == "unreachable" and not search_paths(rankings, gold=gold)["can_complete_within"][2]:
        return start
    return None


def sample_controlled(cases: Sequence[tuple[str, str]], *, n: int = 500,
                      seed: int = 42) -> tuple[list[tuple[str, str]], dict[str, int]]:
    """Offline diagnostic sample; keep natural evaluation as the primary result.

    Only verified real-retriever categories enter this sample. If a category has too
    few cases, record the actual count and redistribute its unused quota; never
    invent missing episodes to match a target percentage.
    """
    ratios = {"full": 0.25, "first": 0.30, "second": 0.30, "unreachable": 0.15}
    if len({qid for qid, _ in cases}) != len(cases):
        raise ValueError("duplicate controlled diagnostic QIDs")
    groups: dict[str, list[str]] = {key: [] for key in ratios}
    for qid, group in cases:
        if group not in groups:
            raise ValueError(f"unknown controlled stratum {group!r}")
        groups[group].append(qid)
    for key in groups:
        groups[key].sort(key=lambda qid: (stable_seed(seed, qid), qid))
    limit = min(n, len(cases))
    quotas = {key: min(int(limit * ratio), len(groups[key])) for key, ratio in ratios.items()}
    while sum(quotas.values()) < limit:
        available = [key for key in ratios if quotas[key] < len(groups[key])]
        if not available:
            break
        key = max(available, key=lambda item: (limit * ratios[item] - quotas[item],
                                               -list(ratios).index(item)))
        quotas[key] += 1
    selected = [(qid, group) for group in ratios for qid in groups[group][:quotas[group]]]
    return sorted(selected), quotas


def oracle_advantage(rankings: Mapping[str, Sequence[str]],
                     terminal: Callable[[tuple[str, ...]], tuple[float, float]], *,
                     search_costs: Mapping[str, float], budget: int = 2) -> dict:
    """Private offline optimal return from an actual reader; never a policy feature."""
    seen = tuple(rankings["bm25"][:1])
    if not seen:
        raise ValueError("no BM25 bootstrap available")

    def value(state: tuple[str, ...], remaining: int) -> tuple[float, dict[str, float]]:
        answer, abstain = terminal(state)
        q = {"ANSWER": float(answer), "ABSTAIN": float(abstain)}
        if remaining:
            for tool in TOOLS:
                addition = next_unseen(rankings[tool], state, 1)
                if addition:
                    future, _ = value(state + tuple(addition), remaining - 1)
                    q[f"SEARCH_{tool.upper()}"] = future - search_costs[tool]
        return max(q.values()), q

    _, q = value(seen, budget)
    best_terminal = max(q["ANSWER"], q["ABSTAIN"])
    searches = [name for name in q if name.startswith("SEARCH_")]
    best_search = max((q[name] for name in searches), default=float("-inf"))
    # Terminal actions win ties: paying to SEARCH without improving return is not useful.
    best = max(q, key=lambda name: (q[name], not name.startswith("SEARCH_"), name))
    return {"root_q": q, "advantage": best_search - best_terminal,
            "optimal_action": best}


def _source(revision: str) -> tuple[dict[str, Path], dict[str, list[dict]]]:
    paths, tables = {}, {}
    for filename in FILES:
        local = Path(hf_hub_download(REPO, filename, repo_type="dataset", revision=revision))
        paths[filename] = local
        tables[filename] = pq.read_table(local).to_pylist()
    return paths, tables


def _rankings(questions: Sequence[Mapping], docs: Mapping[str, Mapping], *,
              dense: bool, model_name: str) -> tuple[dict[str, dict[str, list[str]]], dict]:
    ids = sorted(docs)
    texts = {doc_id: docs[doc_id]["text"] for doc_id in ids}
    bm25 = BM25Index.build(texts)
    ranks = {row["id"]: {"bm25": [d for d, _ in bm25.rank(row["question"], top_n=8)]}
             for row in questions}
    meta = {"dense_computed": False, "bm25_corpus_passages": len(ids), "rankings_top_n": 8}
    if dense:
        encoder = DenseEncoder(model_name=model_name, batch_size=32, max_length=512)
        try:
            bank = DenseBank.build(texts, encoder)
            vectors = encoder.encode_queries([r["question"] for r in questions])
            for row, vector in zip(questions, vectors):
                ranks[row["id"]]["dense"] = [d for d, _ in
                    rank_by_inner_product(vector, bank.vectors, bank.doc_ids)[:8]]
            meta.update({"dense_computed": True, "dense_model": model_name})
        finally:
            encoder.unload()
    return ranks, meta


def run(*, out: Path, n_train: int = 500, n_controlled: int = 60, seed: int = 42,
        dense_model: str = "intfloat/multilingual-e5-small") -> dict:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite suitability output {out}")
    revision = HfApi().dataset_info(REPO).sha
    paths, tables = _source(revision)
    raw_docs = tables["corpus/corpus.parquet"]
    if any(not REQUIRED_DOC <= row.keys() for row in raw_docs):
        raise ValueError("VixSTORY corpus schema changed")
    docs = {row["context_id"]: row for row in raw_docs}
    if len(docs) != len(raw_docs):
        raise ValueError("duplicate corpus context IDs")
    train = tables["data/train.parquet"]
    val = tables["data/validation.parquet"]
    for rows in (train, val):
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("duplicate QA IDs in one split")
        for row in rows:
            audit_record(row, docs)
    if set(r["id"] for r in train) & set(r["id"] for r in val):
        raise ValueError("official train/validation QA IDs overlap")
    sampled = sample_train(train, n=n_train, seed=seed)
    evaluation = sampled + val
    rankings, retrieval_meta = _rankings(evaluation, docs, dense=True, model_name=dense_model)
    summary = {"source": REPO, "source_revision": revision,
               "source_sha256": {key: sha256_file(path) for key, path in paths.items()},
               "license": "CC BY-NC-SA 4.0; research use only", "train_count": len(train),
               "train_sample_count": len(sampled), "train_sample_qids_sha256": sha256_json([r["id"] for r in sampled]),
               "validation_count": len(val), "test_loaded": False,
               "shared_gold_passages_train_val": len({d for r in train for d in r["context_ids"]}
                                                    & {d for r in val for d in r["context_ids"]}),
               "retrieval": retrieval_meta, "splits": {}}
    for name, rows in (("train_sample", sampled), ("validation", val)):
        audit = [audit_record(r, docs) for r in rows]
        evidence = [search_paths(rankings[r["id"]], gold=r["context_ids"], budget=2) for r in rows]
        full = Counter("full" if p["initial_full"] else
                       "first" if p["best_first_full_step"] == 1 else
                       "second" if p["best_first_full_step"] == 2 else "unreachable"
                       for p in evidence)
        summary["splits"][name] = {
            "n": len(rows), "hop_level": dict(Counter(a["hop_level"] for a in audit)),
            "gold_passages": dict(Counter(a["n_gold_passages"] for a in audit)),
            "question_types": dict(Counter(a["question_type"] for a in audit)),
            "hop_count_mismatches": sum(a["hop_mismatch"] for a in audit),
            "answer_exact_span_in_gold": sum(a["answer_exact_span_in_gold"] for a in audit),
            "initial_full_evidence": full["full"], "missing_initial_evidence": len(rows) - full["full"],
            "complete_after_one_search": full["first"],
            "complete_after_two_searches": full["second"],
            "not_complete_within_two_searches": full["unreachable"],
            "reachability_at_1": full["full"] + full["first"],
            "reachability_at_2": len(rows) - full["unreachable"],
            "max_recall_mean": sum(p["max_evidence_recall"] for p in evidence) / len(rows),
            "state_count_max": max(p["states_count"] for p in evidence),
        }
    # Optional diagnostic cohort on *train only*. Rankings remain the natural ones;
    # classification is privileged offline analysis, never the bootstrap input.
    train_cases = []
    for row in sampled:
        reach = search_paths(rankings[row["id"]], gold=row["context_ids"], budget=2)
        label = ("full" if reach["initial_full"] else
                 "first" if reach["best_first_full_step"] == 1 else
                 "second" if reach["best_first_full_step"] == 2 else "unreachable")
        train_cases.append((row["id"], label))
    cohort, cohort_counts = sample_controlled(train_cases, n=n_controlled, seed=seed)
    summary["controlled_train_diagnostic"] = {
        "requested_total": n_controlled,
        "requested_target_fractions": {"full": 0.25, "first": 0.30,
                                       "second": 0.30, "unreachable": 0.15},
        "available_counts": dict(Counter(label for _, label in train_cases)),
        "actual_counts": cohort_counts, "qid_stratum_sha256": sha256_json(cohort),
        "note": "Train-only strata; not an evaluation distribution or policy input. "
                "No gold-injected bootstrap or artificial retrieval ranking."
    }
    summary["reader_gold_evidence_em_f1"] = None
    summary["answer_improvement_after_search"] = None
    summary["positive_search_advantage_rate"] = None
    summary["optimal_action_distribution"] = None
    summary["full_episode_reward"] = None
    summary["missing_reader_metrics_reason"] = (
        "Free-form multi-passage QA requires a verified reader over seen passages. "
        "Gold-passage reachability alone cannot substitute for reader return or oracle Q.")
    out.mkdir(parents=True, exist_ok=False)
    atomic_write_json(out / "suitability.json", summary)
    # Train-only QIDs/rankings are private local intermediate inputs, not gold labels.
    atomic_write_json(out / "train_sample_qids.json", [r["id"] for r in sampled])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--n-train", type=int, default=500)
    parser.add_argument("--n-controlled", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run(out=args.out, n_train=args.n_train,
                 n_controlled=args.n_controlled, seed=args.seed)
    print(json.dumps({"train": result["splits"]["train_sample"],
                      "validation": result["splits"]["validation"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
