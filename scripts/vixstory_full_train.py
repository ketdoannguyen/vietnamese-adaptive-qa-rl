"""Prepare, warm and train the full official VixSTORY train split.

The former train-dev200 QIDs are training examples, not held out. Missing reader
responses block training; official validation and test are never opened.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.vixstory_pilot_cache import (
    DEFAULT_BATCH_SIZE,
    EpisodeBudget,
    assert_within_budget,
    char_blockers,
    next_free_path,
    read_corpus,
    read_questions,
    read_rankings,
    state_prompts,
    warm_prompts,
)
from scripts.vixstory_pilot_manifest import EXPECTED_SHA256, REVISION, TRAIN_COUNT
from scripts.vixstory_pilot_rankings import (
    CORPUS_PATH,
    CORPUS_SHA256,
    E5_MODEL,
    E5_REVISION,
    TRAIN_PATH,
    build_rankings,
)
from scripts.vixstory_reader_eval import (
    CACHE_VERSION,
    MAX_ATTEMPTS_PER_PROMPT,
    CloudReader,
    build_identity,
)
from src.retrieval.dense import DenseEncoder
from src.utils.hashing import sha256_file, sha256_json
from src.utils.io import atomic_write_json, read_json
from src.utils.runs import OutputNamespaceLock

ROOT = Path("outputs/experiments/vixstory-full-train-2026-10-12")
MANIFEST = "manifest.json"
RANKINGS = "rankings_top12.json"
CACHE = "reader-cache-opencode-go-v4.1-flash.json"
FAILED = "failed-qids.json"
TEACHER = "teacher/train2553.jsonl"
TEACHER_PROVENANCE = "teacher/train2553.provenance.json"
LAYA_BASE = (Path.home() / ".cache/huggingface/hub/models--convaiinnovations--laya-multilingual"
             / "snapshots/e4e9ddf21a7b1903b7acffd8814ad4307bf63a67")


def prepare(out: Path, *, train_path: Path = TRAIN_PATH,
            corpus_path: Path = CORPUS_PATH, encoder=None) -> dict:
    """Freeze all official-train QIDs and score label-free BM25/E5 rankings."""
    if (out / MANIFEST).exists():
        raise FileExistsError(f"full-train artifact already exists: {out / MANIFEST}")
    if sha256_file(train_path) != EXPECTED_SHA256["data/train.parquet"]:
        raise ValueError("official train source changed")
    if sha256_file(corpus_path) != CORPUS_SHA256:
        raise ValueError("pinned corpus changed")
    questions = read_questions(train_path)
    if len(questions) != TRAIN_COUNT:
        raise ValueError(f"official train count changed: {len(questions)}")
    corpus = read_corpus(corpus_path)
    qids = sorted(questions)
    manifest = {"artifact": "vixstory_full_official_train", "source_revision": REVISION,
                "train_sha256": EXPECTED_SHA256["data/train.parquet"],
                "corpus_sha256": CORPUS_SHA256, "n_qids": len(qids),
                "qids": qids, "qids_sha256": sha256_json(qids),
                "official_validation_loaded": False, "official_test_loaded": False,
                "evaluation_note": "Former pilot dev200 QIDs are now training data, not held out."}
    owns_encoder = encoder is None
    if owns_encoder:
        encoder = DenseEncoder(model_name=E5_MODEL, revision=E5_REVISION,
                               batch_size=32, max_length=512, device="cpu")
    try:
        rankings = build_rankings(corpus, questions, top_n=12, encoder=encoder)
    finally:
        if owns_encoder:
            encoder.unload()
    if set(rankings) != set(qids):
        raise RuntimeError("full-train retrieval did not cover every QID")
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / RANKINGS, rankings)
    atomic_write_json(out / MANIFEST, {**manifest,
        "rankings_sha256": sha256_file(out / RANKINGS),
        "retrieval": {"bm25": "src.retrieval.bm25.BM25Index",
                      "dense_model": E5_MODEL, "dense_revision": E5_REVISION,
                      "top_n": 12}})
    return {"n_qids": len(qids), "rankings_path": str(out / RANKINGS)}


def _inputs(out: Path, train_path: Path, corpus_path: Path):
    manifest = read_json(out / MANIFEST)
    qids = manifest["qids"]
    if (manifest["n_qids"] != TRAIN_COUNT or qids != sorted(set(qids)) or
            sha256_json(qids) != manifest["qids_sha256"] or
            sha256_file(train_path) != manifest["train_sha256"] or
            sha256_file(corpus_path) != manifest["corpus_sha256"] or
            sha256_file(out / RANKINGS) != manifest["rankings_sha256"]):
        raise ValueError("full-train manifest/source/rankings drifted")
    questions = read_questions(train_path)
    rankings = read_rankings(out / RANKINGS)
    if set(questions) != set(qids) or set(rankings) != set(qids):
        raise ValueError("full-train question/ranking QIDs differ from manifest")
    for qid in qids:
        for tool in ("bm25", "dense"):
            if len(rankings[qid][tool]) < 12:
                raise ValueError(f"{qid}: {tool} ranking has fewer than 12 passages")
    return qids, questions, read_corpus(corpus_path), rankings


def _coverage(qids, questions, corpus, rankings, cache, budget):
    """Reconstruct per-QID coverage from the persisted cache, including after interruption."""
    failed = {}
    complete = []
    n_states = n_missing = 0
    for qid in qids:
        states = state_prompts(qids=[qid], questions=questions, corpus=corpus,
                               rankings=rankings, budget=budget)
        n_states += len(states)
        oversized = char_blockers(states)
        missing = [sp for sp in states if sp.prompt not in cache]
        n_missing += len(missing)
        if oversized or missing:
            reasons = []
            if oversized:
                reasons.append("prompt_exceeds_character_budget")
            if any(sp.prompt in cache.failures for sp in missing):
                reasons.append("recorded_response_failure")
            if any(cache.retries.get(sp.prompt, 0) >= MAX_ATTEMPTS_PER_PROMPT for sp in missing):
                reasons.append("attempts_exhausted")
            if any(sp.prompt not in cache.failures and
                   cache.retries.get(sp.prompt, 0) < MAX_ATTEMPTS_PER_PROMPT for sp in missing):
                reasons.append("pending")
            failed[qid] = {"missing_states": len(missing), "reasons": reasons}
        else:
            complete.append(qid)
    return {"n_official_train": len(qids), "n_complete_qids": len(complete),
            "n_incomplete_qids": len(failed), "n_states": n_states,
            "n_missing_states": n_missing, "complete_qids_sha256": sha256_json(complete),
            "complete_qids": complete, "incomplete": failed,
            "attempts": cache.attempts, "entries": len(cache),
            "identity": cache.identity.as_dict(), "cache_version": CACHE_VERSION,
            "dev200_is_held_out": False}


def warm(out: Path, *, train_path: Path = TRAIN_PATH, corpus_path: Path = CORPUS_PATH,
         max_requests: int, workers: int = 10, batch_size: int = DEFAULT_BATCH_SIZE,
         qid_batch_size: int = 32) -> dict:
    if not 1 <= max_requests <= 30000:
        raise ValueError("full-train request cap must be between 1 and 30000")
    raise RuntimeError("OpenCode Go is documented for coding-agent traffic, not bulk QA "
                       "generation. Obtain provider authorization for this workload and "
                       "implement its required session protocol before enabling warm-up.")
    if not 1 <= workers <= 10 or batch_size < 1 or qid_batch_size < 1:
        raise ValueError("workers must be 1-10; batch sizes must be positive")
    qids, questions, corpus, rankings = _inputs(out, train_path, corpus_path)
    budget = EpisodeBudget()
    identity = build_identity(corpus_path=corpus_path, rankings_path=out / RANKINGS,
                              provider="opencode-go")
    cache_path = out / CACHE
    with OutputNamespaceLock(cache_path.with_name(cache_path.name + ".lock")):
        with OutputNamespaceLock(out / ".full-train-warm.lock"):
            reader = CloudReader(cache_path, identity=identity, max_requests=max_requests)
            if reader.cache.attempts > max_requests:
                raise ValueError("full-train cache already exceeds requested attempt cap")
            print(f"full official train: {len(qids)} QIDs, "
                  f"{reader.cache.attempts}/{max_requests} attempts reserved", flush=True)
            try:
                for start in range(0, len(qids), qid_batch_size):
                    selected = qids[start:start + qid_batch_size]
                    prompts = state_prompts(qids=selected, questions=questions,
                                            corpus=corpus, rankings=rankings, budget=budget)
                    assert_within_budget(prompts, reader=reader)
                    eligible = [sp.prompt for sp in prompts]
                    if eligible:
                        warm_prompts(reader, eligible, batch_size=batch_size, workers=workers)
                    print(f"  processed {min(start + qid_batch_size, len(qids))}/{len(qids)} QIDs; "
                          f"cached {len(reader.cache)}; attempts {reader.cache.attempts}",
                          flush=True)
            finally:
                reader.flush()
                coverage = _coverage(qids, questions, corpus, rankings, reader.cache, budget)
                atomic_write_json(next_free_path(out / FAILED), coverage)
                print("full-train coverage:", json.dumps({k: v for k, v in coverage.items()
                      if k in ("n_official_train", "n_complete_qids", "n_incomplete_qids",
                               "n_states", "n_missing_states", "attempts", "entries")}),
                      flush=True)
            return coverage


def train_inputs(out: Path, *, train_path: Path = TRAIN_PATH,
                 corpus_path: Path = CORPUS_PATH):
    """Fail closed unless every official-train state has a real reader response."""
    from scripts.vixstory_reader_eval import GenerationCache
    from src.pipeline.vixstory_worlds import assemble_split, read_qa_rows
    from src.rl.vixstory_env import VixStoryEnvConfig

    qids, _questions, corpus, rankings = _inputs(out, train_path, corpus_path)
    identity = build_identity(corpus_path=corpus_path, rankings_path=out / RANKINGS,
                              provider="opencode-go")
    cache = GenerationCache(out / CACHE, identity)
    coverage = _coverage(qids, _questions, corpus, rankings, cache, EpisodeBudget())
    if coverage["n_complete_qids"] != TRAIN_COUNT or coverage["n_missing_states"]:
        raise RuntimeError(f"full-train reader coverage incomplete: "
                           f"{coverage['n_complete_qids']}/{TRAIN_COUNT} QIDs, "
                           f"{coverage['n_missing_states']} missing states")
    rows = read_qa_rows(train_path)
    bundle = assemble_split(role="train", qids=qids, corpus=corpus, rows=rows,
                            rankings=rankings, cache=cache, config=VixStoryEnvConfig())
    if bundle.n_states != coverage["n_states"]:
        raise ValueError("full-train world states do not match coverage")
    return bundle, coverage


def make_teacher(out: Path, *, base: Path = LAYA_BASE,
                 train_path: Path = TRAIN_PATH, corpus_path: Path = CORPUS_PATH) -> dict:
    from transformers import AutoTokenizer
    from scripts.vixstory_pilot import write_teacher
    from src.rl.vixstory_env import VixStoryEnvConfig

    bundle, coverage = train_inputs(out, train_path=train_path, corpus_path=corpus_path)
    target, provenance = out / TEACHER, out / TEACHER_PROVENANCE
    if target.exists() or provenance.exists():
        raise FileExistsError("refusing to overwrite full-train teacher/provenance")
    tokenizer = AutoTokenizer.from_pretrained(base / "tokenizer", local_files_only=True)
    result = write_teacher(bundle, VixStoryEnvConfig(), target, tokenizer=tokenizer,
                           expected_role="train", expected_qids=TRAIN_COUNT)
    result["cache_sha256"] = sha256_file(out / CACHE)
    result["rankings_sha256"] = sha256_file(out / RANKINGS)
    result["coverage"] = {key: coverage[key] for key in
                          ("n_official_train", "n_complete_qids", "n_states")}
    atomic_write_json(provenance, result)
    return result


def train_laya(out: Path, *, base: Path = LAYA_BASE,
               train_path: Path = TRAIN_PATH, corpus_path: Path = CORPUS_PATH,
               epochs: int = 2) -> dict:
    from scripts.vixstory_pilot_laya import run

    train_inputs(out, train_path=train_path, corpus_path=corpus_path)
    return run(data=out / TEACHER, base=str(base), pilot_manifest=out / MANIFEST,
               out_root=out / "laya", teacher_provenance=out / TEACHER_PROVENANCE,
               loss="both", epochs=epochs, device="cuda:0", full_train=True)


def train_policy(out: Path, *, train_path: Path = TRAIN_PATH,
                 corpus_path: Path = CORPUS_PATH, max_updates: int = 100) -> dict:
    import torch
    from src.pipeline.vixstory_arms import LayaPrior
    from src.rl.reinforce import RLConfig
    from src.rl.vixstory_env import VixStoryEnv, VixStoryEnvConfig
    from src.rl.vixstory_policy import train_vixstory_policy

    if not torch.cuda.is_available():
        raise RuntimeError("full-train policy training requires CUDA")
    bundle, coverage = train_inputs(out, train_path=train_path, corpus_path=corpus_path)
    teacher = read_json(out / TEACHER_PROVENANCE)
    if (teacher.get("n_qids") != TRAIN_COUNT or teacher.get("n_states") != bundle.n_states
            or teacher.get("cache_sha256") != sha256_file(out / CACHE)):
        raise ValueError("teacher provenance does not match full-train cache/QIDs")
    for arm in ("E04_ce", "E05_rlcd"):
        if not (out / "laya" / arm / "model.safetensors").is_file():
            raise FileNotFoundError(f"missing full-train Laya checkpoint: {arm}")
    result_path = out / "reinforce.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite {result_path}")
    priors = {"ce": LayaPrior(out / "laya/E04_ce", device="cuda"),
              "rlcd": LayaPrior(out / "laya/E05_rlcd", device="cuda")}
    try:
        env = VixStoryEnv(bundle.worlds, VixStoryEnvConfig())
        monitor_qids = tuple(bundle.qids[:200])
        result = train_vixstory_policy(
            env, env, RLConfig(max_updates=max_updates, eval_every=10), priors=priors,
            run_root=out / "reinforce/seed-42", train_qids=bundle.qids,
            dev_qids=monitor_qids, arm_seeds={"ce": 42, "rlcd": 42},
            prior_refs={"ce": str(out / "laya/E04_ce"),
                        "rlcd": str(out / "laya/E05_rlcd")},
            device="cuda", fixed_updates=True, train_monitor=True)
        report = {"status": "SUCCEEDED", "scope": "official_train_only_in_sample",
                  "monitor_qids_sha256": sha256_json(list(monitor_qids)),
                  "n_train_qids": coverage["n_complete_qids"],
                  "reader_cache_sha256": sha256_file(out / CACHE),
                  "arms": {arm: record.to_dict() for arm, record in result.items()}}
        atomic_write_json(result_path, report)
        return report
    finally:
        for prior in priors.values():
            prior.close()


def write_report(out: Path, *, train_path: Path = TRAIN_PATH,
                 corpus_path: Path = CORPUS_PATH) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.eval.reporting import parse_laya_losses

    bundle, coverage = train_inputs(out, train_path=train_path, corpus_path=corpus_path)
    report_path = out / "training-report.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    figure_paths = {}
    laya_losses = {}
    for arm in ("E04_ce", "E05_rlcd"):
        log = out / "laya" / f"{arm}.log"
        manifest = out / "laya" / f"{arm}.manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"full-train Laya manifest missing: {manifest}")
        if log.is_file():
            series = parse_laya_losses(log.read_text(encoding="utf-8", errors="replace"))
            if len(series) >= 2:
                laya_losses[arm] = series
    if laya_losses:
        fig, ax = plt.subplots(figsize=(9, 5))
        for arm, series in laya_losses.items():
            ax.plot([step for step, _ in series], [loss for _, loss in series], label=arm)
        ax.set(xlabel="optimizer step", ylabel="Laya training loss", title="Full-train Laya loss")
        ax.legend()
        ax.grid(alpha=0.25)
        figure_paths["laya_loss"] = plots / "laya_loss_curves.png"
        fig.savefig(figure_paths["laya_loss"], dpi=150, bbox_inches="tight")
        plt.close(fig)
    histories = {}
    for arm in ("E06_rl_rlcd", "E07_rl_ce"):
        path = out / "reinforce" / "seed-42" / arm / "history.json"
        if not path.is_file():
            raise FileNotFoundError(f"RL history missing: {path}")
        histories[arm] = read_json(path)
    for filename, field, label in (
            ("rl_train_reward.png", "train_mean_reward", "In-sample train reward"),
            ("rl_policy_loss.png", "policy_loss", "Policy loss"),
            ("rl_value_loss.png", "value_loss", "Value loss")):
        fig, ax = plt.subplots(figsize=(9, 5))
        for arm, history in histories.items():
            points = [(int(row["update"]), float(row[field])) for row in history if field in row]
            if len(points) >= 2:
                ax.plot([x for x, _ in points], [y for _, y in points], label=arm)
        if ax.lines:
            ax.set(xlabel="update", ylabel=label, title=f"Full-train {label}")
            ax.legend()
            ax.grid(alpha=0.25)
            figure_paths[field] = plots / filename
            fig.savefig(figure_paths[field], dpi=150, bbox_inches="tight")
        plt.close(fig)
    report = {"status": "SUCCEEDED", "scope": "official_train_only_in_sample",
              "n_train_qids": len(bundle.qids), "n_reader_states": coverage["n_states"],
              "reader_training_loss": None, "official_validation_loaded": False,
              "official_test_loaded": False, "reader_cache_sha256": sha256_file(out / CACHE),
              "teacher": read_json(out / TEACHER_PROVENANCE),
              "laya": {arm: read_json(out / "laya" / f"{arm}.manifest.json")
                       for arm in ("E04_ce", "E05_rlcd")},
              "reinforce": read_json(out / "reinforce.json"),
              "plots": {key: str(path) for key, path in figure_paths.items()}}
    atomic_write_json(report_path, report)
    return {"report": str(report_path), "plots": report["plots"],
            "n_train_qids": len(bundle.qids)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "warm", "status", "teacher",
                                          "laya", "reinforce", "report"))
    parser.add_argument("--out", type=Path, default=ROOT)
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--max-requests", type=int, default=30000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--laya-base", type=Path, default=LAYA_BASE)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-updates", type=int, default=100)
    args = parser.parse_args()
    if args.stage == "prepare":
        result = prepare(args.out, train_path=args.train, corpus_path=args.corpus)
    elif args.stage == "warm":
        result = warm(args.out, train_path=args.train, corpus_path=args.corpus,
                      max_requests=args.max_requests, workers=args.workers,
                      batch_size=args.batch_size)
    elif args.stage == "teacher":
        result = make_teacher(args.out, base=args.laya_base, train_path=args.train,
                              corpus_path=args.corpus)
    elif args.stage == "laya":
        result = train_laya(args.out, base=args.laya_base, train_path=args.train,
                            corpus_path=args.corpus, epochs=args.epochs)
    elif args.stage == "reinforce":
        result = train_policy(args.out, train_path=args.train, corpus_path=args.corpus,
                              max_updates=args.max_updates)
    elif args.stage == "report":
        result = write_report(args.out, train_path=args.train, corpus_path=args.corpus)
    else:
        qids, questions, corpus, rankings = _inputs(args.out, args.train, args.corpus)
        identity = build_identity(corpus_path=args.corpus, rankings_path=args.out / RANKINGS,
                                  provider="opencode-go")
        reader = CloudReader(args.out / CACHE, identity=identity, max_requests=args.max_requests)
        result = _coverage(qids, questions, corpus, rankings, reader.cache, EpisodeBudget())
        result = {k: v for k, v in result.items() if k not in
                  ("complete_qids", "incomplete", "identity")}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
