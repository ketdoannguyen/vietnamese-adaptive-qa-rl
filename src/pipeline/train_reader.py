"""P3 CLI: fine-tune the Vietnamese reader on train_strict, tune the null threshold on val.

    python -m src.pipeline.train_reader --profile small --overfit-smoke
    python -m src.pipeline.train_reader --profile small

``--overfit-smoke`` is the acceptance gate of `03_...md` §8.2 and runs first: 32 training
windows must show a real loss decrease, otherwise the bug is in the data/feature path and RL
must not start. It is also run automatically (once) before the real fine-tune unless a
passing gate result already exists for the same config.

Training data is ``train_strict`` only. The null threshold is swept on ``val_strict``, never
on test, and is chosen feasibility-first (§10: at least one non-empty span with valid
provenance) and then by ``f1_answerable + no_answer_accuracy`` — overall EM cannot decide it,
because on an impossible-heavy val set the always-null threshold wins it (D008). A sweep with
no feasible point fails the run instead of freezing an abstain-always reader. The reader is
evaluated in two conditions, both reported:

* **oracle-context** — the reader is given the gold passage (answerable) or the source passage
  in which the answer is absent (impossible);
* **retrieved-context** — the reader is given only the documents the agent can actually
  retrieve within its search budget, i.e. the union of the reachable states of the retrieval
  tree. If P2's rankings for the split are missing this condition is reported ``NOT_RUN``
  with the reason, never faked.

Artifacts (all under ``outputs/models/qa_reader/`` unless noted, registered in
``outputs/runs/index.jsonl`` with SHA256):

* ``model.safetensors`` + tokenizer, ``training_args.json``, ``model_card.md``, ``checksums.json``
* ``overfit_smoke.json`` — the gate result
* ``reader_inference.json`` — the frozen inference contract (threshold + hashes) that the QA
  state cache and the RL environment read
* ``reader_eval.json`` — threshold sweep (per-row metrics + objective/constraint flags), the
  selection audit, oracle/retrieved metrics, acceptance checks
* ``reader_predictions_val.parquet`` — per (condition, qid, doc) decoded candidates
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from src.data import artifacts
from src.qa.features import build_feature_set
from src.qa.infer import PairInput, ReaderInference, inference_config_sha, use_fp16
from src.qa.postprocess import apply_threshold, best_prediction_per_question, \
    choose_best_threshold, selection_report, sweep_threshold
from src.qa.reader_eval import evaluate_predictions, gold_index, threshold_grid
from src.qa.train_reader import load_model, load_tokenizer, overfit_smoke, reader_model_hash, \
    save_model_card, tokenizer_hash, train
from src.retrieval.states import reachable_states, retrieved_doc_ids
from src.utils.config import add_profile_args, config_from_args, profile_limit, resolve, \
    snapshot_config
from src.utils.hashing import sha256_file, sha256_json
from src.utils.io import atomic_write_json, read_jsonl, write_parquet
from src.utils.reproducibility import capture_library_versions
from src.utils.runs import RunRegistry, now_iso

READER_DIR = "qa_reader"
GATE_DIR = "_gate"
GATE_FILE = "overfit_smoke.json"
INFERENCE_FILE = "reader_inference.json"
EVAL_FILE = "reader_eval.json"
PREDICTIONS_FILE = "reader_predictions_val.parquet"
CHECKSUMS_FILE = "checksums.json"
#: The gate needs 32 windows, which a smoke-scale cap cannot reliably supply, so the gate
#: always sees at least this many train questions (still train_strict).
GATE_MIN_QUESTIONS = 64
CACHE_VERSION = "v1"


def use_bf16(reader_cfg: Mapping[str, Any]) -> bool:
    """bf16 for reader *training* when the profile asks for it on a CUDA device.

    Mirrors :func:`src.qa.infer.use_fp16` (config opt-in + CUDA check) but applies to training
    only: unlike fp16 it needs no loss scaling, so it cannot burn steps on inf/nan gradients
    (measured on this L4: fp16 gate/training logs carry ``grad_norm: inf``). Inference keeps
    the fp16 autocast path in :mod:`src.qa.infer`, so ``reader.fp16`` stays as the profile
    set it for the eval/cache stages.
    """
    return bool(reader_cfg.get("bf16")) and torch.cuda.is_available()


# ------------------------------------------------------------------------------ paths

def reader_dir(cfg: Mapping[str, Any]) -> Path:
    return resolve(cfg, "models") / READER_DIR


def gate_dir(cfg: Mapping[str, Any]) -> Path:
    return reader_dir(cfg) / GATE_DIR


def rankings_path(cfg: Mapping[str, Any], split: str) -> Path:
    return resolve(cfg, "data_cache") / "retrieval" / f"{split}_rankings_{CACHE_VERSION}.jsonl"


# ----------------------------------------------------------------------------- inputs

def load_rows(cfg: Mapping[str, Any], split: str, *, profile: str,
              limit_override: int | None = None
              ) -> tuple[list[dict], list[dict[str, str]], dict[str, str]]:
    """Reader rows for one split, with the split's documents."""
    processed = resolve(cfg, "data_processed")
    docs = artifacts.read_docs(processed, split)
    limit = limit_override if limit_override is not None else profile_limit(cfg, profile)
    rows, dropped = artifacts.reader_rows(artifacts.read_questions(processed, split),
                                          limit=limit, docs=docs)
    return rows, dropped, docs


def oracle_pairs(rows: Sequence[Mapping[str, Any]], docs: Mapping[str, str]) -> list[PairInput]:
    return [PairInput(qid=row["qid"], doc_id=row["gold_doc_id"], question=row["question"],
                      context=docs[row["gold_doc_id"]])
            for row in rows if row["gold_doc_id"] in docs]


def retrieved_pairs(cfg: Mapping[str, Any], split: str, rows: Sequence[Mapping[str, Any]],
                    docs: Mapping[str, str]) -> tuple[list[PairInput], str | None]:
    """Pairs for the retrieved-context condition, or a NOT_RUN reason."""
    path = rankings_path(cfg, split)
    if not path.exists():
        return [], f"P2 rankings missing ({path}); run build_retrieval first"
    rankings = {row["qid"]: row["rankings"] for row in read_jsonl(path)}
    retrieval = cfg["retrieval"]
    pairs: list[PairInput] = []
    for row in rows:
        qid = row["qid"]
        if qid not in rankings:
            continue
        states = reachable_states(
            rankings[qid], bootstrap_tool=retrieval["initial_bootstrap"],
            bootstrap_passages=int(retrieval["bootstrap_passages"]),
            passage_per_search_action=int(retrieval["passage_per_search_action"]),
            max_optional_search=int(retrieval["optional_searches"]))
        for doc_id in retrieved_doc_ids(states):
            if doc_id in docs:
                pairs.append(PairInput(qid=qid, doc_id=doc_id, question=row["question"],
                                       context=docs[doc_id]))
    if not pairs:
        return [], f"no {split} question had a usable ranking in {path}"
    return pairs, None


# ------------------------------------------------------------------------------- eval

def evaluate_condition(spans: Mapping[tuple[str, str], Any], *, threshold: float,
                       model_hash: str, context_by_doc: Mapping[str, str],
                       gold: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    predictions = apply_threshold(spans.values(), threshold=threshold,
                                  reader_model_hash=model_hash, context_by_doc=context_by_doc)
    return evaluate_predictions(best_prediction_per_question(predictions.values()), gold)


def prediction_rows(spans: Mapping[tuple[str, str], Any], *, condition: str, threshold: float,
                    model_hash: str, context_by_doc: Mapping[str, str]) -> list[dict[str, Any]]:
    """One row per (condition, qid, doc): the candidate plus the decision at ``threshold``."""
    predictions = apply_threshold(spans.values(), threshold=threshold,
                                  reader_model_hash=model_hash, context_by_doc=context_by_doc)
    chosen = best_prediction_per_question(predictions.values())
    chosen_keys = {(p.qid, p.doc_id) for p in chosen.values()}
    rows = []
    for key, prediction in sorted(predictions.items()):
        rows.append({"condition": condition, "qid": prediction.qid, "doc_id": prediction.doc_id,
                     "answer_text": prediction.answer_text, "start_char": prediction.start_char,
                     "end_char": prediction.end_char,
                     "non_null_score": prediction.non_null_score,
                     "null_score": prediction.null_score,
                     "null_margin": prediction.null_margin,
                     "provenance_ok": bool(prediction.provenance_ok),
                     "is_candidate": key in chosen_keys,
                     "threshold": float(threshold), "reader_model_hash": model_hash})
    return rows


def acceptance(metrics_oracle: Mapping[str, Any], spans: Mapping[tuple[str, str], Any],
               rows: Sequence[Mapping[str, Any]], *, threshold: float, model_hash: str,
               context_by_doc: Mapping[str, str], gate: Mapping[str, Any] | None) -> dict[str, Any]:
    """The four P3 acceptance checks of `03_...md` §10, each with its evidence."""
    predictions = apply_threshold(spans.values(), threshold=threshold,
                                  reader_model_hash=model_hash, context_by_doc=context_by_doc)
    with_span = [(key, p) for key, p in predictions.items()
                 if p.answer_text is not None and p.provenance_ok]
    impossible_qids = {row["qid"] for row in rows if row["is_impossible"]}
    null_on_impossible = sorted(qid for (qid, _), p in predictions.items()
                                if qid in impossible_qids and p.answer_text is None)
    return {
        "overfit_loss_decreased": gate.get("loss_decreased") if gate else None,
        "overfit_smoke": {"path": str(gate.get("path")) if gate else None,
                          "loss_first": gate.get("loss_first") if gate else None,
                          "loss_last": gate.get("loss_last") if gate else None},
        "non_empty_span_with_valid_offset": {
            "passed": bool(with_span),
            "example": {"qid": with_span[0][1].qid, "doc_id": with_span[0][1].doc_id,
                        "answer_text": with_span[0][1].answer_text,
                        "start_char": with_span[0][1].start_char,
                        "end_char": with_span[0][1].end_char} if with_span else None},
        "impossible_returns_null": {"passed": bool(null_on_impossible),
                                    "n_null_on_impossible": len(null_on_impossible),
                                    "example_qid": null_on_impossible[0] if null_on_impossible else None},
        "val_metrics_saved": {"passed": True, "n_val_questions": int(metrics_oracle.get("n", 0))},
    }


def require_acceptance(checks: Mapping[str, Any], *, threshold: float) -> None:
    """D008 hard gate: a reader that never emits a span is a FAILED run, not a SUCCEEDED one.

    `03_QA_READER_TRAINING_AND_CACHE.md` §10 requires at least one prediction with a
    non-empty span and a valid source offset; before D008 the check was recorded as
    ``passed: false`` while the run was still registered SUCCEEDED. Raising here (before
    training is registered as an artifact) makes the registry record FAILED.
    """
    if not checks["non_empty_span_with_valid_offset"]["passed"]:
        raise RuntimeError(
            f"P3 acceptance failed: at the selected threshold {threshold} no val prediction "
            "has a non-empty span with a valid source offset (`03_...md` §10) — the reader "
            "would answer nothing, so the run must not be accepted (D008)")


# -------------------------------------------------------------------------------- run

def run(*, cfg: dict, profile: str, force: bool = False, overfit_smoke_only: bool = False,
        skip_gate: bool = False, registry: RunRegistry | None = None) -> dict[str, Any]:
    reader_cfg = cfg["reader"]
    # Training precision: bf16 wins when the profile opts in (see use_bf16); fp16 and bf16
    # are mutually exclusive in TrainingArguments. Inference precision is untouched.
    train_bf16 = use_bf16(reader_cfg)
    train_fp16 = use_fp16(reader_cfg) and not train_bf16
    out_dir = reader_dir(cfg)
    run_id = f"p3-reader-{profile}-{now_iso().replace(':', '')}"
    registry = registry or RunRegistry(resolve(cfg, "runs") / "index.jsonl")
    registry.append(run_id=run_id, stage="P3_reader", status="RUNNING", profile=profile,
                    overfit_smoke_only=overfit_smoke_only, force=force)
    artifacts_written: list[Path] = []
    try:
        if not overfit_smoke_only and not force and (out_dir / "model.safetensors").exists():
            raise RuntimeError(f"{out_dir} already holds a trained reader; the reader is frozen "
                               "after checkpoint selection — pass --force to retrain")
        base_model, revision = reader_cfg["base_model"], reader_cfg.get("revision")
        tokenizer = load_tokenizer(base_model, revision)
        model = load_model(base_model, revision)
        commit = getattr(model.config, "_commit_hash", None)
        max_length = int(reader_cfg["max_length"])
        doc_stride = int(reader_cfg["doc_stride"])

        gate_summary: dict[str, Any] | None = None
        gate_path = out_dir / GATE_FILE
        if overfit_smoke_only or not skip_gate:
            # The gate is a fixed-size diagnostic so its result is comparable across
            # profiles: 64 train_strict questions (>= the 32 windows the gate needs).
            gate_rows, _, docs_gate = load_rows(cfg, "train_strict", profile=profile,
                                                limit_override=GATE_MIN_QUESTIONS)
            gate_features = build_feature_set(gate_rows, docs_gate, tokenizer,
                                              max_length=max_length, doc_stride=doc_stride)
            gate_summary = overfit_smoke(model=model, tokenizer=tokenizer,
                                         rows=gate_features.rows(), out_dir=gate_dir(cfg),
                                         steps=40, lr=5e-5, batch_size=4,
                                         fp16=train_fp16, bf16=train_bf16)
            gate_summary["path"] = str(gate_path)
            gate_summary["profile"] = profile
            atomic_write_json(gate_path, gate_summary)
            artifacts_written.append(gate_path)
            if overfit_smoke_only:
                registry.record_artifact(run_id, "P3_reader_overfit_smoke",
                                         [gate_path, gate_dir(cfg)], profile=profile)
                return {"run_id": run_id, "reader_dir": str(out_dir),
                        "overfit_smoke": gate_summary, "artifacts": [str(gate_path)]}
            if not gate_summary["loss_decreased"]:
                raise RuntimeError(
                    "the 32-example overfit gate did not decrease the loss "
                    f"({gate_summary['loss_first']} -> {gate_summary['loss_last']}); "
                    "fix the data/feature path before training the reader (`03_...md` §8.2)")

        train_rows, dropped, docs_train = load_rows(cfg, "train_strict", profile=profile)
        if not train_rows:
            raise ValueError("no usable train_strict rows for the reader")
        features = build_feature_set(train_rows, docs_train, tokenizer,
                                     max_length=max_length, doc_stride=doc_stride)
        train_summary = train(model=model, tokenizer=tokenizer, rows=features.rows(),
                              out_dir=out_dir, epochs=int(reader_cfg["epochs"]),
                              lr=float(reader_cfg["lr"]), batch_size=int(reader_cfg["batch_size"]),
                              grad_accum=int(reader_cfg["grad_accum"]),
                              warmup_ratio=float(reader_cfg["warmup_ratio"]),
                              weight_decay=float(reader_cfg["weight_decay"]),
                              fp16=train_fp16, bf16=train_bf16,
                              seed=int(cfg["data"]["scenario_seed"]))
        train_summary.update({"base_model": base_model, "base_model_commit": commit,
                              "profile": profile, "train_rows": len(train_rows),
                              "dropped_rows": dropped, "features": len(features),
                              "skipped_no_visible_span": features.skipped_no_visible_span})
        atomic_write_json(out_dir / "training_args.json", train_summary)

        model_hash = reader_model_hash(out_dir)
        tokenizer_sha = tokenizer_hash(tokenizer)
        evaluation = _evaluate(cfg, profile=profile, out_dir=out_dir, model_hash=model_hash,
                              tokenizer_sha=tokenizer_sha, gate_summary=gate_summary,
                              extra={"train": {"rows": len(train_rows), "features": len(features),
                                               "final_loss": train_summary.get("final_loss"),
                                               "dropped_rows": len(dropped)}})
        artifacts_written += evaluation.pop("paths")
        save_model_card(out_dir, base_model=base_model, revision=commit,
                        tokenizer_sha=tokenizer_sha, summary=train_summary,
                        metrics=evaluation["oracle_context"])
        checksums = _checksums(out_dir)
        atomic_write_json(out_dir / CHECKSUMS_FILE, checksums)
    except Exception as exc:
        registry.append(run_id=run_id, stage="P3_reader", status="FAILED", profile=profile,
                        error=f"{type(exc).__name__}: {exc}")
        raise

    artifacts_written += [out_dir / "training_args.json", out_dir / "model_card.md",
                          out_dir / CHECKSUMS_FILE, out_dir / "model.safetensors"]
    registry.record_artifact(run_id, "P3_reader", artifacts_written, profile=profile,
                             reader_model_hash=model_hash, tokenizer_sha=tokenizer_sha)
    return {"run_id": run_id, "reader_dir": str(out_dir), "reader_model_hash": model_hash,
            "tokenizer_sha": tokenizer_sha, "train": train_summary, "eval": evaluation,
            "artifacts": [str(p) for p in artifacts_written]}


def _evaluate(cfg: dict, *, profile: str, out_dir: Path, model_hash: str, tokenizer_sha: str,
              gate_summary: Mapping[str, Any] | None, extra: Mapping[str, Any]) -> dict[str, Any]:
    """Threshold sweep on val (oracle context), retrieved-context metrics, acceptance."""
    reader_cfg = cfg["reader"]
    val_rows, dropped, docs_val = load_rows(cfg, "val_strict", profile=profile)
    if not val_rows:
        raise ValueError("no usable val_strict rows to evaluate on")
    gold = gold_index(val_rows)
    inference = ReaderInference.load(
        out_dir, max_length=int(reader_cfg["max_length"]), doc_stride=int(reader_cfg["doc_stride"]),
        max_answer_length=int(reader_cfg["max_answer_length"]),
        batch_size=int(reader_cfg["inference_batch_size"]), fp16=use_fp16(reader_cfg))

    oracle_spans = inference.predict(oracle_pairs(val_rows, docs_val))
    if not oracle_spans:
        raise ValueError("no val pair could be built: every val gold document is missing")
    oracle_contexts = dict(docs_val)
    grid = reader_cfg["threshold_sweep"]
    sweep = sweep_threshold(oracle_spans, gold,
                            threshold_grid(float(grid["start"]), float(grid["stop"]),
                                           float(grid["step"])),
                            reader_model_hash=model_hash, context_by_doc=oracle_contexts,
                            evaluate=evaluate_predictions)
    try:
        best = choose_best_threshold(sweep)
    except ValueError as exc:
        # Unsatisfiable sweep (no threshold answers anything): the sweep is the evidence for
        # the failure, so keep it on disk. reader_inference.json is written last and is
        # therefore absent, so no downstream stage can consume this reader.
        atomic_write_json(out_dir / EVAL_FILE, {
            "status": "SELECTION_FAILED", "reason": str(exc),
            "reader_model_hash": model_hash, "tokenizer_sha": tokenizer_sha,
            "threshold_sweep": sweep, "profile": profile, **dict(extra)})
        raise
    threshold = float(best["threshold"])
    selection = selection_report(sweep)

    rows = prediction_rows(oracle_spans, condition="oracle_context", threshold=threshold,
                           model_hash=model_hash, context_by_doc=oracle_contexts)
    retrieved_spans, not_run = retrieved_pairs(cfg, "val_strict", val_rows, docs_val)
    if not_run is None:
        retrieved_spans = inference.predict(retrieved_spans)
        rows += prediction_rows(retrieved_spans, condition="retrieved_context",
                                threshold=threshold, model_hash=model_hash,
                                context_by_doc={doc_id: docs_val[doc_id] for doc_id in docs_val})
        retrieved_metrics = evaluate_condition(
            retrieved_spans, threshold=threshold, model_hash=model_hash,
            context_by_doc={doc_id: docs_val[doc_id] for doc_id in docs_val}, gold=gold)
    else:
        retrieved_metrics = {"status": "NOT_RUN", "reason": not_run}

    paths = [out_dir / INFERENCE_FILE, out_dir / EVAL_FILE, out_dir / PREDICTIONS_FILE]
    predictions_meta = write_parquet(out_dir / PREDICTIONS_FILE, rows)
    checks = acceptance(best, oracle_spans, val_rows, threshold=threshold,
                        model_hash=model_hash, context_by_doc=oracle_contexts,
                        gate=gate_summary)
    evaluation = {
        "reader_model_hash": model_hash, "tokenizer_sha": tokenizer_sha,
        "null_score_diff_threshold": threshold,
        "threshold_sweep": sweep,
        "threshold_selection": selection,
        "oracle_context": best,
        "retrieved_context": retrieved_metrics,
        "acceptance": checks,
        "val_rows": len(val_rows), "val_dropped_rows": dropped,
        "predictions": predictions_meta, "library_versions": capture_library_versions(),
        "paths": paths,
        **dict(extra),
    }
    atomic_write_json(out_dir / EVAL_FILE, {k: v for k, v in evaluation.items()
                                            if k != "paths"})
    require_acceptance(checks, threshold=threshold)
    # The frozen contract is written only after the acceptance gate, so a rejected reader
    # never leaves a reader_inference.json for P4 to consume.
    atomic_write_json(out_dir / INFERENCE_FILE, {
        "reader_model_hash": model_hash, "tokenizer_sha": tokenizer_sha,
        "null_score_diff_threshold": threshold,
        "max_length": int(reader_cfg["max_length"]), "doc_stride": int(reader_cfg["doc_stride"]),
        "max_answer_length": int(reader_cfg["max_answer_length"]),
        "inference_config_sha": inference_config_sha(reader_cfg, threshold=threshold,
                                                     model_hash=model_hash),
        "profile": profile,
        "tuned_on": "val_strict", "sweep_points": len(sweep),
    })
    return evaluation


def _checksums(out_dir: Path) -> dict[str, str]:
    names = ["model.safetensors", "config.json", "tokenizer_config.json", "training_args.json",
             INFERENCE_FILE, EVAL_FILE, PREDICTIONS_FILE, GATE_FILE, "model_card.md"]
    out: dict[str, str] = {}
    for name in names:
        path = out_dir / name
        if path.exists():
            out[name] = sha256_file(path)
    return {"files": out, "config_sha": sha256_json(out)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.pipeline.train_reader",
                                     description="P3: fine-tune the reader and tune its null threshold")
    add_profile_args(parser)
    parser.add_argument("--overfit-smoke", action="store_true",
                        help="run only the 32-example loss-decrease gate")
    parser.add_argument("--skip-gate", action="store_true",
                        help="skip the automatic overfit gate before the real run")
    parser.add_argument("--force", action="store_true", help="retrain even if a reader exists")
    parser.add_argument("--json", action="store_true", help="print the full summary as JSON")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    profile = args.profile or cfg["data"]["profile"]
    cfg["data"]["profile"] = profile
    snapshot_config(cfg, resolve(cfg, "manifests") / f"P3_reader_{profile}_config.yaml")
    summary = run(cfg=cfg, profile=profile, force=args.force,
                  overfit_smoke_only=args.overfit_smoke, skip_gate=args.skip_gate)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.overfit_smoke:
        gate = summary["overfit_smoke"]
        print(f"overfit gate: loss {gate['loss_first']} -> {gate['loss_last']} "
              f"decreased={gate['loss_decreased']} ({gate['path']})")
        return 0
    evaluation = summary["eval"]
    oracle = evaluation["oracle_context"]
    print(f"run_id={summary['run_id']} reader_model_hash={summary['reader_model_hash'][:16]}")
    print(f"  train: {summary['train']['examples']} windows, final loss "
          f"{summary['train']['final_loss']}")
    print(f"  threshold {evaluation['null_score_diff_threshold']}: "
          f"answered(answerable)={oracle['answered_answerable']}/{oracle['n_answerable']} "
          f"F1(answerable)={oracle['f1_answerable']:.4f} "
          f"no-answer acc={oracle['no_answer_accuracy']:.4f} "
          f"false-answer rate={oracle['false_answer_rate']:.4f} "
          f"EM(answerable)={oracle['em_answerable']:.4f} "
          f"EM(overall)={oracle['em_overall']:.4f}")
    selection = evaluation["threshold_selection"]
    print(f"  selection: {selection['objective']}; feasible thresholds "
          f"{selection['n_feasible_thresholds']}/{selection['n_thresholds']} "
          f"(constraint_binding={selection['constraint_binding']}, "
          f"objective-only best {selection['objective_only_threshold']})")
    retrieved = evaluation["retrieved_context"]
    print(f"  retrieved-context: " + (
        f"EM(overall)={retrieved['em_overall']:.4f} EM(answerable)={retrieved['em_answerable']:.4f}"
        if "em_overall" in retrieved else f"NOT_RUN ({retrieved['reason']})"))
    print(f"  reader dir: {summary['reader_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
