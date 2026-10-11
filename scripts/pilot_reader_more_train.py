"""Matched reader-data pilot: repeat 1000 train positives vs 1000 new clean positives.

Both arms contain the same count of positive/negative training rows and steps; only
whether those 1000 positive examples are new differs. Null thresholds are selected
on locked val_tune only. This is not yet a new RL result.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from scripts.audit_rl_search_value import PROTOCOL, verify_protocol
from src.data import artifacts, loader
from src.pipeline import audit_split, train_reader, worlds
from src.qa.features import build_feature_set
from src.qa.infer import ReaderInference, use_fp16
from src.qa.metrics import f1_score
from src.qa.postprocess import apply_threshold, best_prediction_per_question, sweep_threshold, choose_best_threshold
from src.qa.reader_eval import evaluate_predictions, gold_index, threshold_grid
from src.qa.train_reader import load_model, load_tokenizer, reader_model_hash, tokenizer_hash, train, total_steps
from src.utils.config import load_config, resolve
from src.utils.hashing import sha256_file, sha256_json, stable_seed
from src.utils.io import read_yaml
from src.utils.reproducibility import seed_everything

ROOT = Path("outputs/experiments/2026-10-11-rl-bottleneck/reader-pilot-v4")
N_ADDITIONAL = 1000


def train_records(cfg: dict) -> tuple[list[dict], list[dict], dict[str, str], dict]:
    """Existing unflagged strict train + new answerable rows from the same strict group pool."""
    processed = resolve(cfg, "data_processed")
    emitted = artifacts.read_questions(processed, "train_strict")
    existing_qids = {r["qid"] for r in emitted}
    existing_groups = {r["title_group"] for r in emitted}
    excluded = set(worlds.load_ambiguous_qids(cfg, "train_strict"))
    assert len(emitted) == len(existing_qids) == 5000
    kept = [r for r in emitted if r["qid"] not in excluded]
    base_docs = artifacts.read_docs(processed, "train_strict")
    base, dropped = artifacts.reader_rows(kept, docs=base_docs)
    if dropped:
        raise ValueError(f"base training rows with invalid offsets: {dropped[:3]}")

    source = loader.load_source_manifest(resolve(cfg, "data_raw"))
    raw_train = loader.raw_split_paths(resolve(cfg, "data_raw"), source)["train"]
    raw_rows, _, _, failures = audit_split.normalize_raw_split(raw_train, "train", keep_rows=True)
    if failures:
        raise ValueError(f"malformed raw train row(s): {failures[:3]}")
    pool = [row for row in raw_rows if row.question.title_group in existing_groups]
    if len(pool) != 19795 or not existing_qids <= {r.question.qid for r in pool}:
        raise ValueError("frozen strict-train group assignment drifted")
    others = sorted((r for r in pool if r.question.qid not in existing_qids
                     and not r.question.private.is_impossible and not r.offset_unrecoverable),
                    key=lambda r: r.question.qid)
    extra = sorted(others, key=lambda row: (stable_seed(cfg["seed"], row.question.qid),
                                            row.question.qid))[:N_ADDITIONAL]
    if len(extra) != N_ADDITIONAL:
        raise ValueError("not enough verified additional answerable train questions")
    docs = dict(base_docs)
    for row in extra:
        docs.setdefault(row.document.doc_id, row.document.text)
    extended, invalid = artifacts.reader_rows([
        audit_split.question_row(row, "train_strict") for row in extra], docs=docs)
    if invalid or len(extended) != N_ADDITIONAL:
        raise ValueError(f"additional answerable records have unusable offsets: {invalid[:3]}")
    if set(r["qid"] for r in extended) & existing_qids:
        raise ValueError("additional train QIDs overlap emitted train QIDs")
    return base, extended, docs, {
        "raw_train_sha256": sha256_file(raw_train),
        "strict_emitted_train_sha256": sha256_file(artifacts.questions_path(processed, "train_strict")),
        "base_rows": len(base), "additional_positive_rows": len(extended),
        "flagged_train_rows_excluded": len(excluded & existing_qids),
        "selected_qids_sha256": sha256_json([r["qid"] for r in extended]),
        "strict_train_pool_verified": len(pool),
    }


def val_rows(cfg: dict, qids: list[str]) -> tuple[list[dict], dict[str, str]]:
    processed = resolve(cfg, "data_processed")
    gold = pq.read_table(str(artifacts.questions_path(processed, "val_strict")),
                         filters=[("qid", "in", qids)]).to_pylist()
    if len(gold) != len(qids) or {r["qid"] for r in gold} != set(qids):
        raise ValueError("only locked val_tune gold rows allowed")
    docs = artifacts.read_docs(processed, "val_strict")
    rows, dropped = artifacts.reader_rows(gold, docs=docs)
    if dropped or len(rows) != len(qids):
        raise ValueError(f"val_tune reader rows dropped: {dropped[:3]}")
    return rows, docs


def evaluate(model_dir: Path, cfg: dict, rows: list[dict], docs: dict[str, str]) -> dict:
    settings = cfg["reader"]
    inference = ReaderInference.load(
        model_dir, max_length=settings["max_length"], doc_stride=settings["doc_stride"],
        max_answer_length=settings["max_answer_length"],
        batch_size=settings["inference_batch_size"], fp16=use_fp16(settings))
    spans = inference.predict(train_reader.oracle_pairs(rows, docs))
    if len(spans) != len(rows) or {qid for qid, _ in spans} != {row["qid"] for row in rows}:
        raise ValueError("missing or unexpected val_tune gold-context predictions")
    gold = gold_index(rows)
    grid = settings["threshold_sweep"]
    sweep = sweep_threshold(spans, gold, threshold_grid(grid["start"], grid["stop"], grid["step"]),
                            reader_model_hash=inference.model_hash, context_by_doc=docs,
                            evaluate=evaluate_predictions)
    chosen = choose_best_threshold(sweep)
    threshold = float(chosen["threshold"])
    predictions = best_prediction_per_question(apply_threshold(
        spans.values(), threshold=threshold, reader_model_hash=inference.model_hash,
        context_by_doc=docs).values())
    answered_f1 = sum(
        pred.answer_text is not None and pred.provenance_ok
        and f1_score(pred.answer_text, list(gold[qid]["answer_texts"])) >= 0.8 - 1e-12
        for qid, pred in predictions.items() if not gold[qid]["is_impossible"])
    return {"reader_model_hash": inference.model_hash, "tokenizer_sha": inference.tokenizer_sha,
            "threshold": threshold, "val_tune_n": len(rows),
            "f1_ge_0_8_answerable": int(answered_f1),
            "f1_ge_0_8_rate_answerable": answered_f1 / chosen["n_answerable"],
            "metrics": chosen}


def main() -> None:
    result_path = ROOT / "result.json"
    if ROOT.exists():
        raise FileExistsError(f"refusing to overwrite any reader pilot artifacts: {ROOT}")
    cfg = load_config("medium", extra=read_yaml("configs/rl_audit_medium.yaml"))
    locked = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    tune, groups = verify_protocol(cfg, locked)
    original = artifacts.read_questions(resolve(cfg, "data_processed"), "train_strict")
    train_groups = {r["title_group"] for r in original}
    if train_groups & {groups[qid] for qid in tune} or train_groups & {groups[qid] for qid in locked["report_qids"]}:
        raise ValueError("train group overlaps validation")
    base, extra, train_docs, provenance = train_records(cfg)
    validation, val_docs = val_rows(cfg, tune)
    # Match row count, positive fraction, hyperparameters and training steps. Repeated
    # examples control extra update budget; the other arm introduces unique positives.
    # For the pilot use one epoch on both arms: compare data diversity, not extra epochs.
    repeat_pool = sorted((row for row in base if not row["is_impossible"]),
                         key=lambda row: (stable_seed(cfg["seed"], row["qid"]), row["qid"]))
    if len(repeat_pool) < N_ADDITIONAL:
        raise ValueError("not enough clean existing positives for repeat control")
    repeated = repeat_pool[:N_ADDITIONAL]
    provenance["repeated_positive_qids_sha256"] = sha256_json([r["qid"] for r in repeated])
    ROOT.mkdir(parents=True, exist_ok=False)
    output = {"protocol_sha256": sha256_file(PROTOCOL), "reward_spec_hash":
              sha256_json(cfg["reward"]), "train": provenance,
              "tune_qid_sha256": sha256_json(tune), "pilot_epochs_both_arms": 1,
              "historical_reader_epochs": cfg["reader"]["epochs"], "arms": {}}
    examples = (("repeat_1000_train_positives", base + repeated),
                ("add_1000_new_positives", base + extra))
    update_budget = total_steps(5604, batch_size=cfg["reader"]["batch_size"],
                                grad_accum=cfg["reader"]["grad_accum"], epochs=1)
    output["matched_update_budget"] = update_budget
    for name, records in examples:
        path = ROOT / name
        seed_everything(cfg["seed"], deterministic_torch=True)
        tokenizer = load_tokenizer(cfg["reader"]["base_model"], cfg["reader"].get("revision"))
        model = load_model(cfg["reader"]["base_model"], cfg["reader"].get("revision"))
        features = build_feature_set(records, train_docs, tokenizer,
                                     max_length=cfg["reader"]["max_length"],
                                     doc_stride=cfg["reader"]["doc_stride"])
        summary = train(model=model, tokenizer=tokenizer, rows=features.rows(),
                        out_dir=path, epochs=1, lr=cfg["reader"]["lr"],
                        batch_size=cfg["reader"]["batch_size"],
                        grad_accum=cfg["reader"]["grad_accum"],
                        warmup_ratio=cfg["reader"]["warmup_ratio"],
                        weight_decay=cfg["reader"]["weight_decay"],
                        bf16=bool(cfg["reader"].get("bf16")), seed=cfg["seed"],
                        max_steps=update_budget)
        if summary["global_step"] != update_budget:
            raise RuntimeError("reader arms did not reach matched update budget")
        output["arms"][name] = {"train_rows": len(records), "features": len(features),
                                "skipped_no_visible_span": features.skipped_no_visible_span,
                                "training_steps": summary["global_step"],
                                "train_runtime_s": summary["train_runtime_s"],
                                "reader_model_hash": reader_model_hash(path),
                                "tokenizer_sha": tokenizer_hash(tokenizer),
                                "evaluation": evaluate(path, cfg, validation, val_docs)}
        del model, tokenizer, features
        print(name, output["arms"][name]["evaluation"], flush=True)
        result_path.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
    print("reader pilot result:", result_path)


if __name__ == "__main__":
    main()
