"""Stage the VixSTORY train300/dev200 pilot without opening official validation for tuning.

Run ``teacher`` after the selected reader's train/dev cache is warm, then ``calibrate``;
train E04/E05 via ``scripts.vixstory_pilot_laya`` on the emitted teacher JSONL and
train E06/E07 with ``reinforce``. ``dev-eval`` verifies all arms on train-dev200.
Only after locking those outputs, ``validation-eval`` reads official validation once.
Every stage refuses an existing result; no stage trains on validation or official test.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

from scripts.vixstory_pilot_cache import PILOT_DIR, CACHE_NAME, MANIFEST_PATH, RANKINGS_PATH
from scripts.vixstory_reader_eval import CORPUS_PATH, SNAP, build_identity
from src.pipeline.vixstory_worlds import assemble_pilot_split, manifest_split_qids
from src.rl.vixstory_env import VixStoryEnvConfig
from src.utils.hashing import sha256_file, sha256_json
from src.utils.io import atomic_write_json, read_json

STAGES = ("teacher", "calibrate", "reinforce", "dev-eval", "validation-eval")
SPLIT_COUNTS = (300, 200, 246)


def guard_pilot_splits(splits: Mapping[str, Sequence[str]], *,
                       expected: tuple[int, int, int] = SPLIT_COUNTS,
                       require_validation: bool = True) -> None:
    roles = ("train", "dev", "validation") if require_validation else ("train", "dev")
    for role, n in zip(roles, expected):
        if role not in splits:
            raise ValueError(f"pilot manifest has no {role} split")
        qids = splits[role]
        if len(qids) != n or len(set(qids)) != n:
            raise ValueError(f"{role}: expected {n} unique QIDs, got {len(qids)}")
    present = tuple(role for role in ("train", "dev", "validation") if role in splits)
    if any(set(splits[a]) & set(splits[b]) for i, a in enumerate(present)
           for b in present[i + 1:]):
        raise ValueError("pilot split overlap: train/dev/validation QIDs must be disjoint")


def collect_teacher_inputs(bundle, config: VixStoryEnvConfig) -> dict:
    """Private references enter only train teacher targets, never reader or policy inputs."""
    from src.pipeline.vixstory_teacher import enumerate_states

    questions, references, rankings, answers = {}, {}, {}, {}
    docs = bundle.worlds[0].corpus
    for world in bundle.worlds:
        questions[world.qid] = world.question
        references[world.qid] = world.private.answer_texts[0]
        rankings[world.qid] = world.rankings
        for state, _history in enumerate_states(
                world.rankings, bootstrap_tool=config.bootstrap_tool,
                bootstrap_passages=config.bootstrap_passages,
                passage_per_search_action=config.passage_per_search_action,
                max_optional_search=config.max_optional_search):
            prediction = world.reader(state)
            answers[world.qid, state] = (prediction.text, prediction.cited_ids,
                                         prediction.citation_valid)
    if len(answers) != bundle.n_states:
        raise ValueError("teacher state count differs from the identity-bound reader cache")
    return {"questions": questions, "references": references, "rankings": rankings,
            "docs": docs, "reader_answers": answers}


def write_teacher(bundle, config: VixStoryEnvConfig, path: Path, *, tokenizer,
                  max_len: int = 768, head_max_len: int = 192) -> dict:
    from src.pipeline.vixstory_teacher import teacher_rows

    if bundle.role != "train":
        raise ValueError("teacher must only use the train300 split")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite teacher artifact: {path}")
    from src.decision import vixstory_state_text
    from src.decision.vixstory_state_text import STATE_TEXT_VERSION

    tokenizer_contract = json.loads(tokenizer.backend_tokenizer.to_str())
    tokenizer_contract.pop("truncation", None)
    data = collect_teacher_inputs(bundle, config)
    rows = teacher_rows(bundle.qids, **data, search_cost=config.search_cost,
                        tau=config.f1_threshold, gate_citation=config.citation_gate,
                        bootstrap_passages=config.bootstrap_passages,
                        passage_per_search_action=config.passage_per_search_action,
                        max_optional_search=config.max_optional_search,
                        tokenizer=tokenizer, max_len=max_len, head_max_len=head_max_len)
    if len(rows) != bundle.n_states or {r["meta"]["qid"] for r in rows} != set(bundle.qids):
        raise ValueError("teacher produced incomplete train300 coverage")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"n_qids": len(bundle.qids), "n_states": len(rows), "sha256": sha256_file(path),
            "reader_identity": bundle.identity.as_dict(), "config": asdict(config),
            "state_text_version": STATE_TEXT_VERSION,
            "state_text_sha256": sha256_file(Path(vixstory_state_text.__file__)),
            "max_len": max_len, "head_max_len": head_max_len,
            "tokenizer_contract_sha256": sha256_json(tokenizer_contract)}


def require_complete_eval(report: Mapping) -> None:
    if not report.get("success") or report.get("missing_mandatory"):
        raise RuntimeError(f"pilot evaluation missing mandatory arms: "
                           f"{report.get('missing_mandatory')}")


def require_paired_eval(report: Mapping) -> None:
    seeds = report.get("seeds", {})
    if set(seeds) != {"42", "43"}:
        raise RuntimeError("pilot evaluation requires actual seeds 42 and 43")
    for seed in ("42", "43"):
        if seeds[seed].get("seed") != int(seed):
            raise RuntimeError(f"pilot evaluation seed {seed} is missing or mislabeled")
        require_complete_eval(seeds[seed])
    if (seeds["42"].get("n_qids") != seeds["43"].get("n_qids")
            or seeds["42"].get("qids_sha256") != seeds["43"].get("qids_sha256")):
        raise RuntimeError("paired pilot evaluation has inconsistent dev QID coverage")
    if seeds["42"].get("n_qids") != SPLIT_COUNTS[1]:
        raise RuntimeError(f"paired pilot evaluation requires all {SPLIT_COUNTS[1]} dev questions")
    require_complete_eval(report)


def evaluation_identity(paths: Mapping, config: VixStoryEnvConfig, *,
                        require_validation: bool = False) -> dict:
    """Hash dev-evaluation inputs; validation scoring also requires its frozen QIDs."""
    from src.decision.vixstory_state_text import STATE_TEXT_VERSION

    root = paths["pilot"]
    files = {
        "manifest.json": paths["manifest"], "rankings.json": paths["rankings"],
        "reader-cache.json": paths["cache"], "corpus.parquet": paths["corpus"],
        "teacher/train300.jsonl": root / "teacher" / "train300.jsonl",
        "teacher/train300.provenance.json": root / "teacher" / "train300.provenance.json",
        "calibration.json": root / "calibration.json", "reinforce.json": root / "reinforce.json",
        "E03/threshold_selection.json": root / "E03" / "threshold_selection.json",
        "E03/calibrator_model.json": root / "E03" / "calibrator_model.json",
    }
    for arm in ("E04_ce", "E05_rlcd"):
        directory = root / "laya" / arm
        files[f"laya/{arm}.manifest.json"] = root / "laya" / f"{arm}.manifest.json"
        for name in ("model.safetensors", "rl_agent_config.json", "questions.json",
                     "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
                     "encoder/config.json"):
            files[f"laya/{arm}/{name}"] = directory / name
    for seed in (42, 43):
        for arm in ("E06_rl_rlcd", "E07_rl_ce"):
            relative = f"reinforce/seed-{seed}/{arm}"
            directory = root / relative
            for name in ("summary.json", "rl_config.json"):
                files[f"{relative}/{name}"] = directory / name
            for name in ("policy.safetensors", "policy.safetensors.config.json", "scaler.json",
                         "rl_config.json"):
                files[f"{relative}/best/{name}"] = directory / "best" / name
    for name, path in files.items():
        _require_file(path, name)
    splits = manifest_split_qids(read_json(paths["manifest"]))
    guard_pilot_splits(splits, require_validation=require_validation)
    source_root = Path(__file__).resolve().parents[1]
    sources = ("scripts/vixstory_pilot.py", "scripts/vixstory_reader_eval.py",
               "src/decision/vixstory_state_text.py", "src/decision/laya_adapter.py",
               "src/decision/schema.py", "src/pipeline/vixstory_arms.py",
               "src/pipeline/vixstory_calibration.py", "src/pipeline/vixstory_eval.py",
               "src/pipeline/vixstory_worlds.py", "src/rl/vixstory_env.py",
               "src/rl/vixstory_policy.py", "src/rl/reinforce.py", "src/rl/policy.py",
               "src/rl/features.py", "src/qa/metrics.py", "src/retrieval/states.py")
    return {"reader_identity": build_identity(corpus_path=paths["corpus"],
                                               rankings_path=paths["rankings"]).as_dict(),
            "config": asdict(config), "state_text_version": STATE_TEXT_VERSION,
            "source_files": {name: sha256_file(source_root / name) for name in sources},
            "qids_sha256": {role: sha256_json(list(qids)) for role, qids in splits.items()},
            "files": {name: sha256_file(path) for name, path in files.items()}}


def _paths(args) -> dict:
    pilot = args.pilot_dir
    return {"pilot": pilot, "manifest": args.manifest or MANIFEST_PATH,
            "rankings": args.rankings or RANKINGS_PATH,
            "cache": args.cache or pilot / CACHE_NAME,
            "corpus": args.corpus,
            "train_qa": args.train_qa, "validation_qa": args.validation_qa}


def _bundle(role: str, paths: Mapping, config: VixStoryEnvConfig):
    return assemble_pilot_split(
        role=role, manifest_path=paths["manifest"], corpus_path=paths["corpus"],
        qa_path=paths["validation_qa"] if role == "validation" else paths["train_qa"],
        rankings_path=paths["rankings"], cache_path=paths["cache"], config=config)


def _require_file(path: Path, what: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{what} is missing: {path}")


def _require_absent(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite completed artifact: {path}")


def _calibration(paths: Mapping, config: VixStoryEnvConfig, out: Path) -> dict:
    from src.pipeline.vixstory_calibration import run_vixstory_calibration
    from src.rl.vixstory_env import VixStoryEnv

    _require_file(out / "teacher" / "train300.jsonl", "train-only teacher")
    train, dev = _bundle("train", paths, config), _bundle("dev", paths, config)
    directory = out / "E03"
    _require_absent(directory)
    result = run_vixstory_calibration(VixStoryEnv(train.worlds, config), train_qids=train.qids,
                                      dev_env=VixStoryEnv(dev.worlds, config), dev_qids=dev.qids,
                                      save_dir=directory)
    if result["status"] != "SUCCEEDED":
        raise RuntimeError(f"E03 calibration cannot proceed: {result['blockers']}")
    atomic_write_json(out / "calibration.json", result)
    return result


def _laya_paths(paths: Mapping) -> dict[str, Path]:
    root = paths["pilot"] / "laya"
    return {"ce": root / "E04_ce", "rlcd": root / "E05_rlcd"}


def _reinforce(paths: Mapping, config: VixStoryEnvConfig, args) -> dict:
    import torch

    from src.pipeline.vixstory_arms import LayaPrior
    from src.rl.reinforce import RLConfig
    from src.rl.vixstory_env import VixStoryEnv
    from src.rl.vixstory_policy import train_vixstory_policy

    if not torch.cuda.is_available():
        raise RuntimeError("E06/E07 training requires local CUDA")
    ckpts = _laya_paths(paths)
    _require_file(paths["pilot"] / "calibration.json", "successful E03 calibration")
    for path in ckpts.values():
        _require_file(path / "model.safetensors", "trained Laya checkpoint")
    train, dev = _bundle("train", paths, config), _bundle("dev", paths, config)
    ce = LayaPrior(ckpts["ce"], device="cuda")
    rlcd = LayaPrior(ckpts["rlcd"], device="cuda")
    try:
        rl_config = RLConfig(max_updates=args.max_updates, batch_episodes=args.batch_episodes,
                             eval_every=args.eval_every, early_stop_patience=args.patience)
        train_env, dev_env = VixStoryEnv(train.worlds, config), VixStoryEnv(dev.worlds, config)
        results = {}
        for seed in (42, 43):
            pair = train_vixstory_policy(
                train_env, dev_env, rl_config, priors={"ce": ce, "rlcd": rlcd},
                train_qids=train.qids, dev_qids=dev.qids,
                arm_seeds={"ce": seed, "rlcd": seed}, fixed_updates=True,
                prior_refs={"ce": str(ckpts["ce"]), "rlcd": str(ckpts["rlcd"])},
                run_root=paths["pilot"] / "reinforce" / f"seed-{seed}", device="cuda")
            results[str(seed)] = {arm: value.to_dict() for arm, value in pair.items()}
        return results
    finally:
        ce.close()
        rlcd.close()


def _policies(paths: Mapping, config: VixStoryEnvConfig, seed: int):
    from src.pipeline.vixstory_arms import (
        CalibrationPolicy, LayaPolicy, LayaPrior, ReinforcePolicy, basic_policies,
    )
    from src.pipeline.vixstory_calibration import (
        VixStoryCorrectnessEstimator, artifact_identity, reward_criterion,
    )

    root = paths["pilot"]
    _require_file(root / "teacher" / "train300.jsonl", "train-only teacher")
    _require_file(root / "calibration.json", "successful E03 calibration")
    _require_file(root / "reinforce.json", "both REINFORCE training results")
    threshold = read_json(root / "E03" / "threshold_selection.json")
    criterion = reward_criterion(config)
    if threshold["identity"] != artifact_identity(criterion):
        raise ValueError("E03 threshold was fit for a different reward/budget")
    calibration = read_json(root / "calibration.json")
    if calibration["status"] != "SUCCEEDED" or calibration["identity"] != threshold["identity"]:
        raise ValueError("E03 calibration is incomplete or disagrees with the threshold")
    reinforce = read_json(root / "reinforce.json")
    if set(reinforce) != {"42", "43"} or any(
            set(reinforce[str(actual_seed)]) != {"ce", "rlcd"} or any(
                reinforce[str(actual_seed)][arm].get("seed") != actual_seed
                or reinforce[str(actual_seed)][arm].get("n_updates") != 100
                or reinforce[str(actual_seed)][arm].get("best_update") is None
                or reinforce[str(actual_seed)][arm].get("stopped_early") is not False
                for arm in ("ce", "rlcd")) for actual_seed in (42, 43)):
        raise ValueError("E06/E07 require two complete fixed-update paired seeds")
    estimator = VixStoryCorrectnessEstimator.load(
        root / "E03", expected_identity=threshold["identity"])
    laya = _laya_paths(paths)
    for path in laya.values():
        _require_file(path / "model.safetensors", "trained Laya checkpoint")
    rlcd, ce = LayaPrior(laya["rlcd"], device="cuda"), LayaPrior(laya["ce"], device="cuda")
    try:
        policies = basic_policies()
        policies.update({"E03": CalibrationPolicy(estimator, threshold["tau"]),
                         "E04": LayaPolicy(ce), "E05": LayaPolicy(rlcd),
                         "E06": ReinforcePolicy(root / "reinforce" / f"seed-{seed}" /
                                                "E06_rl_rlcd" / "best", rlcd),
                         "E07": ReinforcePolicy(root / "reinforce" / f"seed-{seed}" /
                                                "E07_rl_ce" / "best", ce)})
    except BaseException:
        ce.close()
        rlcd.close()
        raise
    return policies, (ce, rlcd)


def _eval(role: str, paths: Mapping, config: VixStoryEnvConfig, out: Path) -> dict:
    from src.pipeline.vixstory_eval import run_vixstory_eval
    from src.rl.vixstory_env import VixStoryEnv

    reports = {}
    for seed in (42, 43):
        policies, priors = _policies(paths, config, seed)
        try:
            bundle = _bundle(role, paths, config)
            report = run_vixstory_eval(env_factory=lambda: VixStoryEnv(bundle.worlds, config),
                                       policies=policies, qids=bundle.qids, seed=seed)
            require_complete_eval(report)
            reports[str(seed)] = report
        finally:
            for prior in priors:
                prior.close()
    result = {"success": True, "missing_mandatory": [], "seeds": reports,
              "reader_identity": bundle.identity.as_dict(), "config": asdict(config)}
    if role == "dev":
        result["evaluation_identity"] = evaluation_identity(paths, config)
    atomic_write_json(out, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--pilot-dir", type=Path, default=PILOT_DIR)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--rankings", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--train-qa", type=Path, default=SNAP / "data/train.parquet")
    parser.add_argument("--validation-qa", type=Path, default=SNAP / "data/validation.parquet")
    parser.add_argument("--max-updates", type=int, default=100)
    parser.add_argument("--batch-episodes", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--laya-base", type=Path,
                        help="local Laya base checkpoint for train-only teacher tokenizer")
    args = parser.parse_args(argv)
    paths = _paths(args)
    splits = manifest_split_qids(read_json(paths["manifest"]))
    guard_pilot_splits(splits, require_validation=args.stage == "validation-eval")
    config = VixStoryEnvConfig()
    out = paths["pilot"]
    if args.stage == "validation-eval":
        dev_report = read_json(out / "dev-eval.json")
        require_paired_eval(dev_report)
        if dev_report.get("evaluation_identity") is None:
            raise ValueError("validation requires a complete dev evaluation identity")
        if dev_report["evaluation_identity"] != evaluation_identity(paths, config):
            raise ValueError("validation identity differs from locked dev artifacts")
    target = {"teacher": out / "teacher" / "train300.jsonl",
              "calibrate": out / "calibration.json",
              "reinforce": out / "reinforce.json",
              "dev-eval": out / "dev-eval.json",
              "validation-eval": out / "validation-eval.json"}[args.stage]
    _require_absent(target)
    if args.stage == "teacher":
        _require_absent(out / "teacher" / "train300.provenance.json")
        if args.laya_base is None or not (args.laya_base / "tokenizer").is_dir():
            raise ValueError("teacher requires --laya-base with a local tokenizer directory")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.laya_base / "tokenizer",
                                                  local_files_only=True)
        bundle = _bundle("train", paths, config)
        result = write_teacher(bundle, config, target, tokenizer=tokenizer)
        atomic_write_json(out / "teacher" / "train300.provenance.json", result)
    elif args.stage == "calibrate":
        result = _calibration(paths, config, out)
    elif args.stage == "reinforce":
        result = _reinforce(paths, config, args)
        atomic_write_json(target, result)
    else:
        result = _eval("dev" if args.stage == "dev-eval" else "validation", paths,
                       config, target)
    print(json.dumps({"stage": args.stage, "result": result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
