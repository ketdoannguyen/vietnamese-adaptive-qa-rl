"""P6 pipeline tests: CLI wiring, split discipline, determinism, artifacts, loud failures.

Nothing here touches the P1/P2/P3 caches, a Laya checkpoint, the reader or a GPU: the
fixture patches :func:`src.pipeline.train_rl.build_worlds` to hand the stage
``tests/toy_world`` episodes. What is under test is the *pipeline* — which worlds the
trainer sees, which split fits the scaler and drives early stopping, that the published
checkpoint reproduces the policy that was measured, and that every failure is loud and
registered. The real-cache path (``build_worlds`` itself against P1/P2/P3) and reward
direction on the toy MDP ("training improves the return") are covered by the P3/P5 fixtures,
the real runs, and ``tests/test_policy_gradient.py``.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from src.decision.laya_adapter import checkpoint_model_hash
from src.decision.state_text import STATE_TEXT_VERSION
from src.pipeline import train_rl, worlds
from src.pipeline.cache_reader import state_cache_paths
from src.pipeline.worlds import DEFAULT_LAYA_SENTINEL, WorldBundle
from src.rl.env import AdaptiveQAEnv
from src.rl.features import FeatureExtractor, FeatureScaler
from src.rl.policy import PolicyNet, TorchScaler
from src.rl.reinforce import RLConfig, ReinforceTrainer
from src.rl.reward import RewardCalculator, RewardConfig
from src.utils.config import load_config, resolve
from src.utils.hashing import sha256_file
from src.utils.io import read_json, read_jsonl, write_yaml
from tests.toy_world import negative_world, positive_world


# ----------------------------------------------------------------------------- helpers

def _cfg(root: Path) -> dict:
    """A self-contained config: every path inside ``root`` so no test writes to the repo."""
    cfg = load_config(None)
    cfg["paths"] = {key: str(root / key) for key in cfg["paths"]}
    cfg["data"] = {**cfg["data"], "profile": "smoke"}
    cfg["laya"] = {**cfg["laya"], "frozen": "none"}
    cfg["rl"] = {**cfg["rl"], "max_updates": 4, "eval_every": 2, "eval_episodes": 4,
                 "batch_episodes": 2, "hidden_dim": 8, "lr": 0.01, "seed": 7,
                 "early_stop_patience": 1000, "max_optional_search": 1}
    cfg["retrieval"] = {**cfg["retrieval"], "initial_bootstrap": "bm25",
                        "bootstrap_passages": 1, "passage_per_search_action": 1,
                        "optional_searches": 1}
    return cfg


def _write_reader_meta(cfg: dict, split: str) -> dict:
    """The P3 meta sidecar ``train_rl._reader_meta`` reads (the parquet is never touched)."""
    parquet, meta_path = state_cache_paths(cfg, split)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"schema_version": "v1", "split": split,
            "reader_model_hash": "r" * 64, "tokenizer_sha": "t" * 64,
            "inference_config_sha": "i" * 64,
            "retrieval_budget": {"bootstrap_tool": "bm25", "bootstrap_passages": 1,
                                 "passage_per_search_action": 1, "optional_searches": 1}}
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return meta


def _fake_checkpoint(cfg: dict, regime: str) -> Path:
    """A minimal directory that passes ``checkpoint_model_hash`` without any model."""
    path = worlds.laya_checkpoint_dir(cfg, regime)
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.safetensors").write_bytes(b"weights")
    (path / "rl_agent_config.json").write_text("{}", encoding="utf-8")
    return path


def _policy_dir(cfg: dict) -> Path:
    return resolve(cfg, "models") / train_rl.POLICY_DIR


def _rows(cfg: dict) -> list[dict]:
    return list(read_jsonl(resolve(cfg, "runs") / "index.jsonl"))


@pytest.fixture()
def p6(tmp_path, monkeypatch):
    """Toy worlds in place of the real caches; returns (cfg, state, root)."""
    cfg = _cfg(tmp_path)
    for split in (train_rl.TRAIN_SPLIT, train_rl.VAL_SPLIT):
        _write_reader_meta(cfg, split)
    state: dict = {"train": [positive_world(), negative_world()],
                   "val": [positive_world(), negative_world()],
                   "ambiguous": frozenset(), "calls": []}
    # One shared callable for both bundles: the trainer holds a single prior, and
    # worlds.merge_laya_fns refuses two bundles that pre-warm the same qid separately.
    prior = lambda obs: np.full(4, 0.25)  # noqa: E731 - a fixture, not library code

    def fake_build_worlds(cfg_arg, split, *, limit=None, laya_regime=DEFAULT_LAYA_SENTINEL):
        state["calls"].append({"split": split, "limit": limit, "laya_regime": laya_regime})
        key = {train_rl.TRAIN_SPLIT: "train", train_rl.VAL_SPLIT: "val"}[split]
        ws = [replace(w, split=split) for w in state[key]]
        return WorldBundle(split=split, worlds=ws,
                           questions={w.qid: w.question for w in ws},
                           docs={k: v for w in ws for k, v in w.corpus.items()},
                           laya_fn=prior if laya_regime is not None else None,
                           ambiguous_qids=frozenset(state["ambiguous"]))

    monkeypatch.setattr(train_rl, "build_worlds", fake_build_worlds)
    return cfg, state, tmp_path


# ------------------------------------------------------------------- CLI + artifacts

def test_cli_run_writes_model_plots_report_and_registry(p6, capsys):
    cfg, state, root = p6
    overlay = root / "overlay.yaml"
    write_yaml(overlay, cfg)
    assert train_rl.main(["--config", str(overlay), "--seed", "3", "--limit", "5",
                          "--max-updates", "2", "--laya-regime", "none", "--json"]) == 0
    printed = capsys.readouterr().out
    summary = json.loads(printed[printed.index("{"):])  # the JSON is printed last

    # Worlds: both splits, the limit and the explicit no-prior ablation reach build_worlds.
    assert state["calls"] == [
        {"split": train_rl.TRAIN_SPLIT, "limit": 5, "laya_regime": None},
        {"split": train_rl.VAL_SPLIT, "limit": 5, "laya_regime": None}]
    assert summary["seed"] == 3 and summary["updates"] == 2
    assert summary["splits"]["train"]["worlds"] == 2
    assert summary["splits"]["val"]["worlds"] == 2
    assert summary["laya"] == {"regime": None, "checkpoint_ref": None, "model_hash": None,
                               "state_text_version": STATE_TEXT_VERSION}
    assert summary["training_is_real"]["params_changed"] is True
    assert summary["training_is_real"]["grad_norm_nonzero"] is True
    assert summary["on_policy"] == {"actions": "sampled from the current policy at every step",
                                    "dynamics": "cached rankings, P3 reader cache, pre-warmed Laya",
                                    "cached_trajectories": False}
    # The note must pin the published weights to the checkpoint-selection update.
    assert "best/" in summary["checkpoint_note"] and "best_update" in summary["checkpoint_note"]

    out = _policy_dir(cfg)
    for name in ("policy.safetensors", "scaler.json", "rl_config.json", "config.json",
                 train_rl.PRIOR_REF_FILE, train_rl.TRAJECTORIES_FILE, train_rl.TRAIN_LOG_FILE):
        assert (out / name).is_file() and (out / name).stat().st_size > 0, name
    assert (out / train_rl.PRIOR_REF_FILE).read_text(encoding="utf-8") == "none\n"
    assert read_json(out / "scaler.json")["clip"] == 10.0  # plain JSON, not a pickle blob
    assert read_json(out / "rl_config.json")["max_updates"] == 2  # the CLI override is what ran

    config = read_json(out / "config.json")
    assert config["seed"] == 3 and config["val_reward"] == summary["final_val"]["mean_reward"]
    assert config["val_action_counts"] == summary["final_val"]["action_counts"]
    assert config["worlds"] == {"train": 2, "val": 2, "excluded_ambiguous": 0,
                               "excluded_rate": 0.0}
    assert config["env_config"]["max_optional_search"] == 1  # cfg["rl"], not a default
    for split in (train_rl.TRAIN_SPLIT, train_rl.VAL_SPLIT):
        assert config["reader"][split]["reader_model_hash"] == "r" * 64
        assert config["reader"][split]["tokenizer_sha"] == "t" * 64
        assert config["reader"][split]["inference_config_sha"] == "i" * 64

    # Plots and traces.
    plots = resolve(cfg, "plots")
    for name in ("reward_learning_curve.png", "action_histogram.png"):
        assert (plots / name).is_file() and (plots / name).stat().st_size > 1000, name
    traces = list(read_jsonl(out / train_rl.TRAJECTORIES_FILE))
    assert traces and {row["qid"] for row in traces} <= {"toy:pos", "toy:neg"}
    assert all(row["policy"] == "greedy_final" for row in traces)
    assert len(list(read_jsonl(out / train_rl.TRAIN_LOG_FILE))) == 2

    # The summary is both printed and on disk, with provenance, and nothing left the tmpdir.
    stored = read_json(resolve(cfg, "reports") / train_rl.SUMMARY_FILE)
    assert stored["run_id"] == summary["run_id"]
    assert stored["retrieval_config_sha"] == config["retrieval_config_sha"]
    assert stored["reward_spec_hash"] == RewardConfig.from_dict(cfg).spec_hash()
    for path in summary["artifacts"]:
        assert Path(path).is_relative_to(root), path

    rows = _rows(cfg)
    assert [row["status"] for row in rows] == ["RUNNING", "SUCCEEDED"]
    assert rows[0]["stage"] == train_rl.STAGE and rows[0]["laya_regime"] is None
    manifest = rows[-1]["artifacts"]
    assert set(manifest) == set(summary["artifacts"])
    assert all("missing" not in entry for entry in manifest.values())
    assert manifest[str(out / "policy.safetensors")]["sha256"] == sha256_file(
        out / "policy.safetensors")
    assert (resolve(cfg, "manifests") / "P6_rl_smoke_config.yaml").is_file()


def test_cli_dry_run_writes_no_model(p6, capsys):
    cfg, _, root = p6
    overlay = root / "overlay.yaml"
    write_yaml(overlay, cfg)
    assert train_rl.main(["--config", str(overlay), "--dry-run"]) == 0
    printed = capsys.readouterr().out
    assert "(dry run)" in printed and "probe_val_reward=" in printed
    assert not (_policy_dir(cfg) / "policy.safetensors").exists()
    assert not (resolve(cfg, "reports") / train_rl.SUMMARY_FILE).exists()
    rows = _rows(cfg)
    assert [row["status"] for row in rows] == ["RUNNING", "SUCCEEDED"]
    assert rows[-1]["dry_run"] is True and "probe" in rows[-1]


def test_cli_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exit_info:
        train_rl.main(["--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "P6" in out and "--laya-regime" in out and "--dry-run" in out


# ------------------------------------------------------------------- split discipline

def test_ambiguous_negatives_are_excluded_from_train_but_kept_in_val(p6):
    cfg, state, _ = p6
    state["ambiguous"] = frozenset({"toy:pos"})
    summary = train_rl.run(cfg=cfg, profile="smoke", max_updates=2)
    assert summary["splits"]["train"]["worlds"] == 1
    assert summary["splits"]["val"]["worlds"] == 2  # val keeps them: the metric is honest
    assert summary["splits"]["val"]["ambiguous_kept"] == 1
    assert summary["excluded_ambiguous"] == {"n": 1, "rate": 0.5, "qids": ["toy:pos"]}
    traces = list(read_jsonl(_policy_dir(cfg) / train_rl.TRAJECTORIES_FILE))
    assert {row["qid"] for row in traces} == {"toy:neg"}
    config = read_json(_policy_dir(cfg) / "config.json")
    assert config["worlds"] == {"train": 1, "val": 2, "excluded_ambiguous": 1,
                               "excluded_rate": 0.5}


def test_a_train_split_with_nothing_left_fails_loudly_and_is_registered(p6):
    cfg, state, _ = p6
    state["train"] = [negative_world()]
    state["ambiguous"] = frozenset({"toy:neg"})
    with pytest.raises(ValueError, match="ambiguous"):
        train_rl.run(cfg=cfg, profile="smoke", max_updates=2)
    rows = _rows(cfg)
    assert [row["status"] for row in rows] == ["RUNNING", "FAILED"]
    assert "ValueError" in rows[-1]["error"]
    assert not (_policy_dir(cfg) / "policy.safetensors").exists()


def test_max_updates_below_one_is_rejected_before_any_run_is_registered(p6):
    cfg, _, _ = p6
    with pytest.raises(ValueError, match="max_updates"):
        train_rl.run(cfg=cfg, profile="smoke", max_updates=0)
    assert not (resolve(cfg, "runs") / "index.jsonl").exists()


# ---------------------------------------------------------------------- scaler + policy

def test_scaler_is_fit_on_train_rollouts_only(p6):
    cfg, state, _ = p6
    state["train"] = [positive_world()]
    state["val"] = [negative_world()]
    train_rl.run(cfg=cfg, profile="smoke", max_updates=2)
    saved = FeatureScaler.load(_policy_dir(cfg) / "scaler.json")
    rl_config = RLConfig.from_dict(cfg)

    def fit(world_slice):
        env = AdaptiveQAEnv(list(world_slice),
                            RewardCalculator(RewardConfig.from_dict(cfg)),
                            worlds.env_config(cfg))
        return ReinforceTrainer(env, env, rl_config, laya_fn=None, seed=rl_config.seed
                                ).fit_scaler()

    train_fit = fit([replace(positive_world(), split=train_rl.TRAIN_SPLIT)])
    val_fit = fit([replace(negative_world(), split=train_rl.VAL_SPLIT)])
    assert np.allclose(saved.mean, train_fit.mean, rtol=0, atol=0)
    assert np.allclose(saved.std, train_fit.std, rtol=0, atol=0)
    assert np.abs(np.asarray(saved.mean) - np.asarray(val_fit.mean)).max() > 1e-6, \
        "a val-derived scaler must differ, or 'train only' is not being tested"


def test_same_seed_reproduces_the_published_policy(p6):
    cfg, _, root = p6
    other = _cfg(root / "again")
    for split in (train_rl.TRAIN_SPLIT, train_rl.VAL_SPLIT):
        _write_reader_meta(other, split)
    first = train_rl.run(cfg=cfg, profile="smoke", max_updates=2)
    second = train_rl.run(cfg=other, profile="smoke", max_updates=2)
    assert first["final_val"] == second["final_val"]
    assert sha256_file(_policy_dir(cfg) / "policy.safetensors") == sha256_file(
        _policy_dir(other) / "policy.safetensors")


def test_saved_checkpoint_reproduces_the_measured_greedy_actions(p6):
    cfg, state, _ = p6
    # Run with the prior regime: the fixture's frozen 0.25 prior must reach the trainer's
    # features, or the reloaded policy is measured on different inputs than the summary's.
    _fake_checkpoint(cfg, "ce")
    summary = train_rl.run(cfg=cfg, profile="smoke", max_updates=2, laya_regime="ce")
    out = _policy_dir(cfg)
    net = PolicyNet.load(out / "policy.safetensors")
    scaler = FeatureScaler.load(out / "scaler.json")
    extractor = FeatureExtractor()
    scaler_t = TorchScaler(np.asarray(scaler.mean, dtype=np.float32),
                           np.asarray(scaler.std, dtype=np.float32), scaler.clip)
    env = train_rl.make_env(cfg, WorldBundle(
        split=train_rl.VAL_SPLIT,
        worlds=[replace(w, split=train_rl.VAL_SPLIT) for w in state["val"]],
        questions={}, docs={}, laya_fn=None, ambiguous_qids=frozenset()))
    prior = lambda obs: np.full(4, 0.25)  # noqa: E731 - the fixture's frozen prior (constant)
    kwargs = {"episodes": RLConfig.from_dict(cfg).eval_episodes, "device": torch.device("cpu")}
    first = train_rl.evaluate(env, net, extractor, scaler_t, prior, **kwargs)
    second = train_rl.evaluate(env, net, extractor, scaler_t, prior, **kwargs)
    assert first == second  # greedy evaluation is deterministic
    assert first["action_counts"] == summary["final_val"]["action_counts"], \
        "the reloaded checkpoint must reproduce the policy the report measured"
    assert read_json(_policy_dir(cfg) / "config.json")["val_action_counts"] == first["action_counts"]


def test_early_stopping_stops_when_validation_stops_improving(p6, monkeypatch, capsys):
    cfg, _, _ = p6
    cfg["rl"] = {**cfg["rl"], "max_updates": 8, "eval_every": 1, "early_stop_patience": 1,
                 "eval_episodes": 2}

    def flat_eval(env, net, extractor, scaler_t, laya_fn, *, episodes, device,
                  seed_base=10_000):
        return {"mean_reward": 1.0, "std_reward": 0.0, "mean_length": 2.0,
                "episodes": episodes, "action_counts": {"0": 1, "1": 1, "2": 0, "3": 2}}

    monkeypatch.setattr("src.rl.reinforce.evaluate", flat_eval)
    monkeypatch.setattr(train_rl, "evaluate", flat_eval)
    summary = train_rl.run(cfg=cfg, profile="smoke")
    assert summary["stopped_early"] is True
    assert summary["updates"] == 2  # update 1 improves, update 2 does not -> stop
    assert summary["best_update"] == 1
    assert summary["best_val_reward"] == 1.0
    assert summary["final_val"]["mean_reward"] == 1.0
    assert "WARNING" not in capsys.readouterr().out
    # ``best/`` holds exactly the update-1 weights; training continued to update 2, whose
    # weights live in ``last/``. If train() re-saved best/ after the loop the two would be
    # identical, silently shipping the wrong checkpoint.
    best = PolicyNet.load(_policy_dir(cfg) / "best" / "policy.safetensors")
    last = PolicyNet.load(_policy_dir(cfg) / "last" / "policy.safetensors")
    best_params = [p.detach().numpy().ravel() for p in best.parameters()]
    last_params = [p.detach().numpy().ravel() for p in last.parameters()]
    assert not all(np.allclose(b, l) for b, l in zip(best_params, last_params)), \
        "best/ must not hold the final update's weights"
    # The canonical artifact is the best/ weights, so the published policy is update 1's.
    canonical = PolicyNet.load(_policy_dir(cfg) / "policy.safetensors")
    assert all(np.allclose(c.detach().numpy().ravel(), b)
               for c, b in zip(canonical.parameters(), best_params))
    assert (resolve(cfg, "plots") / "reward_learning_curve.png").stat().st_size > 1000


# ------------------------------------------------------------------------- Laya prior

def test_explicit_laya_flag_records_the_checkpoint_reference_and_hash(p6):
    cfg, state, _ = p6
    path = _fake_checkpoint(cfg, "ce")
    summary = train_rl.run(cfg=cfg, profile="smoke", max_updates=2, laya_regime="ce")
    assert summary["laya"]["regime"] == "ce"
    assert summary["laya"]["model_hash"] == checkpoint_model_hash(path)
    assert summary["laya"]["state_text_version"] == STATE_TEXT_VERSION
    assert [call["laya_regime"] for call in state["calls"]] == ["ce", "ce"]
    assert read_json(_policy_dir(cfg) / "config.json")["laya"]["checkpoint_ref"] == str(path)
    ref = (_policy_dir(cfg) / train_rl.PRIOR_REF_FILE).read_text(encoding="utf-8")
    assert ref.strip() == str(path) and Path(ref.strip()).name == "laya_ce"


def test_config_frozen_prior_is_the_default_and_the_flag_overrides_it(p6):
    cfg, state, _ = p6
    cfg["laya"] = {**cfg["laya"], "frozen": "rlcd"}
    path = _fake_checkpoint(cfg, "rlcd")
    default = train_rl.run(cfg=cfg, profile="smoke", max_updates=2)
    assert default["laya"]["regime"] == "rlcd"
    assert default["laya"]["checkpoint_ref"] == str(path)
    assert [call["laya_regime"] for call in state["calls"]] == ["rlcd", "rlcd"]

    state["calls"].clear()
    cfg["rl"] = {**cfg["rl"], "max_updates": 1, "eval_every": 1}
    ablated = train_rl.run(cfg=cfg, profile="smoke", laya_regime="none")
    assert ablated["laya"] == {"regime": None, "checkpoint_ref": None, "model_hash": None,
                               "state_text_version": STATE_TEXT_VERSION}
    assert [call["laya_regime"] for call in state["calls"]] == [None, None]


def test_missing_laya_checkpoint_names_the_path(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(FileNotFoundError, match="laya_ce"):
        worlds.build_worlds(cfg, train_rl.TRAIN_SPLIT, laya_regime="ce")


def test_missing_p1_artifacts_fail_loudly(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(FileNotFoundError, match="docs_train_strict"):
        worlds.build_worlds(cfg, train_rl.TRAIN_SPLIT, laya_regime=None)


def test_prior_lookup_raises_rather_than_inventing_a_probability(p6):
    cfg, state, _ = p6
    env = AdaptiveQAEnv(list(state["train"]), RewardCalculator(RewardConfig.from_dict(cfg)),
                        worlds.env_config(cfg))
    obs, _ = env.reset(seed=0)
    with pytest.raises(KeyError, match="pre-warmed"):
        worlds.prior_lookup({})(obs)


def test_prewarm_enumeration_covers_every_state_a_rollout_visits(p6):
    cfg, state, _ = p6
    ws = [replace(w, split=train_rl.TRAIN_SPLIT) for w in state["train"]]
    requests = worlds.reachable_state_requests(
        ws, RewardCalculator(RewardConfig.from_dict(cfg)), worlds.env_config(cfg))
    keys = [(qid, state_hash) for qid, state_hash, _ in requests]
    assert requests and len(set(keys)) == len(keys)
    fn = worlds.prior_lookup({key: np.full(4, 0.25) for key in keys})
    env = AdaptiveQAEnv(list(ws), RewardCalculator(RewardConfig.from_dict(cfg)),
                        worlds.env_config(cfg))
    extractor = FeatureExtractor()
    net = PolicyNet(extractor.dim, 8)
    rng = np.random.default_rng(0)
    for _ in range(40):
        episode = train_rl.rollout(env, net, extractor, None, fn,
                                   seed=int(rng.integers(0, 2**31 - 1)))
        assert episode.transitions  # every step's Laya lookup hit the pre-warmed keys


# ------------------------------------------------------------------ config defaults

def test_missing_rl_keys_fall_back_to_the_spec_defaults():
    """base.yaml's ``rl:`` need not carry every key (spec 05 §6); these are the fallbacks."""
    base = load_config(None)
    without = ("max_optional_search", "eval_episodes", "gamma", "grad_clip")
    stripped = {**base, "rl": {k: v for k, v in base["rl"].items() if k not in without}}
    assert worlds.env_config(stripped).max_optional_search == 2
    rl_config = RLConfig.from_dict(stripped)
    assert rl_config.eval_episodes == 64 and rl_config.gamma == 1.0
    assert rl_config.grad_clip == 1.0 and rl_config.algorithm == "reinforce"


def test_unknown_rl_keys_are_tolerated_by_the_config_reader():
    """``rl.max_optional_search`` lives in ``rl:`` yet is not an RLConfig field."""
    base = load_config(None)
    assert "max_optional_search" not in RLConfig.__dataclass_fields__
    widened = {**base, "rl": {**base["rl"], "max_optional_search": 1, "future_key": True}}
    assert RLConfig.from_dict(widened).max_updates == base["rl"]["max_updates"]
    assert worlds.env_config(widened).max_optional_search == 1
