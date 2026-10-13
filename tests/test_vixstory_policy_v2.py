"""Contract tests for the V2-only REINFORCE policy and trainer (``src/rl/vixstory_policy_v2.py``).

Fixtures are tiny in-memory worlds built from the V2 environment contract; no dataset, no
reader, no API credentials, no GPU. The tests pin the plan's training protocol
(``.claude/plans/2026-10-12-vixstory-evidence-first-v2.md`` §5): three actions everywhere,
on-policy sampling, the frozen three-probability prior carrying the caller's hash and
receiving no gradient, the seed-42 three-permutation QID schedule chunked into batches of at
most 32 that cross epoch boundaries, train-only scaler fitting, checkpoints every 25 updates
plus the end, and greedy validation evaluation/selection kept outside the trainer.
"""

from __future__ import annotations

import ast
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.decision.schema_v2 import ACTION_ORDER
from src.pipeline.vixstory_arms_v2 import masked_greedy
from src.pipeline.vixstory_eval_v2 import evaluate_policy, select_best
from src.rl.policy import PolicyNet
from src.rl.vixstory_env_v2 import SEARCH_BM25, SEARCH_DENSE, STOP, VixStoryEnvV2, VixStoryWorldV2
from src.rl.vixstory_policy_v2 import (
    N_ACTIONS,
    Episode,
    Transition,
    V2FeatureScaler,
    V2RLConfig,
    VixStoryV2FeatureExtractor,
    VixStoryV2ReinforceTrainer,
    build_schedule,
    evaluate_greedy,
    policy_probs,
    prior_probs,
    reinforce_loss,
    rollout,
    split_batches,
    vixstory_v2_feature_names,
)

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "rl" / "vixstory_policy_v2.py"
PRIOR_HASH = "sha256:e05-frozen-prior-checkpoint"

CORPUS = {
    "d0": "Đoạn 0: trường A toạ lạc tại Hà Nội.",
    "d1": "Đoạn 1: trường A thành lập năm 1995.",
    "d2": "Đoạn 2: trường A có 12 khoa.",
    "d3": "Đoạn 3: trường A có 3 viện nghiên cứu.",
    "d4": "Đoạn 4: thư viện trường A có 20.000 đầu sách.",
    "d5": "Đoạn 5: trường A có 500 giảng viên.",
}
RANKINGS = {"bm25": ["d0", "d1", "d2", "d3"], "dense": ["d4", "d5", "d1", "d3"]}


def make_world(qid: str, gold: tuple[str, ...]) -> VixStoryWorldV2:
    return VixStoryWorldV2(qid=qid, question=f"Câu hỏi cho {qid}?", corpus=CORPUS,
                           rankings=RANKINGS, gold_context_ids=gold)


# q_a: BM25 solves it, q_b: Dense solves it, q_c: needs both tools.
WORLDS = [make_world("q_a", ("d1", "d2")), make_world("q_b", ("d4", "d5")),
          make_world("q_c", ("d1", "d4"))]
QIDS = [world.qid for world in WORLDS]


def make_env() -> VixStoryEnvV2:
    return VixStoryEnvV2(WORLDS)


def make_prior(stop_weight: float = 0.6, bm25_weight: float = 0.25,
               dense_weight: float = 0.15):
    """A deterministic three-probability prior over the legal mask (an E04/E05 stand-in)."""
    def prior(obs):
        mask = np.asarray(obs["action_mask"], dtype=bool)
        base = np.array([stop_weight * mask[STOP], bm25_weight * mask[SEARCH_BM25],
                         dense_weight * mask[SEARCH_DENSE]])
        total = base.sum()
        return base / total if total > 0 else np.array([1.0, 0.0, 0.0])
    return prior


class SpyEnv:
    """Wraps a real V2 env and records every qid handed to ``reset``."""

    def __init__(self, env: VixStoryEnvV2) -> None:
        self._env = env
        self._worlds = env._worlds
        self.reset_qids: list[str] = []

    def reset(self, *, qid=None, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options, qid=qid)
        self.reset_qids.append(obs["qid"])
        return obs, info

    def step(self, action):
        return self._env.step(action)


def forced_net(action: int, *, value_bias: float = 0.0) -> PolicyNet:
    """A net whose body is zeroed, so logits are exactly the policy bias (one forced action)."""
    extractor = VixStoryV2FeatureExtractor()
    torch.manual_seed(0)
    net = PolicyNet(extractor.dim, 8, n_actions=N_ACTIONS)
    with torch.no_grad():
        for parameter in net.body.parameters():
            parameter.zero_()
        net.policy_logits.weight.zero_()
        net.policy_logits.bias.copy_(torch.tensor(
            [10.0 if index == action else 0.0 for index in range(N_ACTIONS)]))
        net.value.weight.zero_()
        net.value.bias.fill_(value_bias)
    return net


def tiny_config(**overrides) -> V2RLConfig:
    base = dict(seed=42, hidden_dim=8, lr=1e-2, batch_size=3, updates=1, epochs=1,
                checkpoint_every=1, entropy_coef=0.01, value_coef=0.5, grad_clip=1.0)
    base.update(overrides)
    return V2RLConfig(**base)


def train_tiny(run_dir: Path | None, *, prior_hash: str = PRIOR_HASH, **overrides) -> dict:
    trainer = VixStoryV2ReinforceTrainer(make_env(), tiny_config(**overrides),
                                         prior=make_prior(), prior_hash=prior_hash,
                                         run_dir=run_dir)
    return trainer.train(QIDS)


# ---------------------------------------------------------------------------------------
# Configuration pins
# ---------------------------------------------------------------------------------------


def test_config_pins_the_protocol_defaults():
    config = V2RLConfig()
    assert config.seed == 42 and config.gamma == 1.0
    assert config.lr == 3e-4 and config.entropy_coef == 0.005 and config.value_coef == 0.5
    assert config.grad_clip == 1.0
    assert config.batch_size == 32 and config.updates == 240 and config.epochs == 3
    assert config.checkpoint_every == 25
    assert V2RLConfig.from_dict({"rl": {"lr": 1e-3}}).lr == 1e-3
    with pytest.raises(ValueError):
        V2RLConfig(batch_size=0)
    with pytest.raises(ValueError):
        V2RLConfig(lr=-1e-3)


# ---------------------------------------------------------------------------------------
# Feature contract: public observation + frozen prior only
# ---------------------------------------------------------------------------------------


def test_feature_contract_is_public_only():
    names = vixstory_v2_feature_names()
    assert names == VixStoryV2FeatureExtractor().names
    assert VixStoryV2FeatureExtractor().dim == len(names) == 15
    assert len(set(names)) == len(names)
    for name in names:
        assert not any(word in name for word in ("gold", "answer", "coverage", "oracle",
                                                 "reader", "candidate"))


def test_extractor_hand_computed_features():
    extractor = VixStoryV2FeatureExtractor()
    env = make_env()
    obs, _ = env.reset(qid="q_a")
    prior = make_prior(0.5, 0.3, 0.2)
    names = extractor.names
    vec = extractor.vector(obs, prior(obs))
    assert vec.shape == (extractor.dim,)
    assert vec.dtype == np.float32
    assert vec[names.index("prior_p_stop")] == pytest.approx(0.5)
    assert vec[names.index("prior_p_bm25")] == pytest.approx(0.3)
    assert vec[names.index("prior_p_dense")] == pytest.approx(0.2)
    expected_entropy = -(0.5 * math.log(0.5) + 0.3 * math.log(0.3) + 0.2 * math.log(0.2))
    assert vec[names.index("prior_entropy")] == pytest.approx(expected_entropy, abs=1e-6)
    assert vec[names.index("seen_count")] == 1.0
    assert vec[names.index("optional_used")] == 0.0
    assert vec[names.index("seen_tokens")] == float(len(CORPUS["d0"].split()))
    assert vec[names.index("mask_stop")] == 1.0
    assert vec[names.index("mask_bm25")] == 1.0
    assert vec[names.index("mask_dense")] == 1.0
    assert vec[names.index("last_action_STOP")] == 0.0
    assert vec[names.index("last_action_SEARCH_BM25")] == 0.0
    assert vec[names.index("question_len_tokens")] == float(len(obs["question"].split()))
    assert vec[names.index("question_len_chars")] == float(len(obs["question"]))

    after, _, _, _, _ = env.step(SEARCH_BM25)
    vec2 = extractor.vector(after, prior(after))
    assert vec2[names.index("seen_count")] == 3.0
    assert vec2[names.index("optional_used")] == 1.0
    assert vec2[names.index("last_action_SEARCH_BM25")] == 1.0
    assert vec2[names.index("last_action_STOP")] == 0.0
    assert vec2[names.index("seen_tokens")] == float(
        sum(len(text.split()) for text in after["seen_doc_texts"]))


def test_extractor_rejects_malformed_prior_or_mask():
    extractor = VixStoryV2FeatureExtractor()
    obs, _ = make_env().reset(qid="q_a")
    with pytest.raises(ValueError):
        extractor.vector(obs, [0.5, 0.5])
    with pytest.raises(ValueError):
        extractor.vector({**obs, "action_mask": [True, True, True, True]}, [0.5, 0.3, 0.2])
    without_mask = {key: value for key, value in obs.items() if key != "action_mask"}
    with pytest.raises(ValueError):
        extractor.vector(without_mask, [0.5, 0.3, 0.2])


def test_features_do_not_depend_on_gold_or_smuggled_labels():
    extractor = VixStoryV2FeatureExtractor()
    prior = make_prior()
    env_a = VixStoryEnvV2([make_world("q_a", ("d1", "d2"))])
    env_b = VixStoryEnvV2([make_world("q_a", ("d3",))])
    obs_a, _ = env_a.reset(qid="q_a")
    obs_b, _ = env_b.reset(qid="q_a")
    np.testing.assert_array_equal(extractor.vector(obs_a, prior(obs_a)),
                                  extractor.vector(obs_b, prior(obs_b)))
    leaky = {**obs_a, "gold_context_ids": ["d1"], "answer": "1995", "coverage": 1.0,
             "oracle_return": 1.5}
    np.testing.assert_array_equal(extractor.vector(obs_a, prior(obs_a)),
                                  extractor.vector(leaky, prior(obs_a)))


FORBIDDEN_IMPORTS = ("src.qa", "src.rl.env", "src.rl.features", "src.rl.reinforce",
                     "src.rl.vixstory_env", "requests", "openai", "urllib", "http")


def test_module_imports_stay_v2_standalone():
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    for name in imported:
        for forbidden in FORBIDDEN_IMPORTS:
            assert not (name == forbidden or name.startswith(forbidden + ".")), name
    assert "src.rl.policy" in imported
    assert "src.rl.vixstory_env_v2" not in imported  # features/rollout never need the env type


def test_module_source_never_reads_private_gold_or_a_reader():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "private_outcome" not in source
    assert "gold_context_ids" not in source


# ---------------------------------------------------------------------------------------
# Frozen prior: validated, detached, hash comes from the caller
# ---------------------------------------------------------------------------------------


def test_prior_probs_validates_and_detaches():
    obs, _ = make_env().reset(qid="q_a")
    assert prior_probs(make_prior(), obs) == pytest.approx((0.6, 0.25, 0.15))
    with pytest.raises(ValueError):
        prior_probs(lambda _obs: [0.5, 0.5], obs)
    with pytest.raises(ValueError):
        prior_probs(lambda _obs: [0.5, 0.5, -0.1], obs)
    with pytest.raises(ValueError):
        prior_probs(lambda _obs: [0.5, 0.5, float("nan")], obs)
    with pytest.raises(ValueError):
        prior_probs(lambda _obs: [0.5, 0.4, 0.2], obs)
    with pytest.raises(TypeError):
        prior_probs("not callable", obs)
    module = nn.Linear(1, N_ACTIONS)
    values = prior_probs(lambda _obs: torch.softmax(module(torch.zeros(1)), dim=-1), obs)
    assert sum(values) == pytest.approx(1.0, abs=1e-6)
    assert all(isinstance(value, float) for value in values)


def test_no_gradient_into_the_frozen_prior(tmp_path):
    class TorchPrior(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.zeros(3))

        def forward(self, obs):
            mask = torch.as_tensor(np.asarray(obs["action_mask"], dtype=bool))
            logits = torch.where(mask, self.scale, torch.full_like(self.scale, -1e9))
            return torch.softmax(logits, dim=-1)

    prior = TorchPrior()
    before = prior.scale.detach().clone()
    trainer = VixStoryV2ReinforceTrainer(make_env(), tiny_config(), prior=prior,
                                         prior_hash=PRIOR_HASH, run_dir=tmp_path / "run")
    trainer.train(QIDS)
    assert all(parameter.grad is None for parameter in prior.parameters())
    torch.testing.assert_close(prior.scale.detach(), before)


def test_prior_hash_from_caller_binds_identity_and_schedule(tmp_path):
    first = train_tiny(tmp_path / "a", prior_hash="sha256:E05")
    second = train_tiny(tmp_path / "b", prior_hash="sha256:E05")
    other = train_tiny(tmp_path / "c", prior_hash="sha256:E04")
    assert first["identity"] == second["identity"]
    assert first["identity"] != other["identity"]
    assert first["schedule_sha256"] == other["schedule_sha256"]
    summary = json.loads((tmp_path / "a" / "summary.json").read_text(encoding="utf-8"))
    assert summary["prior_hash"] == "sha256:E05"
    checkpoint = json.loads(
        (tmp_path / "a" / "checkpoints" / "update_0001" / "checkpoint.json")
        .read_text(encoding="utf-8"))
    assert checkpoint["prior_hash"] == "sha256:E05"
    assert checkpoint["identity"] == first["identity"]
    assert checkpoint["n_actions"] == N_ACTIONS
    with pytest.raises(ValueError):
        VixStoryV2ReinforceTrainer(make_env(), tiny_config(), prior=make_prior(),
                                   prior_hash="   ")


# ---------------------------------------------------------------------------------------
# The pinned QID schedule: three seed-42 permutations, batches <= 32, no early stop
# ---------------------------------------------------------------------------------------


def test_schedule_matches_the_pinned_protocol():
    qids = [f"q{index:04d}" for index in range(2553)]
    schedule = build_schedule(qids, epochs=3, seed=42)
    assert len(schedule) == 7659
    counts = Counter(schedule)
    assert len(counts) == 2553 and set(counts.values()) == {3}
    batches = split_batches(schedule, 32)
    assert len(batches) == 240
    assert [len(batch) for batch in batches[:239]] == [32] * 239
    assert len(batches[-1]) == 11
    assert build_schedule(qids, epochs=3, seed=42) == schedule
    assert build_schedule(qids, epochs=3, seed=43) != schedule


def test_schedule_is_seed42_permutations_in_epoch_blocks():
    qids = [f"q{index}" for index in range(7)]
    schedule = build_schedule(qids, epochs=3, seed=42)
    assert len(schedule) == 21
    blocks = [schedule[start:start + 7] for start in range(0, 21, 7)]
    for block in blocks:
        assert sorted(block) == sorted(qids)
    assert build_schedule(list(reversed(qids)), epochs=3, seed=42) == schedule
    with pytest.raises(ValueError):
        build_schedule([], epochs=3, seed=42)
    with pytest.raises(ValueError):
        build_schedule(["a", "a"], epochs=3, seed=42)


def test_batches_cross_epoch_boundaries():
    schedule = build_schedule(["a", "b", "c"], epochs=3, seed=42)
    batches = split_batches(schedule, 4)
    assert [len(batch) for batch in batches] == [4, 4, 1]
    assert batches[0] == schedule[:3] + schedule[3:4]
    assert Counter(qid for batch in batches for qid in batch) == Counter({"a": 3, "b": 3, "c": 3})
    with pytest.raises(ValueError):
        split_batches(schedule, 0)
    with pytest.raises(ValueError):
        split_batches([], 4)


# ---------------------------------------------------------------------------------------
# Rollout: on-policy, mask-consistent, replayable, greedy is deterministic argmax
# ---------------------------------------------------------------------------------------


def test_rollout_actions_come_from_the_current_network():
    extractor = VixStoryV2FeatureExtractor()
    torch.manual_seed(0)
    net = PolicyNet(extractor.dim, 8, n_actions=N_ACTIONS)
    episode = rollout(make_env(), net, extractor, None, make_prior(), qid="q_a")
    assert episode.qid == "q_a" and episode.transitions
    for transition in episode.transitions:
        x = torch.as_tensor(transition.features, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.as_tensor(transition.mask, dtype=torch.bool).unsqueeze(0)
        logits, _ = net(x)
        dist = torch.distributions.Categorical(logits=logits.masked_fill(~mask_t, -1e9))
        recomputed = float(dist.log_prob(torch.tensor([transition.action])).item())
        assert abs(recomputed - transition.log_prob) < 1e-5


def test_rollout_sampling_is_stochastic():
    extractor = VixStoryV2FeatureExtractor()
    torch.manual_seed(0)
    net = PolicyNet(extractor.dim, 8, n_actions=N_ACTIONS)
    env = make_env()
    first_actions = {rollout(env, net, extractor, None, make_prior(),
                             qid="q_a").transitions[0].action for _ in range(40)}
    assert len(first_actions) > 1


def test_replaying_recorded_actions_reproduces_masks_and_rewards():
    extractor = VixStoryV2FeatureExtractor()
    torch.manual_seed(3)
    net = PolicyNet(extractor.dim, 8, n_actions=N_ACTIONS)
    episode = rollout(make_env(), net, extractor, None, make_prior(), qid="q_c")
    replay = make_env()
    obs, _ = replay.reset(qid="q_c")
    total = 0.0
    terminated = truncated = False
    for transition in episode.transitions:
        np.testing.assert_array_equal(transition.mask, np.asarray(obs["action_mask"], dtype=bool))
        obs, reward, terminated, truncated, _ = replay.step(transition.action)
        assert reward == pytest.approx(transition.reward)
        assert not truncated
        total += reward
    assert terminated
    assert total == pytest.approx(episode.total_reward)
    assert obs["history"][-1] == "STOP"


def test_greedy_rollout_is_a_deterministic_argmax():
    extractor = VixStoryV2FeatureExtractor()
    prior = make_prior()
    stop_episode = rollout(make_env(), forced_net(STOP), extractor, None, prior,
                           qid="q_a", greedy=True)
    assert [ACTION_ORDER[t.action] for t in stop_episode.transitions] == ["STOP"]
    assert stop_episode.total_reward == 0.0
    bm25_episode = rollout(make_env(), forced_net(SEARCH_BM25), extractor, None, prior,
                           qid="q_a", greedy=True)
    assert [ACTION_ORDER[t.action] for t in bm25_episode.transitions] == [
        "SEARCH_BM25", "SEARCH_BM25", "STOP"]
    assert bm25_episode.total_reward == pytest.approx(1.4)
    again = rollout(make_env(), forced_net(SEARCH_BM25), extractor, None, prior,
                    qid="q_a", greedy=True)
    assert [t.action for t in again.transitions] == [t.action for t in bm25_episode.transitions]
    dense_episode = rollout(make_env(), forced_net(SEARCH_DENSE), extractor, None, prior,
                            qid="q_b", greedy=True)
    assert [ACTION_ORDER[t.action] for t in dense_episode.transitions] == [
        "SEARCH_DENSE", "SEARCH_DENSE", "STOP"]
    assert dense_episode.total_reward == pytest.approx(1.4)


# ---------------------------------------------------------------------------------------
# Loss: sign, hand-computed value term, entropy term
# ---------------------------------------------------------------------------------------


def test_reinforce_loss_sign_and_hand_computed_terms():
    extractor = VixStoryV2FeatureExtractor()
    net = forced_net(SEARCH_BM25, value_bias=0.0)
    transition = Transition(features=np.zeros(extractor.dim, dtype=np.float32),
                            mask=np.array([True, True, True]), action=SEARCH_BM25,
                            log_prob=0.0, value=0.0, entropy=0.0, reward=1.0)
    episode = Episode(qid="q_a", transitions=[transition], total_reward=1.0)
    before, stats = reinforce_loss([episode], net, None, V2RLConfig(), device="cpu")
    assert stats["n_steps"] == 1
    assert stats["value_loss"] == pytest.approx(1.0)  # (V - G)^2 = (0 - 1)^2
    with torch.no_grad():
        net.policy_logits.bias[SEARCH_BM25] += 1.0
    after, _ = reinforce_loss([episode], net, None, V2RLConfig(), device="cpu")
    assert float(after.detach()) < float(before.detach())
    with pytest.raises(ValueError):
        reinforce_loss([], net, None, V2RLConfig(), device="cpu")


# ---------------------------------------------------------------------------------------
# Bridge + greedy evaluation: three-action masked probabilities, selection metrics
# ---------------------------------------------------------------------------------------


def test_policy_probs_bridge_is_masked_and_three_action():
    extractor = VixStoryV2FeatureExtractor()
    net = forced_net(SEARCH_DENSE)
    probs = policy_probs(net, extractor, None, make_prior(), device="cpu")
    env = make_env()
    obs, _ = env.reset(qid="q_a")
    vector = np.asarray(probs(obs), dtype=np.float64)
    assert vector.shape == (N_ACTIONS,)
    assert vector.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(vector >= 0.0)
    assert int(np.argmax(vector)) == SEARCH_DENSE
    assert masked_greedy(probs)(obs) == SEARCH_DENSE
    for action in (SEARCH_BM25, SEARCH_DENSE):
        obs, _, _, _, _ = env.step(action)
    terminal = np.asarray(probs(obs), dtype=np.float64)
    assert terminal[SEARCH_BM25] == 0.0 and terminal[SEARCH_DENSE] == 0.0
    assert terminal[STOP] == pytest.approx(1.0)


def test_evaluate_greedy_agrees_with_the_shared_evaluator_and_selector():
    """The bridge the E06/E07 driver uses: greedy net -> evaluate_policy -> select_best."""
    extractor = VixStoryV2FeatureExtractor()
    net = forced_net(SEARCH_BM25)
    prior = make_prior()
    report = evaluate_greedy(make_env(), net, extractor, None, prior, QIDS, device="cpu")
    policy = masked_greedy(policy_probs(net, extractor, None, prior, device="cpu"))
    evaluation = evaluate_policy(WORLDS, policy, "E06")
    assert evaluation["summary"]["mean_retrieval_return"] == pytest.approx(
        report["mean_retrieval_return"])
    assert evaluation["summary"]["mean_n_search"] == pytest.approx(report["mean_n_search"])
    candidates = [{"index": 0, **report}, {"index": 1, **evaluation["summary"]}]
    assert select_best(candidates)["index"] == 0  # identical metrics -> earliest checkpoint


def test_evaluate_greedy_hand_computed_selection_metrics():
    extractor = VixStoryV2FeatureExtractor()
    prior = make_prior()
    # Forced BM25 on every QID: two searches, then STOP.
    report = evaluate_greedy(make_env(), forced_net(SEARCH_BM25), extractor, None, prior,
                             QIDS, device="cpu")
    assert report["n"] == 3
    assert report["mean_n_search"] == pytest.approx(2.0)
    assert report["mean_length"] == pytest.approx(3.0)
    # q_a: full gold -> 1.5 - 0.10; q_b: no gold -> 0.0 - 0.10; q_c: half -> 0.25 - 0.10.
    assert report["mean_retrieval_return"] == pytest.approx((1.4 - 0.1 + 0.15) / 3)
    assert report["action_counts"] == {"STOP": 3, "SEARCH_BM25": 6, "SEARCH_DENSE": 0}
    assert report["qids_sha256"] == report["qids_sha256"]
    stop_report = evaluate_greedy(make_env(), forced_net(STOP), extractor, None, prior,
                                  QIDS, device="cpu")
    assert stop_report["mean_retrieval_return"] == pytest.approx(0.0)
    assert stop_report["mean_n_search"] == pytest.approx(0.0)
    assert stop_report["mean_length"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        evaluate_greedy(make_env(), forced_net(STOP), extractor, None, prior, [], device="cpu")


# ---------------------------------------------------------------------------------------
# Trainer: schedule, train-only scaler, artifacts, checkpoints, overwrite refusal
# ---------------------------------------------------------------------------------------


def test_trainer_follows_the_schedule_without_early_stop(tmp_path):
    spy = SpyEnv(make_env())
    trainer = VixStoryV2ReinforceTrainer(spy, tiny_config(batch_size=4, epochs=3, updates=3,
                                                          checkpoint_every=2),
                                         prior=make_prior(), prior_hash=PRIOR_HASH,
                                         run_dir=tmp_path / "run")
    result = trainer.train(QIDS)
    assert result["n_updates"] == 3 and len(trainer.history) == 3
    assert result["n_episodes"] == 9
    # Exactly three episodes per QID: the whole schedule ran, nothing was cut short.
    assert result["episodes_per_qid"] == {"min": 3, "max": 3, "distinct_qids": 3}
    # Every reset was a train QID: there is no validation environment to touch.
    assert set(spy.reset_qids) == set(QIDS)
    checkpoints = sorted(path.name for path in result["checkpoints"])
    assert checkpoints == ["update_0002", "update_0003"]  # cadence plus the final update
    for name in checkpoints:
        assert (tmp_path / "run" / "checkpoints" / name / "policy.safetensors").is_file()
        assert (tmp_path / "run" / "checkpoints" / name / "scaler.json").is_file()
        assert (tmp_path / "run" / "checkpoints" / name / "checkpoint.json").is_file()
    assert not (tmp_path / "run" / "best").exists()
    assert not hasattr(trainer, "val_env")


def test_trainer_logs_track_actions_qids_and_never_validation(tmp_path):
    run_dir = tmp_path / "run"
    result = train_tiny(run_dir)
    rows = [json.loads(line) for line in
            (run_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["update"] for row in rows] == [1]
    row = rows[0]
    # The exact monitor schema: no validation metric can hide in the train log.
    assert set(row) == {"update", "train_mean_retrieval_return", "mean_episode_length",
                        "unique_qids", "action_counts", "grad_norm", "param_delta",
                        "policy_loss", "value_loss", "entropy", "n_steps"}
    assert sum(row["action_counts"].values()) == row["n_steps"]
    assert row["unique_qids"] == 3
    assert row["param_delta"] > 0.0
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_updates"] == result["n_updates"] == 1
    assert summary["n_episodes"] == 3
    assert summary["prior_hash"] == PRIOR_HASH
    assert summary["schedule_sha256"] == result["schedule_sha256"]
    assert summary["config"]["lr"] == pytest.approx(1e-2)


def test_scaler_is_fit_on_train_qids_only(tmp_path):
    spy = SpyEnv(make_env())
    trainer = VixStoryV2ReinforceTrainer(spy, tiny_config(), prior=make_prior(),
                                         prior_hash=PRIOR_HASH, run_dir=None)
    scaler = trainer.fit_scaler(QIDS, max_episodes=2)
    assert set(spy.reset_qids) == set(sorted(QIDS)[:2])
    assert len(scaler.mean) == trainer.extractor.dim == len(scaler.std)
    assert trainer.scaler_t is not None
    spy.reset_qids.clear()
    trainer.train(QIDS)
    assert set(spy.reset_qids) <= set(QIDS)


def test_trainer_refuses_overwrite_and_protocol_mismatch(tmp_path):
    run_dir = tmp_path / "run"
    train_tiny(run_dir)
    with pytest.raises(FileExistsError):
        train_tiny(run_dir)
    trainer = VixStoryV2ReinforceTrainer(make_env(), V2RLConfig(), prior=make_prior(),
                                         prior_hash=PRIOR_HASH, run_dir=None)
    with pytest.raises(ValueError, match="240"):
        trainer.train(QIDS)
    trainer = VixStoryV2ReinforceTrainer(make_env(), tiny_config(), prior=make_prior(),
                                         prior_hash=PRIOR_HASH, run_dir=None)
    with pytest.raises(ValueError):
        trainer.train(["q_a", "q_z"])
    with pytest.raises(ValueError):
        trainer.train(["q_a", "q_a"])
    with pytest.raises(ValueError):
        trainer.train([])
    trainer = VixStoryV2ReinforceTrainer(make_env(), tiny_config(), prior=make_prior(),
                                         prior_hash=PRIOR_HASH, run_dir=None)
    trainer.train(QIDS)
    with pytest.raises(RuntimeError, match="already ran"):
        trainer.train(QIDS)


def test_same_seed_is_reproducible(tmp_path):
    first = train_tiny(tmp_path / "a")
    second = train_tiny(tmp_path / "b")
    assert first["identity"] == second["identity"]
    assert [row["policy_loss"] for row in first["updates"]] == \
        [row["policy_loss"] for row in second["updates"]]
    assert [row["train_mean_retrieval_return"] for row in first["updates"]] == \
        [row["train_mean_retrieval_return"] for row in second["updates"]]


# ---------------------------------------------------------------------------------------
# Save / load: exactly three actions, prior hash consistent
# ---------------------------------------------------------------------------------------


def test_train_save_load_evaluate_roundtrip_is_three_action(tmp_path):
    run_dir = tmp_path / "run"
    trainer = VixStoryV2ReinforceTrainer(make_env(), tiny_config(), prior=make_prior(),
                                         prior_hash=PRIOR_HASH, run_dir=run_dir)
    trainer.train(QIDS)
    assert trainer.net is not None and trainer.net.n_actions == N_ACTIONS == 3
    checkpoint = run_dir / "checkpoints" / "update_0001"
    direct = PolicyNet.load(checkpoint / "policy.safetensors")
    assert direct.n_actions == 3
    restored = VixStoryV2ReinforceTrainer(make_env(), tiny_config(), prior=make_prior(),
                                          prior_hash=PRIOR_HASH, run_dir=None)
    restored.load(checkpoint)
    assert restored.net is not None and restored.net.n_actions == 3
    report = evaluate_greedy(make_env(), restored.net, restored.extractor, restored.scaler_t,
                             restored.prior, QIDS, device="cpu")
    assert report["n"] == 3
    assert np.isfinite(report["mean_retrieval_return"])


def test_load_refuses_four_action_or_foreign_prior_checkpoints(tmp_path):
    extractor = VixStoryV2FeatureExtractor()
    four_action = tmp_path / "four"
    PolicyNet(extractor.dim, 8, n_actions=4).save(four_action / "policy.safetensors")
    V2FeatureScaler(mean=[0.0] * extractor.dim, std=[1.0] * extractor.dim).save(
        four_action / "scaler.json")
    trainer = VixStoryV2ReinforceTrainer(make_env(), tiny_config(), prior=make_prior(),
                                         prior_hash=PRIOR_HASH, run_dir=None)
    with pytest.raises(ValueError, match="3"):
        trainer.load(four_action)
    trained = train_tiny(tmp_path / "run", prior_hash="sha256:E04")
    assert trained["identity"]
    foreign = VixStoryV2ReinforceTrainer(make_env(), tiny_config(), prior=make_prior(),
                                         prior_hash="sha256:E05", run_dir=None)
    with pytest.raises(ValueError, match="prior"):
        foreign.load(tmp_path / "run" / "checkpoints" / "update_0001")
    missing_scaler = tmp_path / "no_scaler"
    PolicyNet(extractor.dim, 8, n_actions=N_ACTIONS).save(missing_scaler / "policy.safetensors")
    with pytest.raises(ValueError):
        foreign.load(missing_scaler)
