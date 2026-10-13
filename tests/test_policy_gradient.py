"""`training_is_real` tests — `02_IMPLEMENTATION/05_...md` §11.

Unit-level contracts for the shared REINFORCE machinery (``src/rl/reinforce.py``): the
discounted-return recursion, the masked-logit contract, the transition-mean objective, the
entropy term and a real gradient step — plus the feature scaler and the no-gold provenance
of the feature vector. These are the checks that stop a silently-broken RL stage from being
reported as a successful one; the VixStory environment/policy tests cover the end-to-end
rollout and trainer paths.

Everything here is CPU-only and uses in-memory fixtures: episodes are built from
``Transition`` objects directly, and the net is a freshly-seeded ``PolicyNet``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.schemas import find_forbidden_keys
from src.decision.schema import ACTION_ORDER
from src.rl.features import FeatureExtractor, FeatureScaler
from src.rl.policy import PolicyNet
from src.rl.reinforce import (
    Episode,
    RLConfig,
    Transition,
    discounted_returns,
    reinforce_loss,
    rollout,
)

ANSWER = ACTION_ORDER.index("ANSWER")
ABSTAIN = ACTION_ORDER.index("ABSTAIN")
N_ACTIONS = len(ACTION_ORDER)

#: Constant raw features for in-memory transitions (the net recomputes the distribution).
FEATURES = np.zeros(4, dtype=np.float32)


def _params(net: PolicyNet) -> np.ndarray:
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in net.parameters()])


def _episode(*, action: int, reward: float = 1.0, n_steps: int = 1) -> Episode:
    """A one-step (or n-step) in-memory episode with every action legal."""
    transitions = [
        Transition(features=FEATURES, mask=np.ones(N_ACTIONS, dtype=bool), action=action,
                   log_prob=0.0, value=0.0, entropy=0.0, reward=reward)
        for _ in range(n_steps)
    ]
    return Episode(qid="q", transitions=transitions)


# ---------------------------------------------------------------- unit level
def test_discounted_returns_with_gamma_one_is_a_reverse_cumsum():
    rewards = [0.0, -0.1, 1.9]
    assert discounted_returns(rewards, 1.0) == pytest.approx([1.8, 1.8, 1.9])


def test_discounted_returns_discount_future_rewards():
    rewards = [0.0, 1.0]
    assert discounted_returns(rewards, 0.5) == pytest.approx([0.5, 1.0])


@pytest.mark.parametrize("greedy", [False, True])
def test_rollout_masks_invalid_actions_before_sampling(greedy):
    class AbstainOnlyEnv:
        def reset(self, **kwargs):
            return {"qid": "q", "action_mask": [False, False, False, True]}, {}

        def step(self, action):
            assert action == ABSTAIN
            return {}, 0.0, True, False, {}

    class ConstantFeatures:
        def vector(self, obs, laya_probs):
            return FEATURES

    net = PolicyNet(d_in=4, d_hidden=8)
    with torch.no_grad():
        net.policy_logits.weight.zero_()
        net.policy_logits.bias.copy_(torch.tensor([100.0, 100.0, 100.0, -100.0]))
    episode = rollout(AbstainOnlyEnv(), net, ConstantFeatures(), None, None, greedy=greedy)
    transition = episode.transitions[0]
    assert transition.action == ABSTAIN
    assert transition.log_prob == pytest.approx(0.0)
    assert transition.entropy == pytest.approx(0.0)


def test_features_never_contain_gold_labels():
    """Spec 05 §11: no label-bearing field may appear in the training state, and a
    label smuggled in next to the visible fields cannot change the feature vector."""
    extractor = FeatureExtractor()
    obs = {
        "question": "Phạm Văn Đồng giữ chức Thủ tướng từ năm nào?",
        "action_mask": np.ones(N_ACTIONS, dtype=bool),
        "candidate": {"text": "1976", "non_null_score": 5.0, "null_margin": 4.0,
                      "provenance_ok": True},
        "retrieval": {"bm25_top_score": 0.5, "bm25_top_gap": 0.1,
                      "dense_top_score": 0.4, "dense_top_gap": 0.2},
        "seen_doc_ids": ("doc_b",), "optional_used": 1,
        "remaining_optional_searches": 1, "searches_used": 1, "timestep": 2,
        "last_action": ANSWER,
    }
    assert find_forbidden_keys(obs) == []
    probs = [0.1, 0.2, 0.3, 0.4]
    vec = extractor.vector(obs, probs)
    assert np.all(np.isfinite(vec))
    assert vec.shape[0] == extractor.dim

    smuggled = {**obs, "gold_doc_id": "doc_b", "answer_texts": ("1976",),
                "private": {"is_impossible": False, "reward": 2.0}}
    assert find_forbidden_keys(smuggled) != []
    assert np.array_equal(extractor.vector(smuggled, probs), vec)


def test_scaler_matches_declared_width_and_bounds(tmp_path):
    matrix = np.random.default_rng(0).normal(size=(64, 6)).astype(np.float32)
    scaler = FeatureScaler.fit(matrix)
    assert len(scaler.mean) == len(scaler.std) == 6
    probe = np.full((1, 6), 1e6, dtype=np.float32)
    transformed = scaler.transform(probe)[0]
    assert np.all(np.abs(transformed) <= scaler.clip + 1e-6)
    scaler.save(tmp_path / "scaler.json")
    assert FeatureScaler.load(tmp_path / "scaler.json").to_json() == scaler.to_json()


def test_scaler_declares_unit_std_for_constant_columns():
    """A zero-variance column divides by 1, never by 0 — the transform stays finite."""
    constant = np.zeros((4, 2), dtype=np.float32)
    scaler = FeatureScaler.fit(constant)
    assert scaler.std == [1.0, 1.0]
    assert np.all(scaler.transform(constant) == 0.0)


# ------------------------------------------------------------- training real
def test_reinforce_loss_uses_transition_mean_not_episode_mean():
    """One extra zero-advantage transition changes the loss denominator, not the objective."""
    class ConstantNet(torch.nn.Module):
        def forward(self, features):
            return torch.tensor([[0.0, 0.0, 0.0, 0.0]]), torch.tensor([0.0])

    def transition(reward):
        return Transition(features=np.zeros(1, dtype=np.float32),
                          mask=np.ones(4, dtype=bool), action=ANSWER,
                          log_prob=0.0, value=0.0, entropy=0.0, reward=reward)

    config = RLConfig(gamma=1.0, entropy_coef=0, value_coef=0)
    net = ConstantNet()
    short = Episode(qid="short", transitions=[transition(1.0)])
    empty_reward = Episode(qid="long", transitions=[transition(0.0), transition(0.0)])
    only_short, stats_one = reinforce_loss([short], net, None, config,
                                           device=torch.device("cpu"))
    combined, stats_three = reinforce_loss([short, empty_reward], net, None, config,
                                           device=torch.device("cpu"))
    assert stats_one["n_steps"] == 1 and stats_three["n_steps"] == 3
    assert combined.item() == pytest.approx(only_short.item() / 3)


def test_reinforce_loss_refuses_an_empty_batch():
    net = PolicyNet(d_in=4, d_hidden=8)
    with pytest.raises(ValueError, match="no transitions"):
        reinforce_loss([], net, None, RLConfig(), device=torch.device("cpu"))


def test_one_update_changes_policy_parameters_and_yields_finite_gradients():
    """Spec 05 §11: |Δθ| > 0 and finite after a real update."""
    torch.manual_seed(0)
    net = PolicyNet(d_in=4, d_hidden=8)
    before = _params(net).copy()

    episodes = [_episode(action=action) for action in range(N_ACTIONS)]
    loss, _ = reinforce_loss(episodes, net, None, RLConfig(value_coef=0.0, entropy_coef=0.0),
                             device=torch.device("cpu"))
    loss.backward()
    grad_norms = [float(p.grad.norm()) for p in net.parameters() if p.grad is not None]
    assert np.isfinite(loss.detach().item())
    assert any(g > 0 for g in grad_norms), "at least one layer must receive gradient"

    with torch.no_grad():
        for p in net.parameters():
            if p.grad is not None:
                p -= 1e-3 * p.grad
    after = _params(net)
    assert np.abs(after - before).mean() > 0
    assert np.all(np.isfinite(after))


def test_entropy_bonus_changes_the_loss_sign_direction():
    """Higher entropy coefficient must lower the reported entropy term's contribution."""
    torch.manual_seed(0)
    net = PolicyNet(d_in=4, d_hidden=8)
    episodes = [_episode(action=action) for action in range(N_ACTIONS)]
    low, stats_low = reinforce_loss(episodes, net, None, RLConfig(entropy_coef=0.0),
                                    device=torch.device("cpu"))
    high, stats_high = reinforce_loss(episodes, net, None, RLConfig(entropy_coef=1.0),
                                      device=torch.device("cpu"))
    assert stats_low["entropy"] == pytest.approx(stats_high["entropy"])
    assert stats_high["entropy"] > 0.0
    assert high.detach().item() < low.detach().item()
