"""`training_is_real` tests — `02_IMPLEMENTATION/05_...md` §11.

These are the tests that stop a silently-broken RL stage from being reported as a
successful one.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.schemas import find_forbidden_keys
from src.rl.env import ABSTAIN, ANSWER, SEARCH_BM25, AdaptiveQAEnv, EnvConfig, EpisodeWorld
from src.rl.features import FeatureExtractor, FeatureScaler, mask_logits
from src.rl.policy import PolicyNet
from src.rl.reward import EpisodePrivate
from src.rl.reinforce import (
    RLConfig,
    ReinforceTrainer,
    discounted_returns,
    evaluate,
    reinforce_loss,
    rollout,
)
from tests.toy_world import DOC_B, DOC_C, default_calculator, make_reader, positive_world, span_prediction

ANSWER_ONLY = "1976"


def answerable_world() -> EpisodeWorld:
    """Bootstrap lands straight on the gold passage, so ANSWER is always the best action."""
    docs = {"doc_b": DOC_B}
    preds = {("doc_b",): span_prediction("q:answer", "doc_b", DOC_B, ANSWER_ONLY, 5.0)}
    return EpisodeWorld(
        qid="q:answer", question="Phạm Văn Đồng giữ chức Thủ tướng từ năm nào?",
        split="train_strict", scenario_id="s_answer", corpus=docs,
        rankings={"bm25": ["doc_b"], "dense": ["doc_b"]},
        private=EpisodePrivate(qid="q:answer", answer_texts=(ANSWER_ONLY,),
                               is_impossible=False, gold_doc_id="doc_b"),
        reader=make_reader(preds, docs),
    )


def abstain_world() -> EpisodeWorld:
    """Unanswerable, but the reader still offers a confident *wrong* candidate.

    This is the interesting case: ANSWER and ABSTAIN are both legal, so the policy has a
    real decision to make and must learn that a high reader score is not correctness.
    """
    docs = {"doc_b": DOC_B, "doc_c": DOC_C}
    preds = {("doc_b",): span_prediction("q:abstain", "doc_b", DOC_B, "1976", 8.0)}
    return EpisodeWorld(
        qid="q:abstain", question="Ai là tổng thống Hoa Kỳ năm 1865?",
        split="train_strict", scenario_id="s_abstain", corpus=docs,
        rankings={"bm25": ["doc_b", "doc_c"], "dense": ["doc_b", "doc_c"]},
        private=EpisodePrivate(qid="q:abstain", answer_texts=(), is_impossible=True,
                               gold_doc_id=None, gold_in_corpus=False),
        reader=make_reader(preds, docs),
    )


def make_trainer(worlds, *, config: RLConfig | None = None, seed: int = 0,
                 run_dir=None) -> ReinforceTrainer:
    cfg = config or RLConfig(max_updates=40, batch_episodes=8, eval_every=10,
                             eval_episodes=16, hidden_dim=32)
    env = AdaptiveQAEnv(list(worlds), default_calculator(), EnvConfig(max_optional_search=0))
    val = AdaptiveQAEnv(list(worlds), default_calculator(), EnvConfig(max_optional_search=0))
    return ReinforceTrainer(env, val, cfg, seed=seed, run_dir=run_dir)


# ---------------------------------------------------------------- unit level
def test_discounted_returns_with_gamma_one_is_a_reverse_cumsum():
    rewards = [0.0, -0.1, 1.9]
    assert discounted_returns(rewards, 1.0) == pytest.approx([1.8, 1.8, 1.9])


def test_discounted_returns_discount_future_rewards():
    rewards = [0.0, 1.0]
    assert discounted_returns(rewards, 0.5) == pytest.approx([0.5, 1.0])


def test_mask_logits_gives_invalid_actions_no_probability():
    logits = np.array([1.0, 1.0, 1.0, 1.0])
    mask = np.array([False, False, False, True])
    masked = mask_logits(logits, mask)
    probs = np.exp(masked - masked.max())
    probs /= probs.sum()
    assert probs[ABSTAIN] == pytest.approx(1.0)
    assert probs[ANSWER] < 1e-9


def test_mask_logits_refuses_an_all_masked_state():
    with pytest.raises(ValueError, match="ABSTAIN must always be legal"):
        mask_logits(np.zeros(4), np.zeros(4, dtype=bool))


def test_features_never_contain_gold_labels():
    """Spec 05 §11: no label-bearing field may appear in the training state."""
    trainer = make_trainer([answerable_world(), abstain_world()])
    trainer.fit_scaler(episodes=4)
    env = trainer.train_env
    obs, _ = env.reset(seed=0, options={"qid": "q:answer"})
    assert find_forbidden_keys(obs) == []
    vec = trainer.extractor.vector(obs, None)
    assert np.all(np.isfinite(vec))
    assert vec.shape[0] == trainer.extractor.dim


def test_scaler_matches_declared_width_and_bounds():
    trainer = make_trainer([answerable_world(), abstain_world()])
    scaler = trainer.fit_scaler(episodes=6)
    assert len(scaler.mean) == trainer.extractor.dim
    probe = np.full((1, trainer.extractor.dim), 1e6, dtype=np.float32)
    transformed = scaler.transform(probe.reshape(1, -1))[0]
    assert np.all(np.abs(transformed) <= scaler.clip + 1e-6)


# ------------------------------------------------------------- training real
def test_reinforce_loss_uses_transition_mean_not_episode_mean():
    """One extra zero-advantage transition changes the loss denominator, not the objective."""
    from src.rl.reinforce import Episode, Transition

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


def test_one_update_changes_policy_parameters_and_yields_finite_gradients():
    """Spec 05 §11: |Δθ| > 0 and finite after a real update."""
    trainer = make_trainer([answerable_world(), abstain_world()])
    trainer.fit_scaler(episodes=8)
    net = trainer.build()
    before = np.concatenate([p.detach().numpy().ravel() for p in net.parameters()]).copy()

    episodes = [rollout(trainer.train_env, net, trainer.extractor, trainer.scaler_t, None,
                        seed=s) for s in range(6)]
    loss, stats = reinforce_loss(episodes, net, trainer.scaler_t, trainer.config,
                                 device=torch.device("cpu"))
    loss.backward()
    grad_norms = [float(p.grad.norm()) for p in net.parameters() if p.grad is not None]
    assert np.isfinite(loss.detach().item()) and np.isfinite(stats["n_steps"])
    assert any(g > 0 for g in grad_norms), "at least one layer must receive gradient"

    with torch.no_grad():
        for p in net.parameters():
            p -= 1e-3 * (p.grad if p.grad is not None else 0)
    after = np.concatenate([p.detach().numpy().ravel() for p in net.parameters()])
    assert np.abs(after - before).mean() > 0
    assert np.all(np.isfinite(after))


def test_policy_learns_to_answer_when_answering_is_always_correct():
    """Spec 05 §11: the toy reward moves the policy in the expected direction."""
    trainer = make_trainer([answerable_world()],
                           config=RLConfig(max_updates=60, batch_episodes=8, eval_every=20,
                                           eval_episodes=32, hidden_dim=32, lr=5e-3))
    net = trainer.build(d_in=trainer.extractor.dim)
    trainer.fit_scaler(episodes=4)

    before = _prob_of(trainer, net, "q:answer", ANSWER)
    trainer.train()
    after = _prob_of(trainer, net, "q:answer", ANSWER)
    assert after > before, f"P(ANSWER) should rise: {before:.3f} -> {after:.3f}"
    assert after > 0.6, f"policy should strongly prefer ANSWER, got {after:.3f}"


def test_policy_learns_to_search_when_only_the_next_passage_has_the_answer():
    world = positive_world()
    config = RLConfig(max_updates=80, batch_episodes=8, eval_every=20,
                      eval_episodes=32, hidden_dim=32, lr=5e-3)
    train_env = AdaptiveQAEnv([world], default_calculator(),
                              EnvConfig(max_optional_search=1))
    val_env = AdaptiveQAEnv([world], default_calculator(),
                            EnvConfig(max_optional_search=1))
    trainer = ReinforceTrainer(train_env, val_env, config, seed=0)
    net = trainer.build(d_in=trainer.extractor.dim)
    trainer.fit_scaler(episodes=8)
    before = _prob_of(trainer, net, world.qid, SEARCH_BM25)
    trainer.train()
    after = _prob_of(trainer, net, world.qid, SEARCH_BM25)
    episode = rollout(val_env, net, trainer.extractor, trainer.scaler_t, None,
                      qid=world.qid, greedy=True)
    assert after > before
    assert after > 0.6
    assert [t.action for t in episode.transitions] == [SEARCH_BM25, ANSWER]
    assert episode.total_reward == pytest.approx(2.0 - 0.1 - 0.1)


def test_policy_learns_to_abstain_on_unanswerable_questions():
    trainer = make_trainer([abstain_world()],
                           config=RLConfig(max_updates=60, batch_episodes=8, eval_every=20,
                                           eval_episodes=32, hidden_dim=32, lr=5e-3))
    net = trainer.build(d_in=trainer.extractor.dim)
    trainer.fit_scaler(episodes=4)
    before = _prob_of(trainer, net, "q:abstain", ABSTAIN)
    trainer.train()
    after = _prob_of(trainer, net, "q:abstain", ABSTAIN)
    assert after > before
    assert after > 0.6


def _prob_of(trainer: ReinforceTrainer, net: PolicyNet, qid: str, action: int) -> float:
    env = trainer.train_env
    obs, _ = env.reset(seed=0, options={"qid": qid})
    x = torch.as_tensor(trainer.extractor.vector(obs, None), dtype=torch.float32).unsqueeze(0)
    if trainer.scaler_t is not None:
        x = trainer.scaler_t(x)
    with torch.no_grad():
        logits, _ = net(x)
    mask = torch.as_tensor(obs["action_mask"], dtype=torch.bool).unsqueeze(0)
    probs = torch.softmax(logits.masked_fill(~mask, -1e9), dim=-1)
    return float(probs[0, action].item())


def test_rollout_obeys_mask_and_horizon_and_returns_are_finite():
    trainer = make_trainer([answerable_world(), abstain_world()])
    net = trainer.build()
    trainer.fit_scaler(episodes=4)
    for seed in range(10):
        ep = rollout(trainer.train_env, net, trainer.extractor, trainer.scaler_t, None,
                     seed=seed)
        assert ep.transitions, "an episode must contain at least one step"
        assert len(ep.transitions) <= trainer.train_env.config.max_steps
        for t in ep.transitions:
            assert t.mask[t.action], "sampled action must be legal under its own mask"
            assert t.mask[ABSTAIN], "ABSTAIN is always legal"
        assert np.isfinite(ep.total_reward)


def test_evaluate_fixed_qids_visits_each_world_once():
    trainer = make_trainer([answerable_world(), abstain_world()])
    trainer.fit_scaler(episodes=4)
    net = trainer.build()
    qids = ["q:answer", "q:abstain"]
    result = evaluate(trainer.val_env, net, trainer.extractor, trainer.scaler_t, None,
                      episodes=len(qids), qids=qids, device=torch.device("cpu"))
    expected = [rollout(trainer.val_env, net, trainer.extractor, trainer.scaler_t, None,
                        qid=qid, seed=10_000 + i, greedy=True).total_reward
                for i, qid in enumerate(qids)]
    assert result["episodes"] == 2
    assert result["mean_reward"] == pytest.approx(np.mean(expected))
    with pytest.raises(ValueError, match="duplicate"):
        evaluate(trainer.val_env, net, trainer.extractor, trainer.scaler_t, None,
                 episodes=2, qids=[qids[0], qids[0]], device=torch.device("cpu"))


def test_deterministic_evaluation_repeats_the_same_actions():
    """Spec 05 §11: greedy eval with the same seeds gives the same action sequence."""
    trainer = make_trainer([answerable_world()])
    net = trainer.build()
    trainer.fit_scaler(episodes=4)

    def run() -> list[int]:
        ep = rollout(trainer.val_env, net, trainer.extractor, trainer.scaler_t, None,
                     seed=1234, greedy=True)
        return [t.action for t in ep.transitions]

    assert run() == run()


def test_evaluate_reports_action_counts_and_finite_reward():
    trainer = make_trainer([answerable_world(), abstain_world()])
    net = trainer.build()
    trainer.fit_scaler(episodes=4)
    stats = evaluate(trainer.val_env, net, trainer.extractor, trainer.scaler_t, None,
                     episodes=8, device=torch.device("cpu"))
    assert stats["episodes"] == 8
    assert np.isfinite(stats["mean_reward"])
    assert sum(stats["action_counts"].values()) >= 8


def test_training_loop_records_a_real_log_with_param_deltas():
    trainer = make_trainer([answerable_world(), abstain_world()])
    result = trainer.train()
    assert result["updates"], "training must produce at least one logged update"
    first = result["updates"][0]
    assert np.isfinite(first["train_mean_reward"])
    assert first["param_delta"] > 0, "parameters must actually move"
    assert first["grad_norm"] > 0
    assert all(np.isfinite(first[k]) for k in ("policy_loss", "value_loss", "entropy"))


def _scripted_evaluate(monkeypatch, rewards):
    """Replace val evaluation with a scripted reward sequence (control-flow tests only."""
    from src.rl import reinforce as reinforce_mod

    scripted = iter(rewards)

    def fake(env, net, extractor, scaler_t, laya_fn, *, episodes, device, seed_base=10_000):
        return {"mean_reward": next(scripted), "std_reward": 0.0, "mean_length": 2.0,
                "episodes": episodes, "action_counts": {"0": 1, "1": 1, "2": 0, "3": 1}}

    monkeypatch.setattr(reinforce_mod, "evaluate", fake)


def test_best_checkpoint_is_written_once_and_never_overwritten(tmp_path, monkeypatch):
    """Regression: train() used to re-save best/ with the final update's weights."""
    from pathlib import Path
    trainer = make_trainer([answerable_world(), abstain_world()],
                           config=RLConfig(max_updates=3, batch_episodes=4, eval_every=1,
                                           eval_episodes=4, hidden_dim=16,
                                           early_stop_patience=99),
                           run_dir=tmp_path / "rl")
    _scripted_evaluate(monkeypatch, [1.0, 0.5, 0.8])  # best at update 1, training continues

    best_saves: list[np.ndarray] = []
    real_save = ReinforceTrainer.save

    def spy_save(self, path):
        real_save(self, path)
        if Path(path).name == "best":
            best_saves.append(_params(self.net))

    monkeypatch.setattr(ReinforceTrainer, "save", spy_save)
    result = trainer.train()

    assert result["best_update"] == 1
    assert len(best_saves) == 1, "best/ must be written exactly once — at the best update"
    best = PolicyNet.load(tmp_path / "rl" / "best" / "policy.safetensors")
    assert np.allclose(_params(best), best_saves[0])
    assert not np.allclose(_params(best), _params(trainer.net)), \
        "later gradient steps moved the weights; best/ must not hold them"


def test_early_stop_update_is_logged_like_every_other_update(tmp_path, monkeypatch):
    """Regression: the update that trips early stopping was dropped from the log."""
    from src.utils.io import read_jsonl
    trainer = make_trainer([answerable_world(), abstain_world()],
                           config=RLConfig(max_updates=10, batch_episodes=2, eval_every=1,
                                           eval_episodes=2, hidden_dim=16,
                                           early_stop_patience=1),
                           run_dir=tmp_path / "rl")
    _scripted_evaluate(monkeypatch, [1.0, 0.5])  # update 1 improves, update 2 does not

    result = trainer.train()
    assert result["stopped_early"] is True
    assert [row["update"] for row in result["updates"]] == [1, 2]
    assert result["updates"][-1]["val"]["mean_reward"] == 0.5
    assert trainer.history == result["updates"]
    log = read_jsonl(tmp_path / "rl" / "train_log.jsonl")
    assert [row["update"] for row in log] == [1, 2]


def test_features_reach_the_network_standardised_exactly_once():
    """Regression: extractor.vector() used to pre-standardise and rollout applied the
    scaler again, so the net input was doubly standardised while transition features
    (used by reinforce_loss) were not."""
    trainer = make_trainer([answerable_world(), abstain_world()])
    trainer.fit_scaler(episodes=4)
    net = trainer.build()
    captured: list[torch.Tensor] = []
    real_forward = net.forward

    def spy(x):
        captured.append(x.detach().clone())
        return real_forward(x)

    net.forward = spy
    obs, _ = trainer.train_env.reset(seed=0, options={"qid": "q:answer"})
    raw = trainer.extractor.vector(obs, None)
    expected = trainer.scaler_t(torch.as_tensor(raw, dtype=torch.float32).unsqueeze(0))
    rollout(trainer.train_env, net, trainer.extractor, trainer.scaler_t, None,
            seed=0, qid="q:answer", greedy=True)

    assert captured, "rollout must feed the network through forward()"
    assert torch.allclose(captured[0], expected, atol=1e-5), \
        "net input must be the raw features standardised exactly once"
    assert not torch.allclose(captured[0], torch.as_tensor(raw).unsqueeze(0), atol=1e-3), \
        "the scaler is not the identity here: a raw pass-through would be a different bug"


def _params(net: PolicyNet | None) -> np.ndarray:
    assert net is not None
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in net.parameters()])


def test_policy_round_trip_preserves_behaviour(tmp_path):
    trainer = make_trainer([answerable_world()])
    net = trainer.build()
    trainer.fit_scaler(episodes=4)
    trainer.save(tmp_path / "ckpt")
    reloaded = PolicyNet.load(tmp_path / "ckpt" / "policy.safetensors")
    obs, _ = trainer.train_env.reset(seed=0, options={"qid": "q:answer"})
    x = torch.as_tensor(trainer.scaler.transform(
        trainer.extractor.vector(obs, None).reshape(1, -1)), dtype=torch.float32)
    with torch.no_grad():
        a, _ = net(x)
        b, _ = reloaded(x)
    assert torch.allclose(a, b, atol=1e-6)


def test_entropy_bonus_changes_the_loss_sign_direction():
    """Higher entropy coefficient must lower the reported entropy term's contribution."""
    trainer = make_trainer([answerable_world(), abstain_world()])
    net = trainer.build()
    trainer.fit_scaler(episodes=4)
    episodes = [rollout(trainer.train_env, net, trainer.extractor, trainer.scaler_t, None,
                        seed=s) for s in range(4)]
    low = RLConfig(entropy_coef=0.0)
    high = RLConfig(entropy_coef=1.0)
    loss_low, stats_low = reinforce_loss(episodes, net, trainer.scaler_t, low,
                                         device=torch.device("cpu"))
    loss_high, stats_high = reinforce_loss(episodes, net, trainer.scaler_t, high,
                                           device=torch.device("cpu"))
    assert stats_low["entropy"] == pytest.approx(stats_high["entropy"])
    assert loss_high.detach().item() < loss_low.detach().item() + 1e-9
