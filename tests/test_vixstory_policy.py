"""VixSTORY REINFORCE policy tests (``src/rl/vixstory_policy.py``).

Integration-grade: the two-arm trainer is driven by the **real**
:class:`~src.rl.vixstory_env.VixStoryEnv` over frozen
:class:`~src.rl.vixstory_env.VixStoryReaderCache` predictions, so the single canonical feature
contract is exercised end to end — there is no mock observation schema any more. No GPU, no
real pilot data, no official-validation access.

Covered: the canonical 28-feature extractor (no fabricated extractive score, mask/4-way-prior
validation), no label leakage into observations or features, genuinely on-policy sampling, arm
isolation + identity, stale-checkpoint refusal on a safe resume, and strict qid/prior checks.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.data.schemas import find_forbidden_keys
from src.retrieval.states import reachable_states
from src.rl.policy import PolicyNet
from src.rl.reinforce import RLConfig, rollout
from src.rl.vixstory_env import (
    ABSTAIN,
    ANSWER,
    N_ACTIONS,
    SEARCH_BM25,
    SEARCH_DENSE,
    VixStoryEnv,
    VixStoryPrivate,
    VixStoryReaderCache,
    VixStoryReaderPrediction,
    VixStoryWorld,
)
from src.rl.vixstory_policy import (
    VixStoryFeatureExtractor,
    train_vixstory_policy,
    vixstory_feature_names,
)

DOCS = {
    "d0": "Trường A toạ lạc tại Hà Nội.",
    "d1": "Trường A thành lập năm 1995.",
    "d2": "Trường A hiện có 12 khoa.",
    "d3": "Đoạn nhiễu không liên quan.",
}
#: Bootstrap (BM25, 1) -> d0; the two tools then diverge to d1 / d2.
RANKINGS = {"bm25": ["d0", "d1", "d2", "d3"], "dense": ["d0", "d2", "d1", "d3"]}
ANSWER_D1 = "1995"
ANSWER_D2 = "12 khoa"
ANSWER_BOTH = "1995 và 12 khoa"
READER_SHA = "reader-1"
PRED_SHA = dict(reader_sha=READER_SHA, tokenizer_sha="tok-1", inference_config_sha="infer-1")


def _cache(qid: str, *, answer: str, gold_ids: tuple[str, ...],
           reader_sha: str = READER_SHA) -> VixStoryReaderCache:
    """Cache every reachable state: a non-blank answer only once all gold passages are visible."""
    cache = VixStoryReaderCache(**{**PRED_SHA, "reader_sha": reader_sha})
    for state in reachable_states(RANKINGS):
        complete = set(gold_ids) <= set(state)
        cache.put(qid, state, VixStoryReaderPrediction(
            qid=qid, text=answer if complete else None,
            cited_ids=tuple(gold_ids) if complete else (),
            citation_valid=bool(complete)))
    return cache


def world(qid: str, *, answer: str, gold_ids: tuple[str, ...],
          reader_sha: str = READER_SHA) -> VixStoryWorld:
    """A real :class:`VixStoryWorld` whose reader cache covers every reachable state."""
    cache = _cache(qid, answer=answer, gold_ids=gold_ids, reader_sha=reader_sha)
    return VixStoryWorld(
        qid=qid, question=f"Trường A có đặc điểm gì ({qid})?", split="train",
        scenario_id=f"vixstory:train:{qid}", corpus=DOCS, rankings=RANKINGS,
        private=VixStoryPrivate(qid=qid, answer_texts=(answer,), gold_context_ids=gold_ids,
                                hop_level=len(gold_ids), question_type="factual"),
        reader=cache.reader(qid), reader_sha=reader_sha)


TRAIN_WORLDS = [
    world("T1", answer=ANSWER_D1, gold_ids=("d1",)),
    world("T2", answer=ANSWER_D2, gold_ids=("d2",)),
    world("T3", answer=ANSWER_BOTH, gold_ids=("d1", "d2")),
]
DEV_WORLDS = [
    world("V1", answer=ANSWER_D1, gold_ids=("d1",)),
    world("V2", answer=ANSWER_D2, gold_ids=("d2",)),
]
TRAIN_QIDS = [w.qid for w in TRAIN_WORLDS]
DEV_QIDS = [w.qid for w in DEV_WORLDS]


def train_env() -> VixStoryEnv:
    return VixStoryEnv(TRAIN_WORLDS)


def dev_env() -> VixStoryEnv:
    return VixStoryEnv(DEV_WORLDS)


class SpyEnv:
    """Wraps a real env and records every qid handed to ``reset``."""

    def __init__(self, env: VixStoryEnv) -> None:
        self._env = env
        self._worlds = env._worlds
        self.reset_qids: list[str] = []

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self._env.reset(seed=seed, options=options)
        self.reset_qids.append(obs["qid"])
        return obs, info

    def step(self, action: int):
        return self._env.step(action)


def make_prior(scale: float):
    """A deterministic frozen Laya prior returning a ``Sequence[4]`` of probabilities."""

    def prior(obs: dict) -> np.ndarray:
        mask = np.asarray(obs["action_mask"], dtype=bool)
        base = np.array([0.7 * mask[ANSWER], 0.15 * scale * mask[SEARCH_BM25],
                         0.15 * scale * mask[SEARCH_DENSE], 0.2 * mask[ABSTAIN]])
        total = base.sum()
        return base / total if total > 0 else np.array([0.0, 0.0, 0.0, 1.0])

    return prior


def tiny_config(**overrides) -> RLConfig:
    base = dict(seed=7, hidden_dim=8, lr=1e-2, batch_episodes=3, max_updates=2,
                eval_every=1, eval_episodes=len(DEV_QIDS), entropy_coef=0.01, value_coef=0.5,
                grad_clip=1.0, early_stop_patience=50)
    base.update(overrides)
    return RLConfig(**base)


def train_once(run_root, *, train=None, dev=None, config=None, **kwargs):
    train = train if train is not None else train_env()
    dev = dev if dev is not None else dev_env()
    priors = {"rlcd": make_prior(1.0), "ce": make_prior(0.4)}
    return train_vixstory_policy(
        train, dev, config or tiny_config(), priors=priors, run_root=run_root,
        train_qids=TRAIN_QIDS, dev_qids=DEV_QIDS, device="cpu", **kwargs)


def canonical_obs(**overrides) -> dict:
    """A VixSTORY observation with the canonical free-form candidate schema."""
    obs = {
        "qid": "Q", "question": "Trường A thành lập năm nào?", "last_action": -1, "timestep": 0,
        "action_mask": np.array([False, True, True, True]),
        "candidate": {"text": None, "cited_ids": (), "citation_valid": False,
                      "answer_present": False},
        "retrieval": {"bm25_top_score": -1.0, "bm25_top_gap": 0.0,
                      "dense_top_score": -2.0, "dense_top_gap": 0.0},
        "seen_doc_ids": ("d0",), "seen_doc_texts": (DOCS["d0"],),
        "searches_used": 1, "optional_used": 0, "remaining_optional_searches": 2,
    }
    obs.update(overrides)
    return obs


# ---------------------------------------------------------------- extractor
def test_feature_names_exclude_extractive_scores():
    names = vixstory_feature_names()
    extractor = VixStoryFeatureExtractor()
    assert extractor.names == names
    assert extractor.dim == len(names) == 28
    for forbidden in ("non_null", "null_margin", "null_score", "span", "start_char"):
        assert not any(forbidden in name for name in names), forbidden


def test_extractor_ignores_uncalibrated_reader_scores():
    """Fabricated extractive scores smuggled into the candidate must not change the vector."""
    extractor = VixStoryFeatureExtractor()
    prior = [0.1, 0.2, 0.3, 0.4]
    plain = canonical_obs(candidate={"text": ANSWER_D1, "cited_ids": ("d1",),
                                     "citation_valid": True, "answer_present": True})
    noisy = canonical_obs(candidate={"text": ANSWER_D1, "cited_ids": ("d1",),
                                     "citation_valid": True, "answer_present": True,
                                     "non_null_score": 42.0, "null_score": -42.0,
                                     "null_margin": 84.0, "provenance_ok": True})
    np.testing.assert_array_equal(extractor.vector(plain, prior), extractor.vector(noisy, prior))


def test_extractor_reads_the_canonical_candidate_fields():
    extractor = VixStoryFeatureExtractor()
    names = extractor.names
    obs = canonical_obs(candidate={"text": ANSWER_D1, "cited_ids": ("d1", "d2"),
                                   "citation_valid": True, "answer_present": True})
    vec = extractor.vector(obs, [0.1, 0.2, 0.3, 0.4])
    assert vec[names.index("candidate_answer_present")] == 1.0
    assert vec[names.index("candidate_citation_valid")] == 1.0
    assert vec[names.index("candidate_token_len")] == 1.0  # "1995"
    assert vec[names.index("candidate_char_len")] == 4.0
    assert vec[names.index("mask_bm25")] == 1.0
    assert vec[names.index("mask_answer")] == 0.0


def test_extractor_requires_a_valid_four_way_prior():
    extractor = VixStoryFeatureExtractor()
    obs = canonical_obs()
    with pytest.raises(ValueError):
        extractor.vector(obs, [0.5, 0.5, 0.0])          # wrong width
    with pytest.raises(ValueError):
        extractor.vector(obs, [0.5, 0.5, -0.1, 0.1])     # negative mass
    with pytest.raises(ValueError):
        extractor.vector(obs, [0.5, 0.5, float("nan"), 0.0])  # non-finite mass
    missing = extractor.vector(obs, None)
    present = extractor.vector(obs, [0.1, 0.2, 0.3, 0.4])
    names = extractor.names
    assert missing[names.index("laya_missing")] == 1.0
    assert present[names.index("laya_missing")] == 0.0
    assert not np.array_equal(missing, present), "the prior must actually enter the features"


def test_extractor_rejects_a_malformed_observation():
    extractor = VixStoryFeatureExtractor()
    with pytest.raises(ValueError):
        extractor.vector({"qid": "Q"}, [0.1, 0.2, 0.3, 0.4])  # missing action_mask
    with pytest.raises(ValueError):
        extractor.vector(canonical_obs(action_mask=np.zeros(3, dtype=bool)),
                         [0.1, 0.2, 0.3, 0.4])  # wrong mask width


# ---------------------------------------------------------------- no label leak
def test_observation_carries_no_forbidden_keys():
    env = train_env()
    obs, info = env.reset(seed=0)
    assert find_forbidden_keys(obs) == []
    assert find_forbidden_keys(info) == []
    obs, _, _, _, info = env.step(SEARCH_BM25)
    assert find_forbidden_keys(obs) == []
    assert find_forbidden_keys(info) == []


def test_features_ignore_labels_smuggled_into_the_observation():
    extractor = VixStoryFeatureExtractor()
    obs = canonical_obs()
    leaky = dict(obs)
    leaky["gold_answer"] = ANSWER_D1
    leaky["is_impossible"] = False
    leaky["teacher_action"] = ANSWER
    np.testing.assert_array_equal(extractor.vector(obs, [0.1, 0.2, 0.3, 0.4]),
                                  extractor.vector(leaky, [0.1, 0.2, 0.3, 0.4]))


def test_features_do_not_depend_on_private_gold():
    """Two worlds with the same visible state but different gold -> identical features."""
    cache = _cache("Q", answer=ANSWER_D1, gold_ids=("d1",))

    def w(gold_answer: str, gold_ids: tuple[str, ...]) -> VixStoryWorld:
        return VixStoryWorld(
            qid="Q", question="?", split="train", scenario_id="sc:Q", corpus=DOCS,
            rankings=RANKINGS,
            private=VixStoryPrivate(qid="Q", answer_texts=(gold_answer,),
                                    gold_context_ids=gold_ids),
            reader=cache.reader("Q"))

    obs_a, _ = VixStoryEnv([w(ANSWER_D1, ("d1",))]).reset(seed=0, options={"qid": "Q"})
    obs_b, _ = VixStoryEnv([w("9999", ("d3",))]).reset(seed=0, options={"qid": "Q"})
    extractor = VixStoryFeatureExtractor()
    np.testing.assert_array_equal(extractor.vector(obs_a, [0.1, 0.2, 0.3, 0.4]),
                                  extractor.vector(obs_b, [0.1, 0.2, 0.3, 0.4]))


# ---------------------------------------------------------------- no fabrication
def test_trainer_refuses_a_missing_reader_cache_entry(tmp_path):
    """An uncached state is a hard error, never a fabricated candidate or score."""
    empty = VixStoryReaderCache(**PRED_SHA)
    bare = VixStoryWorld(
        qid="M1", question="?", split="train", scenario_id="sc:M1", corpus=DOCS,
        rankings=RANKINGS,
        private=VixStoryPrivate(qid="M1", answer_texts=(ANSWER_D1,), gold_context_ids=("d1",)),
        reader=empty.reader("M1"), reader_sha=READER_SHA)
    with pytest.raises(KeyError, match="cache miss"):
        train_vixstory_policy(
            VixStoryEnv([bare]), dev_env(), tiny_config(),
            priors={"rlcd": make_prior(1.0), "ce": make_prior(0.4)}, run_root=tmp_path,
            train_qids=["M1"], dev_qids=DEV_QIDS, device="cpu")


# ---------------------------------------------------------------- on-policy
def test_rollout_actions_come_from_the_current_network():
    extractor = VixStoryFeatureExtractor()
    env = train_env()
    torch.manual_seed(0)
    net = PolicyNet(extractor.dim, 8)
    episode = rollout(env, net, extractor, None, make_prior(1.0), seed=3)
    assert episode.transitions
    for t in episode.transitions:
        x = torch.as_tensor(t.features, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.as_tensor(t.mask, dtype=torch.bool).unsqueeze(0)
        logits, _ = net(x)
        dist = torch.distributions.Categorical(logits=logits.masked_fill(~mask_t, -1e9))
        recomputed = float(dist.log_prob(torch.tensor([t.action])).item())
        assert abs(recomputed - t.log_prob) < 1e-5, "action was not sampled from the live net"


def test_rollout_actions_are_stochastic():
    extractor = VixStoryFeatureExtractor()
    env = train_env()
    torch.manual_seed(0)
    net = PolicyNet(extractor.dim, 8)
    first_actions = {rollout(env, net, extractor, None, make_prior(1.0), seed=s).transitions[0].action
                     for s in range(40)}
    assert len(first_actions) > 1, "a sampling policy must not always pick the same action"


# ---------------------------------------------------------------- training
def test_trains_two_isolated_arms(tmp_path):
    results = train_once(tmp_path)
    assert set(results) == {"rlcd", "ce"}
    assert results["rlcd"].experiment == "E06"
    assert results["ce"].experiment == "E07"
    assert results["rlcd"].seed != results["ce"].seed, "arms must be independently seeded"
    for arm in ("rlcd", "ce"):
        ckpt = tmp_path / f"{results[arm].experiment}_rl_{arm}"
        for name in ("policy.safetensors", "scaler.json", "history.json", "summary.json",
                     "train_log.jsonl"):
            assert (ckpt / name).is_file(), f"{arm} missing {name}"
        history = json.loads((ckpt / "history.json").read_text())
        assert history, "training produced no history"
        first = history[0]
        assert np.isfinite(first["grad_norm"]) and first["grad_norm"] > 0.0
        assert first["param_delta"] > 0.0, "parameters must move after an update"
        assert "val" in first
        summary = json.loads((ckpt / "summary.json").read_text())
        assert summary["dev_qids"] == DEV_QIDS
        assert summary["train_qids"] == TRAIN_QIDS
        PolicyNet.load(ckpt / "policy.safetensors")


def test_explicit_matched_seeds_run_all_updates_for_both_arms(tmp_path):
    config = tiny_config(max_updates=3, early_stop_patience=1)
    result = train_once(tmp_path, config=config, arm_seeds={"rlcd": 42, "ce": 42},
                        fixed_updates=True)
    assert {arm: value.seed for arm, value in result.items()} == {"rlcd": 42, "ce": 42}
    assert all(value.n_updates == 3 and not value.stopped_early for value in result.values())


def test_selected_best_checkpoint_reloads_same_greedy_actions(tmp_path):
    from src.pipeline.vixstory_arms import ReinforcePolicy

    result = train_once(tmp_path, arm_seeds={"rlcd": 42, "ce": 42}, fixed_updates=True)
    for arm in ("rlcd", "ce"):
        best = tmp_path / f"{result[arm].experiment}_rl_{arm}" / "best"
        assert (best / "policy.safetensors").is_file()
        first = ReinforcePolicy(best, make_prior(1.0), device="cpu")
        second = ReinforcePolicy(best, make_prior(1.0), device="cpu")
        obs, _ = train_env().reset(seed=0, options={"qid": "T1"})
        assert first.act(obs) == second.act(obs)


def test_arms_use_distinct_identity_and_priors(tmp_path):
    results = train_once(tmp_path)
    summaries = {}
    for arm in ("rlcd", "ce"):
        ckpt = tmp_path / f"{results[arm].experiment}_rl_{arm}"
        summaries[arm] = json.loads((ckpt / "summary.json").read_text())
    assert summaries["rlcd"]["identity"] != summaries["ce"]["identity"]
    assert summaries["rlcd"]["seed"] != summaries["ce"]["seed"]


def test_existing_checkpoint_fails_without_resume(tmp_path):
    train_once(tmp_path)
    with pytest.raises(FileExistsError):
        train_once(tmp_path)


def test_safe_resume_and_identity_mismatch(tmp_path):
    first = train_once(tmp_path)
    resumed = train_once(tmp_path, resume=True)
    for arm in ("rlcd", "ce"):
        assert resumed[arm].status == "RESUMED"
        assert resumed[arm].n_updates == first[arm].n_updates
    with pytest.raises(ValueError):
        train_once(tmp_path, resume=True, arm_seeds={"rlcd": 999, "ce": 1000})


def test_resume_rejects_a_checkpoint_from_another_frozen_reader(tmp_path):
    """A run pinned to a different frozen reader carries a different identity -> refused."""
    train_once(tmp_path)
    other = VixStoryEnv([
        world("T1", answer=ANSWER_D1, gold_ids=("d1",), reader_sha="reader-2"),
        world("T2", answer=ANSWER_D2, gold_ids=("d2",), reader_sha="reader-2"),
        world("T3", answer=ANSWER_BOTH, gold_ids=("d1", "d2"), reader_sha="reader-2"),
    ])
    with pytest.raises(ValueError, match="identity"):
        train_once(tmp_path, train=other, resume=True)


def test_training_touches_only_train_and_dev_qids(tmp_path):
    train_spy = SpyEnv(train_env())
    dev_spy = SpyEnv(dev_env())
    train_once(tmp_path, train=train_spy, dev=dev_spy)
    assert set(train_spy.reset_qids) <= set(TRAIN_QIDS)
    assert set(dev_spy.reset_qids) <= set(DEV_QIDS)
    assert set(dev_spy.reset_qids)  # the dev env is exercised (evaluation)
    # No train qid ever appears in the dev env, and vice versa.
    assert set(train_spy.reset_qids) & set(DEV_QIDS) == set()


def test_rejects_overlapping_or_unknown_qids(tmp_path):
    priors = {"rlcd": make_prior(1.0), "ce": make_prior(0.4)}
    with pytest.raises(ValueError):
        train_vixstory_policy(
            train_env(), dev_env(), tiny_config(), priors=priors, run_root=tmp_path,
            train_qids=TRAIN_QIDS, dev_qids=[TRAIN_QIDS[0]], device="cpu")
    with pytest.raises(ValueError):
        train_vixstory_policy(
            train_env(), dev_env(), tiny_config(), priors=priors, run_root=tmp_path,
            train_qids=[*TRAIN_QIDS, "NOPE"], dev_qids=DEV_QIDS, device="cpu")


def test_rejects_wrong_prior_set(tmp_path):
    with pytest.raises(ValueError):
        train_vixstory_policy(
            train_env(), dev_env(), tiny_config(), priors={"rlcd": make_prior(1.0)},
            run_root=tmp_path, train_qids=TRAIN_QIDS, dev_qids=DEV_QIDS, device="cpu")
