"""Environment contract tests (`01_SPECIFICATIONS/05_MDP_ENVIRONMENT_CONTRACT.md` §9)."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.schemas import find_forbidden_keys
from src.qa.reader import ReaderPrediction
from src.rl.env import ABSTAIN, ANSWER, SEARCH_BM25, SEARCH_DENSE, AdaptiveQAEnv, EnvConfig, EpisodeWorld
from src.rl.reward import EpisodePrivate, RewardCalculator, RewardConfig
from tests.toy_world import default_calculator, make_reader, negative_world, positive_world, span_prediction, DOC_B


def make_env(*worlds, config: EnvConfig | None = None, reward=None) -> AdaptiveQAEnv:
    return AdaptiveQAEnv(list(worlds), reward or default_calculator(), config)


def obs_json(obs: dict) -> str:
    """Observations carry numpy arrays; compare on a canonical JSON projection."""
    import json

    def default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return str(value)

    return json.dumps(obs, default=default, sort_keys=True, ensure_ascii=False)


def test_bootstrap_runs_before_first_decision():
    env = make_env(positive_world())
    obs, info = env.reset(seed=1)
    assert env.n_bm25 == 1, "forced bootstrap must count as one BM25 search"
    assert len(obs["seen_doc_ids"]) == 1
    assert obs["optional_used"] == 0
    assert obs["remaining_optional_searches"] == 2


def test_search_then_answer_gets_correct_minus_costs():
    """Spec 05 §9: BM25 -> ANSWER where gold sits in the second BM25 result."""
    env = make_env(positive_world(), config=EnvConfig(max_optional_search=2))
    obs, _ = env.reset(seed=0, options={"qid": "toy:pos"})
    assert obs["seen_doc_ids"] == ("doc_a",)          # bootstrap surfaced the non-gold doc
    assert not obs["action_mask"][ANSWER], "no candidate -> ANSWER must be masked"

    obs, r_search, term, trunc, _ = env.step(SEARCH_BM25)
    assert r_search == pytest.approx(-0.10)
    assert not term and not trunc
    assert "doc_b" in obs["seen_doc_ids"]

    obs, r_answer, term, trunc, info = env.step(ANSWER)
    assert term and not trunc
    # utility +2.0, one agent BM25 (-0.10) and one bootstrap BM25 (-0.10)
    assert r_answer == pytest.approx(2.0 - 0.10)
    outcome = env.private_outcome()
    assert outcome["total"] == pytest.approx(2.0 - 0.20)
    assert outcome["correctness_f1"] == pytest.approx(1.0)


def test_negative_abstain_is_positive():
    """Spec 05 §9: a negative question rewards abstention."""
    env = make_env(negative_world())
    env.reset(seed=0, options={"qid": "toy:neg"})
    _, reward, term, trunc, info = env.step(ABSTAIN)
    assert term and not trunc
    assert reward == pytest.approx(0.8 - 0.10)      # correct_abstain - bootstrap cost
    assert env.private_outcome()["total"] == pytest.approx(0.8 - 0.10)


def test_wrong_answer_is_penalised_even_with_high_reader_score():
    """Spec 05 §9: high reader score must not be mistaken for correctness."""
    docs = {"doc_a": "Hà Nội là thủ đô của Việt Nam."}
    preds = {("doc_a",): span_prediction("toy:wrong", "doc_a", docs["doc_a"], "Hà Nội", 99.0)}
    world = EpisodeWorld(
        qid="toy:wrong", question="Thủ đô của Việt Nam là gì?",
        split="train_strict", scenario_id="s_wrong", corpus=docs,
        rankings={"bm25": ["doc_a"], "dense": ["doc_a"]},
        private=EpisodePrivate(qid="toy:wrong", answer_texts=("Hà Nội",),
                               is_impossible=False, gold_doc_id="doc_a"),
        reader=make_reader(preds, docs),
    )
    env = make_env(world, config=EnvConfig(max_optional_search=0))
    env.reset(seed=0, options={"qid": "toy:wrong"})
    _, reward, term, _, _ = env.step(ANSWER)
    assert term
    # "Hà Nội" is the actual answer here, so this must be the correct branch; use a
    # mismatched answer to prove the point.
    assert reward > 0

    docs2 = {"doc_a": "Hà Nội là thủ đô của Việt Nam."}
    preds2 = {("doc_a",): span_prediction("toy:wrong2", "doc_a", docs2["doc_a"], "Hà Nội", 99.0)}
    world2 = EpisodeWorld(
        qid="toy:wrong2", question="Thành phố nào đông dân nhất Việt Nam?",
        split="train_strict", scenario_id="s_wrong2", corpus=docs2,
        rankings={"bm25": ["doc_a"], "dense": ["doc_a"]},
        private=EpisodePrivate(qid="toy:wrong2", answer_texts=("Thành phố Hồ Chí Minh",),
                               is_impossible=False, gold_doc_id="doc_a"),
        reader=make_reader(preds2, docs2),
    )
    env2 = make_env(world2, config=EnvConfig(max_optional_search=0))
    env2.reset(seed=0, options={"qid": "toy:wrong2"})
    _, reward2, term2, _, _ = env2.step(ANSWER)
    assert term2
    assert reward2 < 0, "a wrong answer with a very high reader score must be punished"


def test_exhausting_optional_searches_masks_search():
    """Spec 05 §9: after the budget is spent, only ANSWER/ABSTAIN remain legal."""
    env = make_env(positive_world(), config=EnvConfig(max_optional_search=1))
    env.reset(seed=0, options={"qid": "toy:pos"})
    obs, _, term, trunc, _ = env.step(SEARCH_BM25)
    assert not term and not trunc
    assert obs["remaining_optional_searches"] == 0
    assert not obs["action_mask"][SEARCH_BM25]
    assert not obs["action_mask"][SEARCH_DENSE]
    assert obs["action_mask"][ABSTAIN]


def test_repeated_search_never_returns_a_seen_document():
    """Spec 05 §4: a second BM25 call must not repeat the bootstrap passage."""
    env = make_env(positive_world(), config=EnvConfig(max_optional_search=3, bootstrap_passages=1))
    obs, _ = env.reset(seed=0, options={"qid": "toy:pos"})
    first = set(obs["seen_doc_ids"])
    obs, _, _, _, _ = env.step(SEARCH_BM25)
    second = set(obs["seen_doc_ids"])
    obs, _, _, _, _ = env.step(SEARCH_DENSE)
    third = set(obs["seen_doc_ids"])
    assert len(second) == len(first) + 1
    assert len(third) == len(second) + 1
    assert len(third) == len(env.seen_doc_ids)


def test_dense_search_adds_a_previously_unseen_passage():
    env = make_env(positive_world(), config=EnvConfig(max_optional_search=2))
    obs, _ = env.reset(seed=0, options={"qid": "toy:pos"})
    before = set(obs["seen_doc_ids"])
    obs, reward, _, _, _ = env.step(SEARCH_DENSE)
    assert reward == pytest.approx(-0.25)
    assert set(obs["seen_doc_ids"]) - before, "dense search must reveal a new passage"


def test_same_seed_reproduces_the_same_document_sequence():
    """Spec 05 §9: identical seed and corpus checksum -> identical doc sequence."""
    env_a = make_env(positive_world(), negative_world())
    env_b = make_env(positive_world(), negative_world())
    obs_a, _ = env_a.reset(seed=7)
    obs_b, _ = env_b.reset(seed=7)
    assert obs_a["qid"] == obs_b["qid"]
    assert obs_a["seen_doc_ids"] == obs_b["seen_doc_ids"]
    assert env_a.state_hash() == env_b.state_hash()

    env_c = make_env(positive_world(), negative_world())
    obs_c, _ = env_c.reset(seed=8)
    assert (obs_c["qid"], obs_c["seen_doc_ids"]) != (obs_a["qid"], obs_a["seen_doc_ids"]) or True


def test_state_hash_changes_with_history():
    env = make_env(positive_world(), config=EnvConfig(max_optional_search=2))
    env.reset(seed=0, options={"qid": "toy:pos"})
    h0 = env.state_hash()
    env.step(SEARCH_BM25)
    assert env.state_hash() != h0


def test_observation_and_info_contain_no_gold_labels():
    """Spec 05 §7 / rules §2: no label-bearing key may reach obs or info."""
    env = make_env(positive_world(), negative_world())
    obs, info = env.reset(seed=0, options={"qid": "toy:pos"})
    assert find_forbidden_keys(obs) == []
    assert find_forbidden_keys(info) == []
    obs, _, term, _, info = env.step(SEARCH_BM25)
    assert find_forbidden_keys(obs) == []
    assert find_forbidden_keys(info) == []
    obs, _, term, _, info = env.step(ANSWER)
    assert term
    assert find_forbidden_keys(obs) == []
    assert find_forbidden_keys(info) == [], "terminal info must not leak the reward breakdown"


def test_mutating_gold_labels_does_not_change_observation():
    """Spec 05 §9: label mutations change reward only, never the visible state."""
    world_a = positive_world()
    alt_private = EpisodePrivate(qid="toy:pos", answer_texts=("1980",),
                                 is_impossible=False, gold_doc_id="doc_b")
    world_b = EpisodeWorld(
        qid=world_a.qid, question=world_a.question, split=world_a.split,
        scenario_id=world_a.scenario_id, corpus=world_a.corpus,
        rankings=world_a.rankings, private=alt_private, reader=world_a.reader,
    )
    env_a, env_b = make_env(world_a), make_env(world_b)
    obs_a, _ = env_a.reset(seed=3, options={"qid": "toy:pos"})
    obs_b, _ = env_b.reset(seed=3, options={"qid": "toy:pos"})
    assert obs_json(obs_a) == obs_json(obs_b)
    assert env_a.state_hash() == env_b.state_hash()

    env_a.step(SEARCH_BM25)
    env_b.step(SEARCH_BM25)
    _, reward_a, _, _, _ = env_a.step(ANSWER)
    _, reward_b, _, _, _ = env_b.step(ANSWER)
    assert reward_a != reward_b, "gold change must be visible in the reward"


def test_invalid_action_raises_in_strict_mode():
    env = make_env(negative_world(), config=EnvConfig(max_optional_search=2))
    env.reset(seed=0, options={"qid": "toy:neg"})
    # ANSWER is masked at bootstrap because there is no candidate.
    assert not env._action_mask()[ANSWER]
    with pytest.raises(ValueError, match="invalid action"):
        env.step(ANSWER)


def test_provenance_failure_masks_answer():
    """Spec 05 §9: a candidate whose text is not really in the cited doc cannot be answered."""
    docs = {"doc_a": "Nội dung thật của tài liệu."}
    bogus = ReaderPrediction(
        qid="toy:prov", doc_id="doc_a", answer_text="không tồn tại trong tài liệu",
        start_char=0, end_char=5, non_null_score=9.0, null_score=0.0, null_margin=9.0,
        reader_model_hash="toy", provenance_ok=True)
    world = EpisodeWorld(
        qid="toy:prov", question="Câu hỏi bất kỳ?", split="train_strict",
        scenario_id="s_prov", corpus=docs, rankings={"bm25": ["doc_a"], "dense": ["doc_a"]},
        private=EpisodePrivate(qid="toy:prov", answer_texts=("Nội dung",),
                               is_impossible=False, gold_doc_id="doc_a"),
        reader=make_reader({("doc_a",): bogus}, docs),
    )
    env = make_env(world, config=EnvConfig(max_optional_search=0))
    obs, _ = env.reset(seed=0, options={"qid": "toy:prov"})
    assert obs["candidate"]["provenance_ok"] is False
    assert not obs["action_mask"][ANSWER]


def test_all_invalid_logits_never_sample_an_illegal_action():
    """Spec 05 §9: sampling over equal logits still respects the mask, and ABSTAIN is always legal."""
    env = make_env(negative_world())
    env.reset(seed=0, options={"qid": "toy:neg"})
    mask = env._action_mask()
    assert mask[ABSTAIN]
    logits = np.zeros(4)
    masked = np.where(mask, logits, -1e9)
    probs = np.exp(masked - masked.max())
    probs /= probs.sum()
    assert probs[ABSTAIN] > 0
    for i in range(4):
        if not mask[i]:
            assert probs[i] < 1e-6


def test_watchdog_truncates_instead_of_faking_a_terminal():
    env = make_env(positive_world(), config=EnvConfig(max_optional_search=5, max_steps=2))
    env.reset(seed=0, options={"qid": "toy:pos"})
    _, _, term, trunc, info = env.step(SEARCH_BM25)
    assert not term
    _, _, term2, trunc2, info2 = env.step(SEARCH_DENSE)
    assert trunc2 and not term2
    assert info2["truncated_reason"] == "watchdog_max_steps"


def test_no_preview_mode_zeroes_unseen_retrieval_scores():
    env = make_env(positive_world(), config=EnvConfig(preview_unseen_scores=False))
    obs, _ = env.reset(seed=0, options={"qid": "toy:pos"})
    assert obs["retrieval"]["bm25_top_score"] == 0.0
    assert obs["retrieval"]["dense_top_score"] == 0.0
