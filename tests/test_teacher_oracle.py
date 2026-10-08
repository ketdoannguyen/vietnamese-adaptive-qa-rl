"""Teacher-oracle quality gates — `01_SPECIFICATIONS/07_...md` §6.

Gate 1 is the important one: oracle Q-values are compared against values worked out by
hand for tiny scenarios before any Laya training is allowed to depend on them.
"""

from __future__ import annotations

import pytest

from src.decision.teacher import (
    OracleConfig,
    _pick_expected,
    _soft_distribution,
    oracle_episode,
    verify_against_hand_enumeration,
)
from src.rl.env import ANSWER, SEARCH_BM25, SEARCH_DENSE, AdaptiveQAEnv, EnvConfig, EpisodeWorld
from src.rl.reward import EpisodePrivate
from tests.toy_world import DOC_A, DOC_B, DOC_C, DOC_D, default_calculator, make_reader, negative_world, positive_world, span_prediction


def make_env(*worlds, max_optional_search: int = 2) -> AdaptiveQAEnv:
    return AdaptiveQAEnv(list(worlds), default_calculator(),
                         EnvConfig(max_optional_search=max_optional_search))


def test_oracle_matches_hand_enumeration_on_the_search_recovery_case():
    """Bootstrap shows a useless passage; BM25 #2 recovers the answer.

    Hand calculation, reward = {correct 2.0, wrong -2.0, good abstain 0.8,
    bad abstain -1.0, bm25 0.10, dense 0.25}:

        s0 (seen=[doc_a], no candidate, 2 searches left)
            ABSTAIN      = -1.0                     (answerable -> unnecessary)
            SEARCH_BM25  = -0.10 + V(s1) = -0.10 + 2.0 =  1.90
            SEARCH_DENSE = -0.25 + V(s1) = -0.25 + 2.0 =  1.75
            argmax = SEARCH_BM25

        s1 (seen=[doc_a, doc_b], candidate "1976", 1 search left)
            ANSWER  = 2.0
            ABSTAIN = -1.0
    """
    env = make_env(positive_world(), max_optional_search=2)
    episode = oracle_episode(env, "toy:pos", OracleConfig(mode="hard"), seed=0)
    hand = {
        (): {"ABSTAIN": -1.0, "SEARCH_BM25": 1.90, "SEARCH_DENSE": 1.75},
        (SEARCH_BM25,): {"ANSWER": 2.0, "ABSTAIN": -1.0},
    }
    verify_against_hand_enumeration(episode, hand)

    first = next(t for t in episode.targets if t.history == ())
    assert first.expected == "SEARCH_BM25"
    assert "ANSWER" not in first.valid_actions, "no candidate -> ANSWER must be illegal"
    assert first.soft.get("ANSWER", 0.0) == 0.0


def test_oracle_prefers_answer_when_the_candidate_is_already_correct():
    # doc_c is an unread distractor: it keeps the search actions legal so the oracle
    # actually has to weigh "answer now" against "keep looking".
    docs = {"doc_b": DOC_B, "doc_c": DOC_C}
    preds = {("doc_b",): span_prediction("q:easy", "doc_b", DOC_B, "1976", 5.0),
             ("doc_c",): span_prediction("q:easy", "doc_c", DOC_C, "1945", 1.0),
             ("doc_b", "doc_c"): span_prediction("q:easy", "doc_b", DOC_B, "1976", 5.0)}
    world = EpisodeWorld(
        qid="q:easy", question="Phạm Văn Đồng giữ chức Thủ tướng từ năm nào?",
        split="train_strict", scenario_id="s_easy", corpus=docs,
        rankings={"bm25": ["doc_b", "doc_c"], "dense": ["doc_b", "doc_c"]},
        private=EpisodePrivate(qid="q:easy", answer_texts=("1976",),
                               is_impossible=False, gold_doc_id="doc_b"),
        reader=make_reader(preds, docs),
    )
    episode = oracle_episode(make_env(world, max_optional_search=2), "q:easy",
                             OracleConfig(mode="hard"), seed=0)
    root = episode.targets[0]
    assert root.expected == "ANSWER"
    assert root.q_values["ANSWER"] == pytest.approx(2.0)
    # Searching costs money and cannot beat an already-correct candidate.
    assert root.q_values["SEARCH_BM25"] < root.q_values["ANSWER"]


def test_oracle_abstains_on_a_negative():
    env = make_env(negative_world(), max_optional_search=2)
    episode = oracle_episode(env, "toy:neg", OracleConfig(mode="hard"), seed=0)
    root = episode.targets[0]
    assert root.expected == "ABSTAIN"
    assert root.q_values["ABSTAIN"] == pytest.approx(0.8)


def test_oracle_never_gives_masked_actions_probability_mass():
    """Spec 07 §6.2: masked actions must carry exactly zero, then renormalise."""
    env = make_env(positive_world(), max_optional_search=0)
    episode = oracle_episode(env, "toy:pos", OracleConfig(mode="soft"), seed=0)
    root = episode.targets[0]
    assert ANSWER not in root.valid_actions
    assert root.soft.get("ANSWER", 0.0) == 0.0
    # With no searches left only ABSTAIN remains, so it carries all the mass.
    assert root.soft.get("ABSTAIN", 0.0) == pytest.approx(1.0)
    assert sum(root.soft.values()) == pytest.approx(1.0)


def test_oracle_reports_ties_deterministically():
    """Equal Q for SEARCH_BM25 and SEARCH_DENSE is broken by the pre-registered order."""
    values = {"ANSWER": 1.0, "ABSTAIN": 1.0, "SEARCH_BM25": 1.0, "SEARCH_DENSE": 0.0}
    assert _pick_expected(values, ("ANSWER", "ABSTAIN", "SEARCH_BM25", "SEARCH_DENSE")) == "ANSWER"


def test_soft_distribution_puts_more_mass_on_better_actions():
    values = {"ANSWER": 2.0, "ABSTAIN": -1.0}
    soft = _soft_distribution(values, OracleConfig(temperature=0.5))
    assert soft["ANSWER"] > soft["ABSTAIN"]
    assert sum(soft.values()) == pytest.approx(1.0)
    # Lower temperature sharpens the distribution.
    sharp = _soft_distribution(values, OracleConfig(temperature=0.05))
    assert sharp["ANSWER"] > soft["ANSWER"]


def test_oracle_state_text_contains_no_gold_fields():
    """Spec 07 §6.3: changing only private labels must not change the serialized state."""
    env_a = make_env(positive_world(), max_optional_search=2)
    episode_a = oracle_episode(env_a, "toy:pos", OracleConfig(mode="soft"), seed=0)

    world_b = positive_world()
    alt = EpisodeWorld(
        qid=world_b.qid, question=world_b.question, split=world_b.split,
        scenario_id=world_b.scenario_id, corpus=world_b.corpus, rankings=world_b.rankings,
        private=EpisodePrivate(qid="toy:pos", answer_texts=("9999",),
                               is_impossible=False, gold_doc_id="doc_c"),
        reader=world_b.reader,
    )
    env_b = make_env(alt, max_optional_search=2)
    episode_b = oracle_episode(env_b, "toy:pos", OracleConfig(mode="soft"), seed=0)

    texts_a = {t.history: t.state_text for t in episode_a.targets}
    texts_b = {t.history: t.state_text for t in episode_b.targets}
    assert texts_a == texts_b, "state text must depend on visible evidence only"
    # ...but the targets themselves must differ, because the reward changed.
    assert {t.history: t.expected for t in episode_a.targets} != \
           {t.history: t.expected for t in episode_b.targets}


def test_oracle_is_reproducible_across_processes():
    """Spec 07 §6.4: same seed + manifest -> identical Q and distributions."""
    first = oracle_episode(make_env(positive_world()), "toy:pos", OracleConfig(mode="soft"), seed=7)
    second = oracle_episode(make_env(positive_world()), "toy:pos", OracleConfig(mode="soft"), seed=7)
    assert [t.state_key for t in first.targets] == [t.state_key for t in second.targets]
    assert [t.q_values for t in first.targets] == [t.q_values for t in second.targets]
    assert [t.soft for t in first.targets] == [t.soft for t in second.targets]


def test_oracle_respects_the_state_cap():
    env = make_env(positive_world(), max_optional_search=2)
    episode = oracle_episode(env, "toy:pos", OracleConfig(max_states_per_question=2), seed=0)
    assert len(episode.targets) <= 2


def test_oracle_handles_an_exhausted_budget_state():
    """At the horizon the only legal actions are terminal ones."""
    env = make_env(positive_world(), max_optional_search=1)
    episode = oracle_episode(env, "toy:pos", OracleConfig(mode="hard"), seed=0)
    exhausted = [t for t in episode.targets if set(t.valid_actions) <= {"ANSWER", "ABSTAIN"}]
    assert exhausted, "an exhausted-budget state must be reachable in this fixture"


def test_oracle_q_values_are_all_finite():
    for world in (positive_world(), negative_world()):
        episode = oracle_episode(make_env(world), world.qid, OracleConfig(mode="soft"), seed=0)
        for target in episode.targets:
            for name, value in target.q_values.items():
                assert value == value and abs(value) != float("inf"), (world.qid, name, value)
