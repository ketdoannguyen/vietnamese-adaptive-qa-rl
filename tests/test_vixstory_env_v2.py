"""Contract tests for the V2 evidence-retrieval environment (:mod:`src.rl.vixstory_env_v2`).

Fixtures are tiny in-memory worlds: no dataset, no reader, no API credentials. They pin the
plan's budget (1 free BM25 bootstrap passage, at most 2 optional SEARCH, at most 2 passages
per SEARCH, at most 5 seen passages, at most 3 actions), the exact reward, the mask rules,
the public/private split and the pure ``initial_state`` / ``transition`` helpers the teacher
enumerates with.
"""

from __future__ import annotations

import copy
import inspect
import json

import pytest

from src.decision.schema_v2 import N_ACTIONS, assert_public_state_is_clean
from src.rl import vixstory_env_v2
from src.rl.vixstory_env_v2 import (
    BOOTSTRAP_PASSAGES,
    COVERAGE_WEIGHT,
    FULL_SUCCESS_BONUS,
    MAX_ACTIONS,
    MAX_OPTIONAL_SEARCHES,
    MAX_SEEN_DOCS,
    PASSAGES_PER_SEARCH,
    PUBLIC_OBS_KEYS,
    SEARCH_BM25,
    SEARCH_COST,
    SEARCH_DENSE,
    STOP,
    VixStoryEnvV2,
    VixStoryWorldV2,
    initial_state,
    stop_reward,
    transition,
    validate_obs,
)

CORPUS = {
    "d0": "Đoạn 0: trường A toạ lạc tại Hà Nội.",
    "d1": "Đoạn 1: trường A thành lập năm 1995.",
    "d2": "Đoạn 2: trường A có 12 khoa.",
    "d3": "Đoạn 3: trường A có 3 viện nghiên cứu.",
    "d4": "Đoạn 4: thư viện trường A có 20.000 đầu sách.",
    "d5": "Đoạn 5: trường A có 500 giảng viên.",
    "d9": "Đoạn 9: không xuất hiện trong bảng xếp hạng nào.",
}
QID = "q_env"
RANKINGS = {"bm25": ["d0", "d1", "d2", "d3"], "dense": ["d4", "d5", "d1", "d3"]}
GOLD = ("d1", "d2")
QUESTION = "Trường A thành lập năm nào và có bao nhiêu khoa?"


def world(qid: str = QID, *, rankings=RANKINGS, gold=GOLD, question: str = QUESTION,
          corpus=CORPUS) -> VixStoryWorldV2:
    return VixStoryWorldV2(qid=qid, question=question, corpus=corpus, rankings=rankings,
                           gold_context_ids=gold)


def env(w: VixStoryWorldV2) -> VixStoryEnvV2:
    return VixStoryEnvV2([w])


# --- bootstrap, budget and masks ---------------------------------------------------------

def test_public_observation_key_set_is_exactly_the_contract():
    obs, info = env(world()).reset(qid=QID)
    assert set(obs) == set(PUBLIC_OBS_KEYS)
    assert obs["qid"] == QID and obs["question"] == QUESTION
    assert obs["seen_doc_ids"] == ("d0",)                      # free BM25 bootstrap passage
    assert obs["seen_doc_texts"] == (CORPUS["d0"],)
    assert obs["history"] == [] and obs["optional_used"] == 0
    assert obs["remaining_optional_searches"] == MAX_OPTIONAL_SEARCHES == 2
    assert obs["action_mask"] == [True, True, True]
    assert set(info) == {"qid", "timestep"}
    assert_public_state_is_clean(obs)
    assert_public_state_is_clean(info)
    json.dumps(obs, ensure_ascii=False)                        # public obs is JSON-serializable


def test_bootstrap_budget_constants_match_the_plan():
    assert BOOTSTRAP_PASSAGES == 1 and PASSAGES_PER_SEARCH == 2
    assert MAX_OPTIONAL_SEARCHES == 2 and MAX_SEEN_DOCS == 5 and MAX_ACTIONS == 3
    assert BOOTSTRAP_PASSAGES + PASSAGES_PER_SEARCH * MAX_OPTIONAL_SEARCHES == MAX_SEEN_DOCS
    assert MAX_ACTIONS == MAX_OPTIONAL_SEARCHES + 1
    assert SEARCH_COST == 0.05 and COVERAGE_WEIGHT == 0.5 and FULL_SUCCESS_BONUS == 1.0


def test_search_reveals_two_unseen_documents_in_ranking_order():
    e = env(world())
    e.reset(qid=QID)
    obs, reward, terminated, truncated, _ = e.step(SEARCH_BM25)
    assert reward == pytest.approx(-SEARCH_COST) and not terminated and not truncated
    assert obs["seen_doc_ids"] == ("d0", "d1", "d2")            # next two unseen BM25 docs
    assert obs["history"] == ["SEARCH_BM25"] and obs["optional_used"] == 1
    assert obs["remaining_optional_searches"] == 1
    assert obs["action_mask"] == [True, True, True]             # dense still has unseen docs


def test_second_search_skips_documents_already_seen():
    e = env(world())
    e.reset(qid=QID)
    e.step(SEARCH_BM25)
    obs, reward, _, _, _ = e.step(SEARCH_DENSE)
    assert reward == pytest.approx(-SEARCH_COST)
    assert obs["seen_doc_ids"] == ("d0", "d1", "d2", "d4", "d5")   # d1/d3 already seen
    assert len(obs["seen_doc_ids"]) == MAX_SEEN_DOCS == 5
    assert obs["action_mask"] == [True, False, False]
    assert obs["history"] == ["SEARCH_BM25", "SEARCH_DENSE"]


def test_budget_exhaustion_leaves_stop_as_the_only_legal_action():
    e = env(world())
    e.reset(qid=QID)
    e.step(SEARCH_BM25)
    obs, _, _, _, _ = e.step(SEARCH_BM25)
    assert obs["action_mask"] == [True, False, False]
    assert obs["remaining_optional_searches"] == 0
    with pytest.raises(ValueError, match="SEARCH_DENSE"):
        e.step(SEARCH_DENSE)
    obs, reward, terminated, _, _ = e.step(STOP)
    assert terminated and obs["history"] == ["SEARCH_BM25", "SEARCH_BM25", "STOP"]
    assert len(obs["history"]) == MAX_ACTIONS == 3


def test_a_search_that_cannot_reveal_anything_new_is_illegal():
    w = world(rankings={"bm25": ["d0", "d1", "d2", "d3"], "dense": ["d0"]})
    e = env(w)
    obs, _ = e.reset(qid=QID)
    assert obs["action_mask"] == [True, True, False]           # dense ranking is exhausted
    with pytest.raises(ValueError, match="SEARCH_DENSE"):
        e.step(SEARCH_DENSE)
    obs, _, _, _, _ = e.step(SEARCH_BM25)
    assert obs["seen_doc_ids"] == ("d0", "d1", "d2")
    assert obs["action_mask"] == [True, True, False]           # dense still exhausted


def test_an_exhausted_ranking_reveals_only_what_remains():
    w = world(rankings={"bm25": ["d0", "d1"], "dense": ["d0", "d9"]})
    e = env(w)
    e.reset(qid=QID)
    obs, _, _, _, _ = e.step(SEARCH_BM25)                      # only d1 is left unseen
    assert obs["seen_doc_ids"] == ("d0", "d1")
    assert obs["action_mask"] == [True, False, True]           # BM25 exhausted, dense not
    obs, _, _, _, _ = e.step(SEARCH_DENSE)                     # only d9 is left unseen
    assert obs["seen_doc_ids"] == ("d0", "d1", "d9")
    assert obs["action_mask"] == [True, False, False]          # budget exhausted


def test_stop_is_always_legal():
    e = env(world())
    obs, _ = e.reset(qid=QID)
    assert obs["action_mask"][STOP]
    obs, _, _, _, _ = e.step(SEARCH_BM25)
    assert obs["action_mask"][STOP]
    obs, _, _, _, _ = e.step(SEARCH_BM25)
    assert obs["action_mask"][STOP]


# --- reward (hand cases) ----------------------------------------------------------------

@pytest.mark.parametrize("gold,searches,expected_reward,expected_total", [
    # coverage 0 -> 0.5*0 + 0 = 0.0, total = -0.05 * n_search
    (("d1", "d2"), 0, 0.0, 0.0),
    # one BM25 search reveals d1+d2: coverage 1, full gold -> 1.5, total 1.45
    (("d1", "d2"), 1, 1.5, 1.45),
    # one BM25 search covers d1 only: coverage 0.5, no full success -> 0.25, total 0.20
    (("d1", "d4"), 1, 0.25, 0.20),
    # two searches: same 1.5 terminal reward minus 0.10 -> total 1.40
    (("d1", "d2"), 2, 1.5, 1.40),
    # d9 is in no ranking: zero coverage after one search -> 0.0, total -0.05
    (("d9",), 1, 0.0, -0.05),
])
def test_stop_reward_and_episode_return_match_the_plan(gold, searches, expected_reward,
                                                       expected_total):
    e = env(world(gold=gold))
    e.reset(qid=QID)
    total = 0.0
    for _ in range(searches):
        _, reward, _, _, _ = e.step(SEARCH_BM25)
        total += reward
    _, reward, terminated, truncated, _ = e.step(STOP)
    total += reward
    assert terminated and not truncated
    assert reward == pytest.approx(expected_reward)            # the plan's STOP reward
    assert total == pytest.approx(expected_total)              # reward minus SEARCH costs
    assert e.private_outcome()["n_search"] == searches


def test_stop_records_the_ordered_final_evidence_set():
    e = env(world())
    e.reset(qid=QID)
    e.step(SEARCH_BM25)
    obs, reward, terminated, truncated, _ = e.step(STOP)
    assert terminated and not truncated
    assert reward == pytest.approx(1.5)
    assert obs["seen_doc_ids"] == ("d0", "d1", "d2")           # ordered, unchanged by STOP
    assert obs["history"] == ["SEARCH_BM25", "STOP"]
    outcome = e.private_outcome()
    assert outcome["seen_doc_ids"] == ["d0", "d1", "d2"]
    assert outcome["gold_context_ids"] == list(GOLD)
    assert outcome["n_search"] == 1 and outcome["terminal_action"] == "STOP"


def test_stop_reward_helper_matches_the_environment():
    w = world()
    obs = initial_state(w)
    assert stop_reward(w, obs) == 0.0                          # no gold coverage yet
    after_bm25 = transition(w, obs, SEARCH_BM25)
    assert stop_reward(w, after_bm25) == pytest.approx(1.5)    # full gold: 0.5*1 + 1.0
    w_partial = world(gold=("d1", "d4"))
    assert stop_reward(w_partial, transition(w_partial, initial_state(w_partial),
                                             SEARCH_BM25)) == pytest.approx(0.25)


# --- episode lifecycle ------------------------------------------------------------------

def test_private_outcome_is_none_until_stop_and_exposes_only_private_bookkeeping():
    e = env(world(gold=("d9",)))
    e.reset(qid=QID)
    assert e.private_outcome() is None
    e.step(SEARCH_BM25)
    assert e.private_outcome() is None
    e.step(STOP)
    outcome = e.private_outcome()
    assert set(outcome) == {"qid", "terminal_action", "n_search", "seen_doc_ids",
                            "gold_context_ids"}
    assert outcome["qid"] == QID and outcome["n_search"] == 1
    assert outcome["gold_context_ids"] == ["d9"]               # gold lives only here


def test_step_before_reset_and_after_termination_raise():
    e = env(world())
    with pytest.raises(RuntimeError):
        e.step(STOP)
    e.reset(qid=QID)
    e.step(STOP)
    with pytest.raises(RuntimeError):
        e.step(STOP)


@pytest.mark.parametrize("action", [-1, N_ACTIONS, 1.5, "STOP", None, True])
def test_step_rejects_non_integer_or_out_of_range_actions(action):
    e = env(world())
    e.reset(qid=QID)
    with pytest.raises(ValueError):
        e.step(action)


def test_illegal_search_reports_the_action_name():
    e = env(world())
    e.reset(qid=QID)
    e.step(SEARCH_BM25)
    e.step(SEARCH_BM25)
    with pytest.raises(ValueError, match="SEARCH_BM25"):
        e.step(SEARCH_BM25)


def test_returned_observation_is_a_copy_the_caller_cannot_corrupt():
    e = env(world())
    obs, _ = e.reset(qid=QID)
    obs["action_mask"][SEARCH_DENSE] = False
    obs["seen_doc_ids"] = ()
    obs, _, _, _, _ = e.step(SEARCH_DENSE)                     # still legal for the env
    assert obs["seen_doc_ids"] == ("d0", "d4", "d5")


# --- reset selection --------------------------------------------------------------------

def test_reset_selects_worlds_by_qid_or_option_and_rejects_conflicts():
    a, b = world("q_a"), world("q_b")
    e = VixStoryEnvV2([a, b])
    assert e.reset(qid="q_a")[0]["qid"] == "q_a"
    assert e.reset(options={"qid": "q_b"})[0]["qid"] == "q_b"
    assert e.reset(seed=42)[0]["qid"] == e.reset(seed=42)[0]["qid"]
    with pytest.raises(KeyError):
        e.reset(qid="q_missing")
    with pytest.raises(ValueError, match="conflicting"):
        e.reset(qid="q_a", options={"qid": "q_b"})
    with pytest.raises(ValueError, match="options"):
        e.reset(options={"unknown": 1})


def test_environment_rejects_empty_or_duplicate_worlds():
    with pytest.raises(ValueError):
        VixStoryEnvV2([])
    with pytest.raises(ValueError, match="duplicate"):
        VixStoryEnvV2([world("q_a"), world("q_a")])


# --- private/public separation ----------------------------------------------------------

def test_observation_and_info_never_carry_unretrieved_gold():
    w = world(gold=("d9",))                                    # d9 appears in no ranking
    e = env(w)
    obs, info = e.reset(qid=QID)
    for action in (SEARCH_BM25, SEARCH_DENSE, STOP):
        obs, _, _, _, info = e.step(action)
        assert set(obs) == set(PUBLIC_OBS_KEYS)
        assert "d9" not in repr(obs) and "d9" not in repr(info)
    assert "d9" in repr(e.private_outcome())                   # private side only


def test_retrieved_gold_id_is_a_legal_public_document():
    w = world(gold=("d1",))
    obs = transition(w, initial_state(w), SEARCH_BM25)         # d1 is gold *and* retrieved
    assert "d1" in obs["seen_doc_ids"]
    assert_public_state_is_clean(obs)                          # no false sentinel rejection


def test_no_reader_scores_or_candidate_fields_exist_in_v2():
    obs, _ = env(world()).reset(qid=QID)
    assert "candidate" not in obs and "retrieval" not in obs
    source = inspect.getsource(vixstory_env_v2)
    assert "src.qa" not in source and "requests" not in source and "openai" not in source


# --- pure helpers -----------------------------------------------------------------------

def test_initial_state_is_the_reset_observation():
    w = world()
    assert initial_state(w) == env(w).reset(qid=QID)[0]


def test_transition_is_pure_and_does_not_mutate_its_input():
    w = world()
    obs = initial_state(w)
    snapshot = copy.deepcopy(obs)
    first = transition(w, obs, SEARCH_BM25)
    second = transition(w, obs, SEARCH_BM25)
    assert first == second
    assert obs == snapshot
    assert first["history"] == ["SEARCH_BM25"] and first["optional_used"] == 1


def test_transition_stop_only_appends_the_terminal_action():
    w = world()
    obs = transition(w, initial_state(w), STOP)
    assert obs["seen_doc_ids"] == ("d0",)
    assert obs["history"] == ["STOP"] and obs["optional_used"] == 0


@pytest.mark.parametrize("action", [STOP, SEARCH_BM25, SEARCH_DENSE])
def test_pure_transition_rejects_actions_after_stop(action):
    w = world()
    stopped = transition(w, initial_state(w), STOP)
    with pytest.raises(ValueError, match="terminated"):
        transition(w, stopped, action)


@pytest.mark.parametrize("history", [["STOP", "STOP"], ["STOP", "SEARCH_BM25"]])
def test_validate_obs_rejects_nonterminal_or_repeated_stop(history):
    w = world()
    obs = initial_state(w)
    if "SEARCH_BM25" in history:
        obs = transition(w, obs, SEARCH_BM25)
    obs["history"] = history
    with pytest.raises(ValueError, match="STOP"):
        validate_obs(w, obs)


def test_validate_obs_rejects_evidence_beyond_the_search_budget():
    w = world()
    obs = initial_state(w)
    obs["seen_doc_ids"] = ("d0", "d1", "d2")
    obs["seen_doc_texts"] = tuple(w.corpus[doc_id] for doc_id in obs["seen_doc_ids"])
    with pytest.raises(ValueError, match="budget"):
        validate_obs(w, obs)


def test_transition_rejects_foreign_qids_and_illegal_actions():
    w = world()
    other = initial_state(world("q_other"))
    with pytest.raises(ValueError, match="q_other"):
        transition(w, other, SEARCH_BM25)
    w_dense_exhausted = world(rankings={"bm25": ["d0", "d1"], "dense": ["d0"]})
    with pytest.raises(ValueError, match="SEARCH_DENSE"):
        transition(w_dense_exhausted, initial_state(w_dense_exhausted), SEARCH_DENSE)


def test_validate_obs_fails_closed_on_malformed_public_states():
    w = world()
    obs = initial_state(w)

    with pytest.raises(ValueError, match="gold_context_ids"):
        bad = dict(obs, gold_context_ids=["d1"])
        validate_obs(w, bad)

    with pytest.raises(ValueError, match="keys"):
        bad = dict(obs, candidate={"text": "x"})
        validate_obs(w, bad)

    with pytest.raises(ValueError, match="mask"):
        bad = dict(obs, action_mask=[True, True, False])
        validate_obs(w, bad)

    with pytest.raises(ValueError, match="corpus"):
        bad = dict(obs, seen_doc_texts=("văn bản khác",))
        validate_obs(w, bad)

    with pytest.raises(ValueError, match="optional_used"):
        bad = dict(obs, optional_used=1, remaining_optional_searches=1)
        validate_obs(w, bad)

    with pytest.raises(ValueError, match="history"):
        bad = dict(obs, history=["SEARCH_E5"])
        validate_obs(w, bad)

    validate_obs(w, obs)                                       # the canonical obs passes


# --- world validation (fail closed) -----------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"gold": ()},                                              # empty gold
    {"gold": ("d1", "d1")},                                    # duplicate gold id
    {"gold": ("d1", "missing")},                               # gold outside the corpus
    {"gold": ("d1", 7)},                                       # non-string gold id
    {"rankings": {"bm25": ["d0", "d1"]}},                      # missing dense ranking
    {"rankings": {"bm25": ["d0"], "dense": ["d0"], "e5": ["d0"]}},   # unknown tool
    {"rankings": {"bm25": ["d0", "ghost"], "dense": ["d0"]}},  # doc outside the corpus
    {"rankings": {"bm25": ["d0", "d1", "d1"], "dense": ["d0"]}},     # duplicate in a ranking
    {"rankings": {"bm25": [], "dense": ["d0"]}},               # empty ranking
    {"rankings": ["bm25", "dense"]},                           # not a mapping
    {"corpus": {}},                                            # empty corpus
    {"corpus": {"d0": 7}},                                     # non-string text
    {"qid": ""},                                               # empty qid
    {"question": "  "},                                        # blank question
])
def test_world_validation_rejects_bad_inputs(kwargs):
    defaults = {"qid": QID, "question": QUESTION, "corpus": CORPUS, "rankings": RANKINGS,
                "gold": GOLD}
    defaults.update(kwargs)
    with pytest.raises(ValueError):
        world(**defaults)


def test_world_is_an_immutable_dataclass():
    import dataclasses

    w = world()
    with pytest.raises(dataclasses.FrozenInstanceError):
        w.qid = "khác"
    assert dataclasses.is_dataclass(w)
