"""Tests for the V2 CPU gold teacher (:mod:`src.pipeline.vixstory_teacher_v2`).

Small in-memory worlds only: no dataset, no reader, no GPU. Hand-computed Bellman values,
collision histories, remaining-budget memoization, the tie order (STOP, SEARCH_BM25,
SEARCH_DENSE, i.e. ACTION_ORDER) and an *independent* brute-force replay through
:class:`~src.rl.vixstory_env_v2.VixStoryEnvV2` rewards all appear here. The gold-only
:func:`retrieval_metrics` is pinned with the plan's worked examples.
"""

from __future__ import annotations

import json
import math

import pytest

from src.decision.schema_v2 import ACTION_ORDER, N_ACTIONS, assert_public_state_is_clean, laya_record
from src.pipeline.vixstory_teacher_v2 import VixStoryTeacherV2, retrieval_metrics
from src.rl.vixstory_env_v2 import (
    SEARCH_BM25,
    SEARCH_COST,
    SEARCH_DENSE,
    STOP,
    VixStoryEnvV2,
    initial_state,
    transition,
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
QID = "q_teacher"
RANKINGS = {"bm25": ["d0", "d1", "d2", "d3"], "dense": ["d4", "d5", "d1", "d3"]}


def make_world(*, rankings=RANKINGS, gold=("d1", "d2"), qid: str = QID):
    from src.rl.vixstory_env_v2 import VixStoryWorldV2

    return VixStoryWorldV2(qid=qid, question="Trường A thành lập năm nào và có bao nhiêu khoa?",
                           corpus=CORPUS, rankings=rankings, gold_context_ids=gold)


def teacher(*, rankings=RANKINGS, gold=("d1", "d2"), temperature: float = 0.5,
            qid: str = QID) -> VixStoryTeacherV2:
    return VixStoryTeacherV2(make_world(rankings=rankings, gold=gold, qid=qid),
                             temperature=temperature)


# --- independent replay through the environment -----------------------------------------

def _replay(world, history):
    """A fresh environment replayed to the state reached by ``history``."""
    env = VixStoryEnvV2([world])
    obs, _ = env.reset(qid=world.qid)
    for name in history:
        obs, *_ = env.step(ACTION_ORDER.index(name))
    return env, obs


def _brute_force_q(world, obs):
    """Bellman values computed only from the environment's own step rewards."""
    def value(current):
        return max(q_values(current))

    def q_values(current):
        out = [float("-inf")] * N_ACTIONS
        for action in [i for i, ok in enumerate(current["action_mask"]) if ok]:
            env, _ = _replay(world, current["history"])
            next_obs, reward, terminated, _, _ = env.step(action)
            out[action] = reward if terminated else reward + value(next_obs)
        return out

    return q_values(obs)


def assert_q_equal(got, want):
    assert len(got) == len(want) == N_ACTIONS
    for index, (a, b) in enumerate(zip(got, want)):
        if math.isinf(a) or math.isinf(b):
            assert a == b, f"action {ACTION_ORDER[index]}: {a} != {b}"
        else:
            assert a == pytest.approx(b), f"action {ACTION_ORDER[index]}: {a} != {b}"


# --- state enumeration -------------------------------------------------------------------

def test_enumeration_covers_the_max_two_search_tree():
    states = teacher().enumerate_states()
    depths = [len(obs["history"]) for obs in states]
    assert sorted(depths) == [0, 1, 1, 2, 2, 2, 2]             # root + 2 + 4
    assert states[0] == initial_state(make_world())            # root first, canonical obs
    identities = {(tuple(o["seen_doc_ids"]), tuple(o["history"])) for o in states}
    assert len(identities) == len(states)                      # no state enumerated twice
    for obs in states:
        assert set(obs) == set(states[0])
        assert_public_state_is_clean(obs)


def test_enumeration_retains_colliding_histories_with_distinct_state_text():
    identical = {"bm25": ["d0", "d1", "d2", "d3"], "dense": ["d0", "d1", "d2", "d3"]}
    states = teacher(rankings=identical).enumerate_states()
    depth_one = [o for o in states if o["history"] in (["SEARCH_BM25"], ["SEARCH_DENSE"])]
    assert [o["seen_doc_ids"] for o in depth_one] == [("d0", "d1", "d2"), ("d0", "d1", "d2")]
    assert {tuple(o["history"]) for o in depth_one} == {("SEARCH_BM25",), ("SEARCH_DENSE",)}
    depth_two = [o for o in states if len(o["history"]) == 2]
    assert len(depth_two) == 4 and len({tuple(o["history"]) for o in depth_two}) == 4
    assert len({tuple(o["seen_doc_ids"]) for o in depth_two}) == 1   # same docs, 4 histories


def test_enumerated_states_replay_exactly_through_the_environment():
    world = make_world()
    for obs in teacher().enumerate_states():
        _, replayed = _replay(world, obs["history"])
        assert obs == replayed


def test_enumeration_is_stable_and_defensive():
    t = teacher()
    first = t.enumerate_states()
    first[0]["seen_doc_ids"] = ("corrupted",)
    first[0]["history"].append("SEARCH_DENSE")
    first[0]["action_mask"][0] = False
    first.append({"qid": "rác"})
    assert t.enumerate_states()[0] == initial_state(t.world)
    assert t.full_gold_reachable() is True
    assert len(t.enumerate_states()) == 7                      # a caller cannot corrupt it
    assert t.enumerate_states() == t.enumerate_states()


# --- Bellman values (hand-computed) ------------------------------------------------------

def test_root_values_match_hand_enumeration():
    t = teacher()
    root = initial_state(t.world)
    # STOP: no gold yet -> 0.0. BM25 reveals d1+d2 (full gold, 1.5) -> -0.05 + 1.5 = 1.45.
    # Dense reaches the same 1.45 two levels down -> -0.05 + 1.45 = 1.40.
    assert t.q_values(root) == pytest.approx([0.0, 1.45, 1.40])
    assert t.oracle_return() == pytest.approx(1.45)
    assert t.full_gold_reachable() is True


def test_values_at_the_bm25_child_state():
    t = teacher()
    child = transition(t.world, initial_state(t.world), SEARCH_BM25)
    # Full gold already: STOP pays 1.5; either further search pays -0.05 + 1.5.
    assert t.q_values(child) == pytest.approx([1.5, 1.45, 1.45])


def test_search_cost_is_charged_once_per_edge_on_a_two_search_path():
    t = teacher(gold=("d4", "d5"))
    root = initial_state(t.world)
    # Dense reveals both gold docs at once: -0.05 + 1.5 = 1.45; BM25 needs two searches
    # to reach the same set: -0.05 + (-0.05 + 1.5) = 1.40.
    values = t.q_values(root)
    assert values[SEARCH_DENSE] == pytest.approx(1.45)
    assert values[SEARCH_BM25] == pytest.approx(1.40)
    assert t.oracle_return() == pytest.approx(1.45)


def test_illegal_actions_are_minus_infinity():
    t = teacher()
    leaf = [o for o in t.enumerate_states() if len(o["history"]) == 2][0]
    values = t.q_values(leaf)
    assert math.isfinite(values[STOP]) and values[STOP] == pytest.approx(1.5)
    assert values[SEARCH_BM25] == float("-inf") and values[SEARCH_DENSE] == float("-inf")
    # The leaf stop reward is exact: coverage 1 (0.5) + full success (1.0).
    assert values[STOP] == pytest.approx(0.5 * 1.0 + 1.0)


def test_stop_reward_hand_cases_through_the_teacher():
    t = teacher(gold=("d1", "d4"))
    root = initial_state(t.world)
    assert t.q_values(root)[STOP] == 0.0
    child = transition(t.world, root, SEARCH_BM25)
    assert t.q_values(child)[STOP] == pytest.approx(0.25)      # coverage 0.5 only


def test_colliding_observations_keep_their_history_and_their_remaining_budget():
    # (d0, d1, d2) is reachable in one search (BM25 reveals two at once) and in two
    # searches (Dense reveals one, BM25 reveals the last one). The ordered doc ids collide;
    # the histories and the remaining budgets do not.
    rankings = {"bm25": ["d0", "d1", "d2"], "dense": ["d0", "d1"]}
    world = make_world(rankings=rankings, gold=("d1", "d2"))
    rem1 = transition(world, initial_state(world), SEARCH_BM25)
    _, rem0 = _replay(world, ["SEARCH_DENSE", "SEARCH_BM25"])
    assert rem1["seen_doc_ids"] == rem0["seen_doc_ids"] == ("d0", "d1", "d2")
    assert rem1["optional_used"] == 1 and rem0["optional_used"] == 2
    assert rem1["history"] == ["SEARCH_BM25"]
    assert rem0["history"] == ["SEARCH_DENSE", "SEARCH_BM25"]

    t = teacher(rankings=rankings, gold=("d1", "d2"))
    colliding = [o for o in t.enumerate_states() if o["seen_doc_ids"] == ("d0", "d1", "d2")]
    assert len(colliding) == 2 and len({tuple(o["history"]) for o in colliding}) == 2
    # Full gold already retrieved and both rankings are exhausted: only STOP is legal.
    for obs in (rem1, rem0):
        assert t.q_values(obs) == [pytest.approx(1.5), float("-inf"), float("-inf")]
    # The memo keys the remaining budget separately (plan §3), even when the visible
    # document tuple collides: both (seen, remaining) entries exist and were computed.
    assert (("d0", "d1", "d2"), 1) in t._memo and (("d0", "d1", "d2"), 0) in t._memo


def test_unreachable_gold_is_distinguished_from_a_bad_policy():
    t = teacher(rankings={"bm25": ["d0", "d1", "d2", "d3"], "dense": ["d0", "d1", "d2", "d4"]},
                gold=("d9",))
    assert t.full_gold_reachable() is False
    # Every search costs 0.05 and cannot improve on coverage 0, so the oracle stops at once.
    assert t.oracle_return() == 0.0
    assert t.q_values(initial_state(t.world)) == pytest.approx([0.0, -SEARCH_COST, -SEARCH_COST])


@pytest.mark.parametrize("gold,expected", [
    (("d1", "d2"), True),      # one BM25 search
    (("d4", "d5"), True),      # one Dense search
    (("d1", "d4"), True),      # one search each side of the tree
    (("d3", "d4"), True),      # Dense then BM25
    (("d9",), False),          # in no ranking
])
def test_full_gold_reachable_hand_cases(gold, expected):
    assert teacher(gold=gold).full_gold_reachable() is expected


def test_teacher_matches_an_independent_brute_force_replay():
    world = make_world()
    t = VixStoryTeacherV2(world)
    for obs in t.enumerate_states():
        assert_q_equal(t.q_values(obs), _brute_force_q(world, obs))


def test_teacher_q_values_reject_foreign_or_malformed_observations():
    t = teacher()
    foreign = initial_state(make_world(qid="q_other"))
    with pytest.raises(ValueError, match="q_other"):
        t.q_values(foreign)
    with pytest.raises(ValueError, match="gold_context_ids"):
        t.q_values(dict(initial_state(t.world), gold_context_ids=["d1"]))
    without_mask = {k: v for k, v in initial_state(t.world).items() if k != "action_mask"}
    with pytest.raises(ValueError, match="keys"):
        t.q_values(without_mask)


# --- probabilities -----------------------------------------------------------------------

def test_probabilities_are_a_softmax_over_legal_actions_only():
    t = teacher()
    probs = t.probabilities(initial_state(t.world))
    assert len(probs) == N_ACTIONS and probs[STOP] > 0.0
    assert sum(probs) == pytest.approx(1.0)
    values = t.q_values(initial_state(t.world))
    # Hand ratios: exp(1.45 / 0.5) = exp(2.9) and exp(-0.05 / 0.5) = exp(-0.1).
    assert probs[SEARCH_BM25] / probs[STOP] == pytest.approx(math.exp(2.9))
    assert probs[SEARCH_DENSE] / probs[SEARCH_BM25] == pytest.approx(math.exp(-0.1))
    assert values[SEARCH_BM25] > values[SEARCH_DENSE] > values[STOP]


def test_probabilities_give_zero_mass_to_illegal_actions_and_are_json_safe():
    rankings = {"bm25": ["d0", "d1", "d2"], "dense": ["d0", "d1"]}
    world = make_world(rankings=rankings, gold=("d1", "d2"))
    _, rem0 = _replay(world, ["SEARCH_DENSE", "SEARCH_BM25"])
    assert rem0["remaining_optional_searches"] == 0
    t = teacher(rankings=rankings, gold=("d1", "d2"))
    probs = t.probabilities(rem0)
    assert probs == pytest.approx([1.0, 0.0, 0.0])             # only STOP is legal
    json.dumps(probs, allow_nan=False)                         # JSON-safe, unlike -inf Q values
    assert t.q_values(rem0)[SEARCH_BM25] == float("-inf")
    with pytest.raises(ValueError):
        json.dumps(t.q_values(rem0), allow_nan=False)          # caller must sanitize Q values


def test_ties_resolve_toward_stop_then_bm25_then_dense():
    t = teacher(gold=("d1", "d4"))
    root = initial_state(t.world)
    values = t.q_values(root)
    assert values[SEARCH_BM25] == values[SEARCH_DENSE]         # exactly tied: 1.40
    probs = t.probabilities(root)
    assert probs[SEARCH_BM25] == probs[SEARCH_DENSE]
    first_max = max(range(N_ACTIONS), key=lambda index: probs[index])
    assert ACTION_ORDER[first_max] == "SEARCH_BM25"            # earlier ACTION_ORDER wins
    # STOP is the first maximum when it is (weakly) best: STOP > BM25 == Dense here.
    stopped = teacher(rankings={"bm25": ["d0", "d1"], "dense": ["d0", "d1"]}, gold=("d9",))
    stop_probs = stopped.probabilities(initial_state(stopped.world))
    assert ACTION_ORDER[max(range(N_ACTIONS), key=lambda i: stop_probs[i])] == "STOP"


def test_softmax_bands_for_a_lower_temperature():
    root = initial_state(make_world())
    cold = teacher(temperature=0.1).probabilities(root)
    warm = teacher(temperature=0.5).probabilities(root)
    assert max(cold) > max(warm)
    assert sum(cold) == pytest.approx(1.0) and sum(warm) == pytest.approx(1.0)


@pytest.mark.parametrize("temperature", [0.0, -0.5, float("nan"), float("inf"), "0.5", True])
def test_teacher_rejects_invalid_temperatures(temperature):
    with pytest.raises(ValueError):
        teacher(temperature=temperature)


def test_default_temperature_is_the_locked_half():
    assert teacher().temperature == 0.5


# --- teacher -> Laya record integration --------------------------------------------------

def test_teacher_probabilities_feed_the_v2_laya_record():
    t = teacher()
    for obs in t.enumerate_states():
        record = laya_record("VĂN BẢN TRẠNG THÁI", t.probabilities(obs),
                             meta={"qid": obs["qid"], "history": obs["history"]})
        assert record["state"] == "VĂN BẢN TRẠNG THÁI"
        assert list(record["questions"]["next_action"]["criteria"]) == list(ACTION_ORDER)
        assert sum(record["gold"]["next_action"]["probabilities"].values()) == pytest.approx(1.0)
        assert record["meta"]["history"] == obs["history"]
        json.dumps(record, ensure_ascii=False)


# --- gold-only retrieval metrics (plan §4 worked examples) -------------------------------

def test_metrics_match_the_plan_worked_examples():
    partial = retrieval_metrics(["A", "C"], ["A", "B"], 1)
    assert partial["coverage"] == pytest.approx(0.5)
    assert partial["precision"] == pytest.approx(0.5)
    assert partial["full_success"] == 0.0
    assert partial["retrieval_return"] == pytest.approx(0.20)  # 0.25 - 0.05
    assert partial["matched_gold_ids"] == ["A"] and partial["missing_gold_ids"] == ["B"]
    assert partial["n_search"] == 1 and partial["n_seen"] == 2

    full = retrieval_metrics(["A", "B", "C"], ["A", "B"], 1)
    assert full["coverage"] == 1.0 and full["full_success"] == 1.0
    assert full["precision"] == pytest.approx(2 / 3)           # S = G is *not* required
    assert full["retrieval_return"] == pytest.approx(1.45)     # 0.5 + 1.0 - 0.05
    assert full["matched_gold_ids"] == ["A", "B"] and full["missing_gold_ids"] == []


def test_metrics_zero_coverage_and_one_gold_hit_are_not_full_success():
    zero = retrieval_metrics(["C"], ["A", "B"], 0)
    assert zero["coverage"] == 0.0 and zero["full_success"] == 0.0
    assert zero["precision"] == 0.0 and zero["retrieval_return"] == 0.0
    one_hit = retrieval_metrics(["A"], ["A", "B"], 2)
    assert one_hit["matched_gold_ids"] == ["A"] and one_hit["full_success"] == 0.0
    assert one_hit["retrieval_return"] == pytest.approx(0.25 - 0.10)


@pytest.mark.parametrize("seen,gold,n_search", [
    (["A", "A"], ["A"], 0),
    (["A", "A", "B", "B"], ["A", "B"], 1),
    (list("ABCDE"), ["A"], 2),
])
def test_metrics_apply_budget_to_unique_evidence(seen, gold, n_search):
    result = retrieval_metrics(seen, gold, n_search)
    assert result["n_seen"] == len(set(seen))
    assert result["coverage"] == result["full_success"] == 1.0


def test_metrics_deduplicate_seen_ids_in_first_occurrence_order():
    m = retrieval_metrics(["A", "A", "B"], ["B", "A"], 2)
    assert m["n_seen"] == 2 and m["precision"] == 1.0 and m["coverage"] == 1.0
    assert m["full_success"] == 1.0 and m["retrieval_return"] == pytest.approx(1.4)
    assert m["matched_gold_ids"] == ["B", "A"]                 # gold order is preserved
    assert m["missing_gold_ids"] == []


@pytest.mark.parametrize("seen,gold,n_search", [
    ([], ["A"], 0),                    # empty evidence
    (["A"], [], 0),                    # empty gold
    (["A"], ["A"], 3),                # exceeds the two-search budget
    (["A", 7], ["A"], 0),              # non-string seen id
    (["A"], ["A", "A"], 0),            # duplicate gold
    ("A", ["A"], 0),                   # a string is not a sequence of ids
    (["A"], "A", 0),                   # a string is not a sequence of ids
    (None, ["A"], 0),                  # not a sequence
    (["A"], ["A"], -1),                # negative search count
    (["A"], ["A"], 1.0),               # non-integer search count
    (["A"], ["A"], True),              # bool is not a count
])
def test_metrics_fail_closed_instead_of_scoring_zero(seen, gold, n_search):
    with pytest.raises(ValueError):
        retrieval_metrics(seen, gold, n_search)


def test_metrics_are_positional_and_match_a_real_episode_return():
    world = make_world()
    env = VixStoryEnvV2([world])
    obs, _ = env.reset(qid=world.qid)
    total = 0.0
    for action in (SEARCH_BM25, SEARCH_DENSE, STOP):
        obs, reward, terminated, _, _ = env.step(action)
        total += reward
        if terminated:
            break
    outcome = env.private_outcome()
    metrics = retrieval_metrics(outcome["seen_doc_ids"], outcome["gold_context_ids"],
                                outcome["n_search"])
    assert metrics["coverage"] == 1.0 and metrics["full_success"] == 1.0
    assert metrics["retrieval_return"] == pytest.approx(total)
    assert metrics["retrieval_return"] == pytest.approx(1.5 - SEARCH_COST * 2)


def test_oracle_return_is_an_upper_bound_for_environment_rollouts():
    world = make_world()
    env = VixStoryEnvV2([world])
    for actions in ([STOP], [SEARCH_BM25, STOP], [SEARCH_DENSE, STOP],
                    [SEARCH_BM25, SEARCH_DENSE, STOP]):
        env.reset(qid=world.qid)
        total = 0.0
        for action in actions:
            _, reward, _, _, _ = env.step(action)
            total += reward
        assert total <= teacher().oracle_return() + 1e-9
