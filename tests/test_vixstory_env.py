"""Contract tests for the isolated VixSTORY environment (``src/rl/vixstory_env.py``).

Coverage matches the pilot requirements (`.claude/plans/vixstory-pilot-300-e01-e07.md` §4):
multi-gold single-path evidence, search-cost accounting, the ANSWER/SEARCH mask, no label
leak, cache identity and hard miss, and strict rejection of illegal actions.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.data.schemas import find_forbidden_keys
from src.rl.vixstory_env import (
    ABSTAIN,
    ANSWER,
    N_ACTIONS,
    NULL_PREDICTION,
    SEARCH_BM25,
    SEARCH_DENSE,
    VixStoryEnv,
    VixStoryEnvConfig,
    VixStoryPrivate,
    VixStoryReaderCache,
    VixStoryReaderPrediction,
    VixStoryWorld,
)

# --- fixtures -------------------------------------------------------------------------
DOC_0 = "VixSTORY: trường A toạ lạc tại Hà Nội."
DOC_1 = "Trường A thành lập năm 1995 tại Hà Nội."
DOC_2 = "Trường A hiện có 12 khoa và 3 viện nghiên cứu."
DOC_3 = "Đoạn nhiễu không liên quan tới câu hỏi."
CORPUS = {"d0": DOC_0, "d1": DOC_1, "d2": DOC_2, "d3": DOC_3}
#: Bootstrap (BM25, 1) -> d0; the two tools then diverge to d1 / d2.
RANKINGS = {"bm25": ["d0", "d1", "d2", "d3"], "dense": ["d0", "d2", "d1", "d3"]}
#: Full free-form multi-gold answer, made of facts from d1 and d2.
GOLD_MULTI = "Trường A thành lập năm 1995 và có 12 khoa"


def pred(qid: str, *, text: str | None, cited: tuple[str, ...] = (),
         valid: bool = False) -> VixStoryReaderPrediction:
    return VixStoryReaderPrediction(qid=qid, text=text, cited_ids=tuple(cited), citation_valid=valid)


def make_world(qid: str, *, gold_ids=("d1",), answer: str = "1995", predictions=None,
               rankings=RANKINGS, scores=None, corpus=CORPUS,
               question: str = "Trường A có đặc điểm gì?") -> VixStoryWorld:
    cache = VixStoryReaderCache(reader_sha="reader-1", tokenizer_sha="tok-1",
                                inference_config_sha="infer-1")
    for visible, prediction in (predictions or {}).items():
        cache.put(qid, visible, prediction)
    return VixStoryWorld(
        qid=qid, question=question, split="train_strict", scenario_id=f"sc:{qid}",
        corpus=corpus, rankings=rankings, scores=scores or {},
        private=VixStoryPrivate(qid=qid, answer_texts=(answer,), gold_context_ids=tuple(gold_ids),
                                hop_level=len(gold_ids), question_type="factual"),
        reader=cache.reader(qid))


def simple_world(qid: str, *, gold_ids=("d1",), answer: str = "1995") -> VixStoryWorld:
    """Every reachable ordered state is cached, so any policy path stays in the cache."""
    predictions = {
        ("d0",): pred(qid, text=None),
        ("d0", "d1"): pred(qid, text=answer, cited=("d1",), valid=True),
        ("d0", "d2"): pred(qid, text=answer, cited=("d2",), valid=True),
        ("d0", "d1", "d2"): pred(qid, text=answer, cited=("d1",), valid=True),
        ("d0", "d2", "d1"): pred(qid, text=answer, cited=("d1",), valid=True),
    }
    return make_world(qid, gold_ids=gold_ids, answer=answer, predictions=predictions)


def env_for(world: VixStoryWorld, config: VixStoryEnvConfig | None = None) -> VixStoryEnv:
    return VixStoryEnv([world], config)


# --- bootstrap / costs -----------------------------------------------------------------
def test_bootstrap_is_free_and_defines_the_root_state():
    env = env_for(make_world("q1", predictions={("d0",): pred("q1", text=None)}))
    obs, info = env.reset(options={"qid": "q1"})
    assert obs["seen_doc_ids"] == ("d0",)
    assert obs["optional_used"] == 0
    assert obs["remaining_optional_searches"] == 2
    assert env.n_bm25 == 1 and env.n_dense == 0
    assert not obs["action_mask"][ANSWER]  # reader declined on the noise passage
    _, reward, term, _, _ = env.step(ABSTAIN)
    assert term and reward == 0.0
    outcome = env.private_outcome()
    assert outcome["search_cost_total"] == 0.0
    assert outcome["total"] == 0.0  # the bootstrap is not charged


def test_each_optional_search_costs_five_hundredths_exactly_once():
    env = env_for(simple_world("q3"))
    env.reset(options={"qid": "q3"})
    _, r1, term, trunc, _ = env.step(SEARCH_BM25)
    assert r1 == pytest.approx(-0.05) and not term and not trunc
    _, r2, term, trunc, _ = env.step(SEARCH_DENSE)
    assert r2 == pytest.approx(-0.05) and not term and not trunc
    _, r3, term, _, _ = env.step(ABSTAIN)
    assert r3 == 0.0 and term
    outcome = env.private_outcome()
    assert outcome["n_optional_searches"] == 2
    assert outcome["search_cost_total"] == pytest.approx(0.10)
    assert outcome["total"] == pytest.approx(-0.10)


def test_abstain_always_scores_zero():
    env = env_for(make_world("q_abs", predictions={("d0",): pred("q_abs", text="khác")}))
    env.reset(options={"qid": "q_abs"})
    _, reward, term, _, _ = env.step(ABSTAIN)
    assert term and reward == 0.0
    assert env.private_outcome()["correct"] is False


# --- ANSWER reward and citation gate ---------------------------------------------------
def test_answer_scores_one_when_f1_passes_and_citation_is_valid():
    env = env_for(make_world("q4", predictions={
        ("d0",): pred("q4", text=None),
        ("d0", "d1"): pred("q4", text="1995", cited=("d1",), valid=True)}))
    env.reset(options={"qid": "q4"})
    env.step(SEARCH_BM25)
    _, reward, term, _, info = env.step(ANSWER)
    assert term and reward == 1.0
    assert info["terminal_action"] == "ANSWER"
    outcome = env.private_outcome()
    assert outcome["correct"] is True
    assert outcome["f1"] == pytest.approx(1.0)
    assert outcome["total"] == pytest.approx(1.0 - 0.05)


def test_answer_below_f1_threshold_scores_zero_even_with_valid_citation():
    env = env_for(make_world("q_f1", answer=GOLD_MULTI, predictions={
        ("d0",): pred("q_f1", text=None),
        ("d0", "d1"): pred("q_f1", text="1995", cited=("d1",), valid=True)}))
    env.reset(options={"qid": "q_f1"})
    env.step(SEARCH_BM25)
    _, reward, term, _, _ = env.step(ANSWER)
    assert term and reward == 0.0
    assert 0.0 < env.private_outcome()["f1"] < 0.5


def test_invalid_citation_scores_zero_but_the_answer_is_still_submittable():
    env = env_for(make_world("q5", predictions={
        ("d0",): pred("q5", text=None),
        ("d0", "d1"): pred("q5", text="1995", cited=("dX",), valid=False)}))
    obs, _ = env.reset(options={"qid": "q5"})
    obs, _, _, _, _ = env.step(SEARCH_BM25)
    assert obs["action_mask"][ANSWER], "text present => ANSWER legal regardless of citation"
    assert obs["candidate"]["citation_valid"] is False
    _, reward, term, _, _ = env.step(ANSWER)
    assert term and reward == 0.0
    assert env.private_outcome()["correct"] is False


def test_citation_gate_off_pays_on_f1_alone():
    world = make_world("q6", predictions={
        ("d0",): pred("q6", text=None),
        ("d0", "d1"): pred("q6", text="1995", cited=("dX",), valid=False)})
    env = env_for(world, VixStoryEnvConfig(citation_gate=False))
    env.reset(options={"qid": "q6"})
    env.step(SEARCH_BM25)
    _, reward, term, _, _ = env.step(ANSWER)
    assert term and reward == 1.0


def test_answer_is_masked_when_the_reader_text_is_blank():
    for blank in (None, "", "   "):
        env = env_for(make_world("q2", predictions={
            ("d0",): pred("q2", text=blank),
            ("d0", "d1"): pred("q2", text=blank)}))
        obs, _ = env.reset(options={"qid": "q2"})
        assert not obs["action_mask"][ANSWER]
        obs, _, _, _, _ = env.step(SEARCH_BM25)
        assert not obs["action_mask"][ANSWER]


# --- multi-gold, single path -----------------------------------------------------------
MULTI_PREDS = {
    ("d0",): pred("qm", text=None),
    ("d0", "d1"): pred("qm", text="1995", cited=("d1",), valid=True),
    ("d0", "d2"): pred("qm", text="12 khoa", cited=("d2",), valid=True),
    ("d0", "d1", "d2"): pred("qm", text=GOLD_MULTI, cited=("d1", "d2"), valid=True),
    ("d0", "d2", "d1"): pred("qm", text=GOLD_MULTI, cited=("d1", "d2"), valid=True),
}


def test_multi_gold_evidence_requires_a_single_ordered_path():
    gold_ids = ("d1", "d2")
    # Path A: BM25 then Dense. Neither branch alone is complete; the ordered tuple is.
    env = env_for(make_world("qm", gold_ids=gold_ids, answer=GOLD_MULTI, predictions=MULTI_PREDS))
    obs, _ = env.reset(options={"qid": "qm"})
    assert env.private_outcome() is None
    obs, _, _, _, _ = env.step(SEARCH_BM25)
    assert obs["seen_doc_ids"] == ("d0", "d1")
    _, reward_partial, term, _, _ = env.step(ANSWER)
    assert term and reward_partial == 0.0
    assert env.private_outcome()["gold_evidence_recall"] == pytest.approx(0.5)
    assert env.private_outcome()["correct"] is False

    env = env_for(make_world("qm", gold_ids=gold_ids, answer=GOLD_MULTI, predictions=MULTI_PREDS))
    env.reset(options={"qid": "qm"})
    env.step(SEARCH_BM25)
    obs, _, _, _, _ = env.step(SEARCH_DENSE)
    assert obs["seen_doc_ids"] == ("d0", "d1", "d2")
    _, reward_full, term, _, _ = env.step(ANSWER)
    assert term and reward_full == 1.0
    outcome = env.private_outcome()
    assert outcome["gold_evidence_recall"] == pytest.approx(1.0)
    assert outcome["correct"] is True

    # Path B: the reverse tool order also holds both gold passages in one state.
    env = env_for(make_world("qm", gold_ids=gold_ids, answer=GOLD_MULTI, predictions=MULTI_PREDS))
    env.reset(options={"qid": "qm"})
    env.step(SEARCH_DENSE)
    obs, _, _, _, _ = env.step(SEARCH_BM25)
    assert obs["seen_doc_ids"] == ("d0", "d2", "d1")
    _, reward_reverse, term, _, _ = env.step(ANSWER)
    assert term and reward_reverse == 1.0
    assert env.private_outcome()["gold_evidence_recall"] == pytest.approx(1.0)


# --- reader cache ----------------------------------------------------------------------
def test_reader_cache_key_distinguishes_question_and_document_order():
    cache = VixStoryReaderCache(reader_sha="r", tokenizer_sha="t", inference_config_sha="i")
    assert cache.key("q", ["d0", "d1"]) != cache.key("q", ["d1", "d0"])
    assert cache.key("q", ["d0", "d1"]) != cache.key("q2", ["d0", "d1"])
    cache.put("q", ["d0", "d1"], pred("q", text="x"))
    assert cache.lookup("q", ["d0", "d1"]).text == "x"
    with pytest.raises(KeyError):
        cache.lookup("q", ["d1", "d0"])


def test_a_reader_cache_miss_fails_instead_of_inferring():
    env = env_for(make_world("q10", predictions={
        ("d0",): pred("q10", text=None),
        ("d0", "d1"): pred("q10", text="1995", cited=("d1",), valid=True)}))
    env.reset(options={"qid": "q10"})
    env.step(SEARCH_BM25)
    with pytest.raises(KeyError, match="cache miss"):
        env.step(SEARCH_DENSE)  # state (d0, d1, d2) was never cached


# --- leakage ---------------------------------------------------------------------------
def test_observation_and_info_carry_no_gold_answer_ids_hop_or_oracle():
    secret = "9999"
    world = make_world("q11", gold_ids=("d1", "d2"), answer=secret, predictions={
        ("d0",): pred("q11", text="một câu trả lời khác"),
        ("d0", "d1"): pred("q11", text="khác", cited=("d1",), valid=True)})
    env = env_for(world)
    obs, info = env.reset(options={"qid": "q11"})
    obs, _, _, _, info = env.step(SEARCH_BM25)
    assert find_forbidden_keys(obs) == []
    assert find_forbidden_keys(info) == []
    for obj in (obs, info):
        blob = json.dumps(obj, default=str, ensure_ascii=False)
        assert secret not in blob
        assert "hop" not in blob
        assert "question_type" not in blob
        assert "oracle" not in blob
        assert "gold" not in blob


def test_candidate_exposes_no_fabricated_span_or_null_scores():
    env = env_for(simple_world("q22"))
    obs, _ = env.reset(options={"qid": "q22"})
    assert set(obs["candidate"]) == {"text", "cited_ids", "citation_valid", "answer_present"}
    for fabricated in ("non_null_score", "null_score", "null_margin", "doc_id", "start_char"):
        assert fabricated not in obs["candidate"]


def test_terminal_metrics_live_only_in_the_private_outcome():
    env = env_for(simple_world("q12"))
    env.reset(options={"qid": "q12"})
    env.step(SEARCH_BM25)
    obs, _, term, _, info = env.step(ANSWER)
    assert term
    outcome = env.private_outcome()
    for key in ("f1", "em", "correct", "gold_evidence_recall", "search_cost_total"):
        assert key in outcome
    assert "f1" not in info and "gold_evidence_recall" not in info
    assert "f1" not in obs
    assert "reward" not in info


# --- masks and strictness --------------------------------------------------------------
def test_search_is_masked_when_the_optional_budget_is_spent():
    env = VixStoryEnv([simple_world("q14")], VixStoryEnvConfig(max_optional_search=1))
    env.reset(options={"qid": "q14"})
    obs, _, _, _, _ = env.step(SEARCH_BM25)
    assert obs["remaining_optional_searches"] == 0
    assert not obs["action_mask"][SEARCH_BM25]
    assert not obs["action_mask"][SEARCH_DENSE]
    assert obs["action_mask"][ABSTAIN]


def test_search_is_masked_when_the_tool_ranking_is_exhausted():
    rankings = {"bm25": ["d0", "d1", "d2"], "dense": ["d0"]}
    env = env_for(make_world("q15", rankings=rankings,
                             predictions={("d0",): pred("q15", text=None)}))
    obs, _ = env.reset(options={"qid": "q15"})
    assert obs["action_mask"][SEARCH_BM25]
    assert not obs["action_mask"][SEARCH_DENSE]


def test_invalid_action_raises_in_strict_mode():
    env = env_for(make_world("q13", predictions={("d0",): pred("q13", text=None)}))
    env.reset(options={"qid": "q13"})
    assert not env._action_mask()[ANSWER]
    with pytest.raises(ValueError, match="invalid action"):
        env.step(ANSWER)


def test_watchdog_truncates_instead_of_faking_a_terminal():
    env = VixStoryEnv([simple_world("q18")], VixStoryEnvConfig(max_optional_search=6, max_steps=2))
    env.reset(options={"qid": "q18"})
    _, _, term, trunc, _ = env.step(SEARCH_BM25)
    assert not term and not trunc
    _, _, term2, trunc2, info2 = env.step(SEARCH_DENSE)
    assert trunc2 and not term2
    assert info2["truncated_reason"] == "watchdog_max_steps"


# --- construction validation -----------------------------------------------------------
def test_world_rejects_rankings_outside_the_shared_corpus():
    with pytest.raises(ValueError, match="outside the shared corpus"):
        VixStoryWorld(qid="q", question="?", split="s", scenario_id="sc",
                      corpus={"d0": "x"}, rankings={"bm25": ["d0", "zz"], "dense": []},
                      private=VixStoryPrivate(qid="q", answer_texts=("a",),
                                              gold_context_ids=("d0",)),
                      reader=lambda _seen: NULL_PREDICTION)


def test_world_requires_a_non_empty_gold_answer():
    with pytest.raises(ValueError, match="answerable"):
        VixStoryWorld(qid="q", question="?", split="s", scenario_id="sc", corpus={"d0": "x"},
                      rankings={"bm25": ["d0"], "dense": ["d0"]},
                      private=VixStoryPrivate(qid="q", answer_texts=(), gold_context_ids=("d0",)),
                      reader=lambda _seen: NULL_PREDICTION)


def test_duplicate_qid_is_rejected():
    with pytest.raises(ValueError, match="duplicate qid"):
        VixStoryEnv([simple_world("qdup"), simple_world("qdup")])


def test_config_rejects_bad_values():
    with pytest.raises(ValueError):
        VixStoryEnvConfig(f1_threshold=1.5)
    with pytest.raises(ValueError):
        VixStoryEnvConfig(bootstrap_tool="tfidf")
    with pytest.raises(ValueError):
        VixStoryEnvConfig(search_cost=-0.1)
    with pytest.raises(ValueError):
        VixStoryEnvConfig(max_optional_search=-1)


def test_reset_seed_reproduces_the_same_episode():
    env_a = VixStoryEnv([simple_world("qa"), simple_world("qb")])
    env_b = VixStoryEnv([simple_world("qa"), simple_world("qb")])
    obs_a, _ = env_a.reset(seed=7)
    obs_b, _ = env_b.reset(seed=7)
    assert obs_a["qid"] == obs_b["qid"]
    assert obs_a["seen_doc_ids"] == obs_b["seen_doc_ids"]
    assert env_a.state_hash() == env_b.state_hash()


# --- downstream compatibility ----------------------------------------------------------
def test_observation_supports_the_baseline_policy_interface():
    from src.eval.policies import choose_tool, legal_actions

    env = env_for(simple_world("q19"))
    obs, _ = env.reset(options={"qid": "q19"})
    mask = np.asarray(obs["action_mask"], dtype=bool)
    # Tool choice reads only the action mask and search history, not reader scores.
    assert ABSTAIN in legal_actions(mask)
    assert choose_tool(obs, mode="alternate") == SEARCH_BM25  # first search ties to BM25
    obs, _, _, _, _ = env.step(SEARCH_BM25)
    assert choose_tool(obs, mode="alternate") == SEARCH_DENSE  # BM25 already used once
    assert ABSTAIN in legal_actions(obs["action_mask"])
    with pytest.raises(ValueError, match="ABSTAIN must always be legal"):
        legal_actions(np.zeros(N_ACTIONS, dtype=bool))


def test_reinforce_rollout_and_loss_drive_the_vixstory_env():
    """The stock rollout/reinforce_loss machinery runs on the honest feature adapter."""
    pytest.importorskip("torch")
    import torch

    from src.rl.policy import PolicyNet
    from src.rl.reinforce import RLConfig, reinforce_loss, rollout
    from src.rl.vixstory_env import VixStoryFeatureExtractor

    env = env_for(simple_world("q17"))
    extractor = VixStoryFeatureExtractor()
    net = PolicyNet(extractor.dim, d_hidden=8)
    episode = rollout(env, net, extractor, None, None, seed=0, qid="q17")
    assert episode.qid == "q17"
    assert 1 <= len(episode.transitions) <= 4
    for transition in episode.transitions:
        assert transition.mask.shape == (N_ACTIONS,)
        assert transition.features.shape == (extractor.dim,)
        assert np.isfinite(transition.features).all()
    assert env.private_outcome() is not None

    loss, stats = reinforce_loss([episode], net, None, RLConfig(), device=torch.device("cpu"))
    assert torch.isfinite(loss)
    assert stats["n_steps"] == len(episode.transitions)
    loss.backward()  # the env's own rewards reach the policy through the real loss
    assert any(param.grad is not None for param in net.parameters())
