"""VixSTORY paired evaluation tests (CPU/synthetic; no dataset, no GPU, no validation read).

Two layers:

* **Integration** — the real :class:`src.rl.vixstory_env.VixStoryEnv` over two synthetic
  two-passage worlds, driven by real E00-E07 + ABSTAIN/RANDOM_VALID policies through
  :func:`src.pipeline.vixstory_eval.run_vixstory_eval`. The tests pin the evaluator's
  promises against the *actual* ``private_outcome()`` contract (``f1``/``em``/``total``,
  ``n_optional_searches`` separated from the free bootstrap), fair search
  (same seed, same reader, same budget for every arm), no private-label access to policies,
  and honest FAILED/NOT_RUN statuses.
* **Unit** — the seeded paired bootstrap and the input refusals.

Nothing here reads the official validation split and no metric is fabricated.
"""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

from src.data.schemas import find_forbidden_keys
from src.pipeline.vixstory_eval import (
    ARM_IDS, MANDATORY_ARMS, SCHEMA, STATUS_FAILED, STATUS_NOT_RUN, STATUS_SUCCEEDED,
    paired_bootstrap_ci, run_vixstory_eval,
)
from src.rl.vixstory_env import (
    ABSTAIN, ANSWER, SEARCH_BM25, SEARCH_DENSE, VixStoryEnv, VixStoryPrivate, VixStoryReaderCache,
    VixStoryReaderPrediction, VixStoryWorld,
)
from src.utils.hashing import sha256_json, stable_seed

SEED = 42

# ------------------------------------------------------------------ synthetic worlds (2 qids)
CORPUS = {
    "d0": "VixSTORY đoạn nhiễu: không chứa câu trả lời.",
    "d1": "Trường A thành lập năm 1995 tại Hà Nội.",
    "d2": "Trường A có 12 khoa và 3 viện nghiên cứu.",
    "d3": "Đoạn văn bản bổ sung không liên quan tới câu hỏi.",
}
#: Bootstrap (BM25, 1) always lands on the noise doc d0; the two tools then diverge to d1/d2.
RANKINGS = {"bm25": ["d0", "d1", "d2", "d3"], "dense": ["d0", "d2", "d1", "d3"]}
#: Every ordered state reachable with a 2-search budget, so no policy path hits a cache miss.
_REACHABLE = (("d0",), ("d0", "d1"), ("d0", "d2"), ("d0", "d1", "d2"), ("d0", "d2", "d1"))


def _pred(qid: str, *, text: str | None, cited: tuple[str, ...] = (),
          valid: bool = False) -> VixStoryReaderPrediction:
    return VixStoryReaderPrediction(qid=qid, text=text, cited_ids=cited, citation_valid=valid)


def build_world(qid: str, *, answer: str, evidence_doc: str) -> VixStoryWorld:
    """A reader that stays silent until ``evidence_doc`` is among the visible passages.

    The reader's *output* is the same for every world that shares a qid here; only the private
    gold differs between worlds, which is what the no-leak test exploits.
    """
    cache = VixStoryReaderCache(reader_sha="test-reader", tokenizer_sha="test-tokenizer",
                                inference_config_sha="test-inference")
    for visible in _REACHABLE:
        if evidence_doc in visible:
            cache.put(qid, visible, _pred(qid, text=answer, cited=(evidence_doc,), valid=True))
        else:
            cache.put(qid, visible, _pred(qid, text=None))
    return VixStoryWorld(
        qid=qid, question=f"Câu hỏi cho {qid}?", split="train_strict", scenario_id=f"sc:{qid}",
        corpus=CORPUS, rankings=RANKINGS, scores={},
        private=VixStoryPrivate(qid=qid, answer_texts=(answer,), gold_context_ids=(evidence_doc,),
                                hop_level=1, question_type="factual"),
        reader=cache.reader(qid))


def _with_gold(world: VixStoryWorld, *, answer_texts: tuple[str, ...]) -> VixStoryWorld:
    """Same world (same reader/corpus/rankings) but a different private gold answer."""
    private = VixStoryPrivate(qid=world.qid, answer_texts=answer_texts,
                              gold_context_ids=world.private.gold_context_ids,
                              hop_level=world.private.hop_level,
                              question_type=world.private.question_type)
    return VixStoryWorld(qid=world.qid, question=world.question, split=world.split,
                         scenario_id=world.scenario_id, corpus=world.corpus,
                         rankings=world.rankings, private=private, reader=world.reader)


QIDS = ("vq:1", "vq:2")
#: q1's answer lives only in d1, q2's only in d2 — so no single tool wins both questions.
WORLDS = [build_world("vq:1", answer="1995", evidence_doc="d1"),
          build_world("vq:2", answer="12 khoa", evidence_doc="d2")]


class RecordingEnv(VixStoryEnv):
    """The real environment, additionally recording every ``reset(seed, qid)`` it receives."""

    def __init__(self, worlds, config=None):
        self.worlds_arg = worlds
        self.resets: list[tuple[int | None, str | None]] = []
        super().__init__(worlds, config)

    def reset(self, *, seed=None, options=None):
        self.resets.append((seed, (options or {}).get("qid")))
        return super().reset(seed=seed, options=options)


def make_factory(created, *, config=None):
    """A zero-argument factory; each call builds one fresh, fully deterministic env."""
    def make():
        env = RecordingEnv(WORLDS, config)
        created.append(env)
        return env
    return make


# -------------------------------------------------------------------------------- policies
class NoOptionalSearch:
    """E00: answer when the reader offers one, otherwise abstain; never spends the budget."""

    def begin_episode(self, *, qid, seed):
        pass

    def act(self, obs):
        return ANSWER if obs["action_mask"][ANSWER] else ABSTAIN


class AlwaysSearch:
    """Spend the whole optional budget on one tool, then answer if possible."""

    def __init__(self, tool: int) -> None:
        self.tool = tool

    def begin_episode(self, *, qid, seed):
        pass

    def act(self, obs):
        if obs["remaining_optional_searches"] > 0 and obs["action_mask"][self.tool]:
            return self.tool
        return ANSWER if obs["action_mask"][ANSWER] else ABSTAIN


class AnswerElseSearch:
    """Answer as soon as the reader offers one, else spend the budget (BM25 first)."""

    def begin_episode(self, *, qid, seed):
        pass

    def act(self, obs):
        if obs["action_mask"][ANSWER]:
            return ANSWER
        if obs["remaining_optional_searches"] > 0:
            for tool in (SEARCH_BM25, SEARCH_DENSE):
                if obs["action_mask"][tool]:
                    return tool
        return ABSTAIN


class AlternateSearch:
    """Spend the budget alternating BM25 then Dense, then answer if possible."""

    def begin_episode(self, *, qid, seed):
        pass

    def act(self, obs):
        if obs["remaining_optional_searches"] > 0:
            for tool in (SEARCH_BM25, SEARCH_DENSE):
                if obs["action_mask"][tool]:
                    return tool
        return ANSWER if obs["action_mask"][ANSWER] else ABSTAIN


class AlwaysAbstain:
    def begin_episode(self, *, qid, seed):
        pass

    def act(self, obs):
        return ABSTAIN


class RandomValid:
    """Uniform over the legal actions, seeded from ``(seed, qid)`` so it is reproducible."""

    def begin_episode(self, *, qid, seed):
        self.rng = random.Random((int(seed), qid))

    def act(self, obs):
        return self.rng.choice([i for i, ok in enumerate(obs["action_mask"]) if ok])


class BoomPolicy:
    def begin_episode(self, *, qid, seed):
        pass

    def act(self, obs):
        raise RuntimeError("boom")


class SpyPolicy:
    """Records every observation it is handed and delegates the decision."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.seen: list[dict] = []

    def begin_episode(self, *, qid, seed):
        self.inner.begin_episode(qid=qid, seed=seed)

    def act(self, obs):
        self.seen.append(obs)
        return self.inner.act(obs)


def default_policies() -> dict:
    return {
        "E00": NoOptionalSearch(),
        "E01": AlwaysSearch(SEARCH_BM25),
        "E02": AlwaysSearch(SEARCH_DENSE),
        "E03": AnswerElseSearch(),
        "E04": AlternateSearch(),
        "E05": AlwaysSearch(SEARCH_BM25),
        "E06": AlternateSearch(),
        "E07": AnswerElseSearch(),
        "ABSTAIN": AlwaysAbstain(),
        "RANDOM_VALID": RandomValid(),
    }


def _canonical(obs: dict) -> dict:
    """A numpy-free projection of an observation so equality never hits array ambiguity."""
    return {key: tuple(value) if isinstance(value, np.ndarray) else value
            for key, value in obs.items()}


def _run(**overrides) -> dict:
    kwargs = {"env_factory": make_factory([]), "policies": default_policies(), "qids": QIDS,
              "n_resamples": 200}
    kwargs.update(overrides)
    return run_vixstory_eval(**kwargs)


# ------------------------------------------------------ arms, schema, fair search execution

def test_every_arm_covers_exactly_the_injected_qids_and_success_holds():
    created: list = []
    report = _run(env_factory=make_factory(created))
    assert report["schema"] == SCHEMA
    assert report["success"] is True
    assert report["missing_mandatory"] == []
    assert report["n_qids"] == len(QIDS)
    assert report["qids_sha256"] == sha256_json(list(QIDS))
    for arm in ARM_IDS:
        block = report["arms"][arm]
        assert block["status"] == STATUS_SUCCEEDED, (arm, block["reason"])
        assert block["n"] == len(QIDS)
        assert list(block["per_qid"]) == list(QIDS)  # same qids, same order, every arm
    assert len(created) == len(ARM_IDS)  # one fresh env per arm


def test_all_arms_share_the_same_seed_reader_and_budget():
    created: list = []
    _run(env_factory=make_factory(created))
    # Same immutable worlds => same corpus and same frozen reader for every arm.
    assert all(env.worlds_arg is WORLDS for env in created)
    # Same per-episode seed for every arm (stable_seed("vixstory", SEED, qid)).
    expected_resets = [(stable_seed("vixstory", SEED, qid), qid) for qid in QIDS]
    assert all(env.resets == expected_resets for env in created)
    # Same retrieval budget and bootstrap configuration for every arm.
    assert all(env.config == created[0].config for env in created)
    config = created[0].config
    assert (config.bootstrap_tool, config.bootstrap_passages) == ("bm25", 1)
    assert (config.passage_per_search_action, config.max_optional_search) == (1, 2)
    assert config.search_cost == pytest.approx(0.05)


def test_policies_never_see_private_gold_and_all_arms_start_from_the_same_root():
    spies = {arm: SpyPolicy(policy) for arm, policy in default_policies().items()}
    report = _run(env_factory=make_factory([]), policies=spies)
    assert report["success"] is True
    for arm, spy in spies.items():
        assert spy.seen, f"arm {arm} produced no observations"
        assert all(find_forbidden_keys(obs) == [] for obs in spy.seen)
    # Identical visible root state across arms (same bootstrap, reader and budget).
    roots = {arm: _canonical(spy.seen[0]) for arm, spy in spies.items()}
    first = roots["E00"]
    assert first["qid"] == QIDS[0] and first["seen_doc_ids"] == ("d0",)
    assert all(obs == first for obs in roots.values())


def test_only_the_private_gold_moves_the_score_not_the_policy_state():
    alt_worlds = [_with_gold(w, answer_texts=("hoàn toàn khác",)) for w in WORLDS]
    spy_real, spy_alt = SpyPolicy(AlwaysSearch(SEARCH_BM25)), SpyPolicy(AlwaysSearch(SEARCH_BM25))
    report_real = run_vixstory_eval(env_factory=lambda: RecordingEnv(WORLDS),
                                    policies={"E01": spy_real}, qids=QIDS, n_resamples=50)
    report_alt = run_vixstory_eval(env_factory=lambda: RecordingEnv(alt_worlds),
                                   policies={"E01": spy_alt}, qids=QIDS, n_resamples=50)
    # The policy sees byte-identical states although the private gold answer changed.
    assert [_canonical(o) for o in spy_real.seen] == [_canonical(o) for o in spy_alt.seen]
    assert all(find_forbidden_keys(obs) == [] for obs in spy_real.seen)
    real, alt = report_real["arms"]["E01"]["per_qid"], report_alt["arms"]["E01"]["per_qid"]
    assert any(real[qid]["reward"] != alt[qid]["reward"] for qid in QIDS)
    assert any(real[qid]["f1"] != alt[qid]["f1"] for qid in QIDS)


# ------------------------------------------------- real private_outcome fields and search accounting

def test_per_qid_records_use_the_real_outcome_fields():
    report = _run()
    expected_keys = {"reward", "f1", "em", "answered", "success", "optional_searches",
                     "bootstrap_searches", "citation_valid"}
    for arm in ARM_IDS:
        block = report["arms"][arm]
        assert block["metrics"]["n"] == len(QIDS)
        for record in block["per_qid"].values():
            assert set(record) == expected_keys
            assert isinstance(record["reward"], float)
            assert isinstance(record["answered"], bool)
            assert isinstance(record["success"], bool)
            assert isinstance(record["optional_searches"], int)
            assert isinstance(record["bootstrap_searches"], int)
    metrics = report["arms"]["E01"]["metrics"]
    assert 0.0 <= metrics["citation_valid_rate"] <= 1.0
    assert 0.0 <= metrics["answer_rate"] <= 1.0
    assert metrics["success_rate"] == pytest.approx(
        np.mean([row["success"] for row in report["arms"]["E01"]["per_qid"].values()]))


def test_optional_searches_are_separated_from_the_free_bootstrap():
    report = _run()
    for arm in ARM_IDS:
        metrics = report["arms"][arm]["metrics"]
        # Exactly one free BM25 bootstrap retrieval per episode, never an optional SEARCH.
        assert metrics["total_bootstrap_searches"] == len(QIDS)
        assert metrics["mean_bootstrap_searches"] == pytest.approx(1.0)
        for record in report["arms"][arm]["per_qid"].values():
            assert record["bootstrap_searches"] == 1
            assert record["optional_searches"] >= 0
    # E00 spends nothing; E01 spends the whole 2-search budget on both questions.
    assert report["arms"]["E00"]["metrics"]["total_optional_searches"] == 0
    assert report["arms"]["E01"]["metrics"]["total_optional_searches"] == 2 * len(QIDS)


def test_reward_is_the_episode_return_and_adaptive_beats_no_adaptation():
    report = _run()
    arms = report["arms"]
    # E00 never searches, so it can never see an answer and scores the 0.0 ABSTAIN floor.
    assert arms["E00"]["metrics"]["mean_reward"] == pytest.approx(0.0)
    assert arms["E00"]["metrics"]["answer_rate"] == pytest.approx(0.0)
    assert arms["E00"]["metrics"]["total_optional_searches"] == 0
    # E01 answers both (utility 1.0) after two optional searches (2 * 0.05 cost) => 0.90.
    assert arms["E01"]["metrics"]["mean_reward"] == pytest.approx(0.90)
    assert arms["E01"]["metrics"]["answer_rate"] == pytest.approx(1.0)
    assert arms["E01"]["metrics"]["mean_f1"] == pytest.approx(1.0)
    assert arms["E01"]["metrics"]["mean_reward"] > arms["E00"]["metrics"]["mean_reward"]
    # The always-abstain control is the arithmetic floor, so E00 ties it.
    assert arms["ABSTAIN"]["metrics"]["mean_reward"] == pytest.approx(0.0)


# ------------------------------------------------------------------------ control comparisons

def test_control_comparisons_are_paired_seeded_and_exclude_self():
    report = _run(ci_seed=7)
    assert "E01_minus_ABSTAIN" in report["comparisons"]
    assert "E01_minus_RANDOM_VALID" in report["comparisons"]
    assert "E07_minus_E00" in report["comparisons"]
    assert not [key for key in report["comparisons"] if key.split("_minus_")[0]
                == key.split("_minus_")[1]]
    block = report["comparisons"]["E01_minus_ABSTAIN"]
    assert set(block) == {"reward", "f1", "em"}
    assert block["reward"]["mean"] > 0
    assert block["reward"]["ci_low"] <= block["reward"]["mean"] <= block["reward"]["ci_high"]
    # Same seed -> byte-identical intervals; a fresh, independent run proves it is seeded.
    assert _run(ci_seed=7)["comparisons"] == report["comparisons"]


def test_paired_bootstrap_ci_is_zero_for_identical_sides_and_seed_stable():
    zero = paired_bootstrap_ci({qid: 0.0 for qid in QIDS}, n_resamples=100, seed=1)
    assert zero["mean"] == 0.0 and zero["ci_low"] == 0.0 and zero["ci_high"] == 0.0
    diffs = {qid: float(i) for i, qid in enumerate(QIDS)}
    first = paired_bootstrap_ci(diffs, n_resamples=100, seed=3)
    assert first == paired_bootstrap_ci(diffs, n_resamples=100, seed=3)
    assert first["ci_low"] <= first["mean"] <= first["ci_high"]
    for bad in (dict(n_resamples=0), dict(alpha=0.0), dict(alpha=1.0)):
        with pytest.raises(ValueError):
            paired_bootstrap_ci(diffs, **bad)
    with pytest.raises(ValueError, match="at least one"):
        paired_bootstrap_ci({}, n_resamples=10)


# ------------------------------------------------------------- honest statuses and refusals

def test_missing_arm_is_not_run_and_blocks_mandatory_success():
    policies = default_policies()
    policies.pop("E05")
    report = _run(policies=policies)
    block = report["arms"]["E05"]
    assert block["status"] == STATUS_NOT_RUN
    assert block["per_qid"] == {}
    assert block["metrics"]["mean_reward"] is None and block["metrics"]["n"] == 0
    assert report["success"] is False
    assert report["missing_mandatory"] == ["E05"]
    assert not [key for key in report["comparisons"]
                if key.startswith("E05_minus_") or key.endswith("_minus_E05")]
    assert any("E05" in note for note in report["limitations"])
    assert MANDATORY_ARMS == ("E01", "E02", "E03", "E04", "E05", "E06", "E07")


def test_raising_arm_is_failed_without_losing_the_others():
    policies = default_policies()
    policies["E04"] = BoomPolicy()
    report = _run(policies=policies)
    assert report["arms"]["E04"]["status"] == STATUS_FAILED
    assert "RuntimeError: boom" in report["arms"]["E04"]["reason"]
    assert report["arms"]["E04"]["metrics"]["mean_f1"] is None
    assert report["arms"]["E00"]["status"] == STATUS_SUCCEEDED
    assert report["success"] is False and report["missing_mandatory"] == ["E04"]


def test_missing_real_outcome_key_fails_the_arm_and_names_the_key():
    class MissingF1Env(RecordingEnv):
        def private_outcome(self):
            outcome = super().private_outcome()
            if outcome is not None:
                outcome.pop("f1")
            return outcome

    report = _run(env_factory=lambda: MissingF1Env(WORLDS))
    arm = report["arms"]["E01"]
    assert arm["status"] == STATUS_FAILED
    assert "missing" in arm["reason"] and "f1" in arm["reason"]
    assert arm["per_qid"] == {}


def test_legacy_viquad_correctness_keys_are_not_the_contract():
    """An env exposing the ViQuAD ``correctness_f1``/``correctness_em`` names must be rejected."""
    class LegacyEnv(RecordingEnv):
        def private_outcome(self):
            outcome = super().private_outcome()
            if outcome is not None:
                outcome["correctness_f1"] = outcome.pop("f1")
                outcome["correctness_em"] = outcome.pop("em")
            return outcome

    report = _run(env_factory=lambda: LegacyEnv(WORLDS))
    arm = report["arms"]["E01"]
    assert arm["status"] == STATUS_FAILED
    assert "missing" in arm["reason"] and "f1" in arm["reason"]


def test_absent_citation_validity_is_na_with_a_recorded_limitation():
    class NoCitationEnv(RecordingEnv):
        def private_outcome(self):
            outcome = super().private_outcome()
            if outcome is not None:
                outcome.pop("citation_valid")
            return outcome

    report = _run(env_factory=lambda: NoCitationEnv(WORLDS))
    metrics = report["arms"]["E01"]["metrics"]
    assert report["arms"]["E01"]["status"] == STATUS_SUCCEEDED
    assert metrics["citation_valid_rate"] is None and metrics["citation_valid_n"] == 0
    assert any("citation_valid" in note for note in report["limitations"])


def test_inconsistent_tool_counts_fail_the_arm():
    """Optional searches may never exceed the total tool calls (bootstrap cannot be negative)."""
    class BadCountsEnv(RecordingEnv):
        def private_outcome(self):
            outcome = super().private_outcome()
            if outcome is not None:
                outcome["n_optional_searches"] = outcome["n_bm25_searches"] + 5
            return outcome

    report = _run(env_factory=lambda: BadCountsEnv(WORLDS))
    arm = report["arms"]["E01"]
    assert arm["status"] == STATUS_FAILED
    assert "bootstrap" in arm["reason"]


def test_duplicate_empty_and_unknown_arms_are_refused():
    with pytest.raises(ValueError, match="duplicate"):
        _run(qids=("vq:1", "vq:1"))
    with pytest.raises(ValueError, match="non-empty"):
        _run(qids=())
    with pytest.raises(ValueError, match="unknown arm"):
        _run(policies={"B9": NoOptionalSearch()})


def test_report_is_json_serializable():
    report = _run()
    assert json.loads(json.dumps(report))["schema"] == SCHEMA
