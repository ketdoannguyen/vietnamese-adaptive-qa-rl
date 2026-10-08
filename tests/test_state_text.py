"""State text is public-only, deterministic, and hashes stably (`02_IMPL/04` §2).

Real observations, not hand-built dicts: the text is rendered from what
:class:`AdaptiveQAEnv` actually emits, so a change in either side of that contract fails
here instead of at rollout time.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from src.decision import state_text
from src.decision.schema import ACTION_ORDER
from src.pipeline import build_retrieval, worlds
from src.rl.env import ANSWER, SEARCH_BM25, SEARCH_DENSE, AdaptiveQAEnv
from src.rl.reward import RewardCalculator, RewardConfig
from src.utils.hashing import sha256_text
from tests import p5_fixtures as fx


@pytest.fixture()
def env(tmp_path):
    cfg = fx.p5_config(tmp_path)
    fx.write_all(cfg, n_train=2, n_val=1)
    bundle = worlds.build_worlds(cfg, "train_strict", laya_regime=None)
    return AdaptiveQAEnv(bundle.worlds, RewardCalculator(RewardConfig.from_dict(cfg)),
                         worlds.env_config(cfg))


@pytest.fixture()
def root_obs(env):
    obs, _ = env.reset(seed=0, options={"qid": "train_strict:q000"})
    return {**obs, "candidate": dict(obs["candidate"]), "retrieval": dict(obs["retrieval"])}


def test_text_is_deterministic_and_carries_the_public_fields(root_obs):
    text = state_text.serialize_state(root_obs)
    assert text == state_text.serialize_state(root_obs)
    assert text.startswith("Câu hỏi:")
    assert "Bằng chứng đã xem: 1 đoạn" in text            # the forced bootstrap revealed one
    assert "Ứng viên trả lời:" in text
    assert "Điểm đọc:" in text and "Truy hồi:" in text
    assert "Hành động đã thực hiện: (chưa có)" in text
    assert "Số lượt tìm đã dùng: 0" in text
    assert f"Hành động hợp lệ: {', '.join(ACTION_ORDER)}" in text  # everything legal at t=0


def test_distinct_histories_render_distinct_text_and_hash(env):
    seen = {}
    for history in ((), (SEARCH_BM25,), (SEARCH_DENSE,)):
        obs, _ = env.reset(seed=0, options={"qid": "train_strict:q000"})
        for action in history:
            obs, *_ = env.step(action)
        text = state_text.serialize_state(obs)
        assert text not in seen.values(), f"history {history} collided with {seen}"
        seen[history] = text
    assert "Hành động đã thực hiện: SEARCH_BM25" in seen[(SEARCH_BM25,)]
    assert "Hành động đã thực hiện: SEARCH_DENSE" in seen[(SEARCH_DENSE,)]


def test_state_hash_is_a_pure_function_of_the_text(root_obs):
    text = state_text.serialize_state(root_obs)
    assert state_text.state_hash(text) == sha256_text(text)
    assert state_text.state_hash(text) == state_text.state_hash(text)
    assert state_text.state_hash(text) != state_text.state_hash(text + "x")


def test_forbidden_keys_are_rejected_before_rendering(root_obs):
    for needle in ("gold_answer", "is_impossible", "expected_action", "plausible_answers"):
        bad = {**root_obs, needle: "anything"}
        with pytest.raises(ValueError, match="forbidden"):
            state_text.serialize_state(bad)
        with pytest.raises(ValueError, match="forbidden"):
            state_text.assert_state_is_public(bad)


def test_nested_forbidden_keys_are_found(root_obs):
    bad = copy.deepcopy(root_obs)
    bad["candidate"]["plausible_answers"] = ["x"]
    with pytest.raises(ValueError, match="forbidden"):
        state_text.serialize_state(bad)


def test_a_missing_observation_key_is_named(root_obs):
    bad = {key: value for key, value in root_obs.items() if key != "action_mask"}
    with pytest.raises(KeyError, match="action_mask"):
        state_text.serialize_state(bad)


def test_a_forbidden_literal_in_public_text_is_a_leak(root_obs):
    bad = {**root_obs, "question": "gold_answer của câu hỏi này là gì?"}
    with pytest.raises(ValueError, match="leaks"):
        state_text.serialize_state(bad)


def test_long_evidence_is_snippet_capped(root_obs):
    obs = {**root_obs, "seen_doc_ids": ("doc:1",),
           "seen_doc_texts": (" ".join(["từ"] * 500),)}
    text = state_text.serialize_state(obs, max_snippet_chars=40)
    line = [ln for ln in text.splitlines() if ln.startswith("[1]")][0]
    assert line.endswith("…")
    assert len(line.split(" ", 2)[2]) == 41  # 40 chars + the ellipsis


def test_floats_render_with_fixed_precision_and_non_finite_as_na(root_obs):
    obs = copy.deepcopy(root_obs)
    obs["candidate"]["non_null_score"] = float("nan")
    obs["retrieval"]["bm25_top_gap"] = float("inf")
    text = state_text.serialize_state(obs)
    assert "non_null=n/a" in text and "khoảng cách n/a" in text
    obs["candidate"]["non_null_score"] = 1.0 / 3.0
    assert "non_null=0.333" in state_text.serialize_state(obs)


def test_masked_actions_are_absent_from_the_valid_list(env):
    obs, _ = env.reset(seed=0, options={"qid": "train_strict:q000"})
    obs, *_ = env.step(SEARCH_BM25)
    obs, *_ = env.step(SEARCH_DENSE)
    mask = np.asarray(obs["action_mask"], dtype=bool)
    assert not mask[1:3].any(), "the two optional searches are spent; both must be masked"
    text = state_text.serialize_state(obs)
    valid_line = text.split("Hành động hợp lệ: ", 1)[1].splitlines()[0]
    expected = ", ".join(name for index, name in enumerate(ACTION_ORDER) if mask[index])
    assert valid_line == expected
    assert "SEARCH_BM25" not in valid_line and "SEARCH_DENSE" not in valid_line


def test_answer_action_renders_as_history(env):
    obs, _ = env.reset(seed=0, options={"qid": "train_strict:q000"})
    obs, _, terminated, _, _ = env.step(ANSWER)
    assert terminated
    assert "Hành động đã thực hiện: ANSWER" in state_text.serialize_state(obs)


def test_a_state_cache_over_changed_sources_is_refused(tmp_path):
    """D011: worlds must refuse a cache whose recorded inputs changed since it was built."""
    cfg = fx.p5_config(tmp_path)
    fx.write_all(cfg, n_train=2, n_val=1)
    rankings = build_retrieval.rankings_path(cfg, "train_strict")
    with rankings.open("a", encoding="utf-8") as handle:
        handle.write("\n")  # any byte-level change must be detected
    with pytest.raises(ValueError, match="stale.*rankings changed.*rerun cache_reader"):
        worlds.build_worlds(cfg, "train_strict", laya_regime=None)
