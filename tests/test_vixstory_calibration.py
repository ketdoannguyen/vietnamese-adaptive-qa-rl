"""Contract tests for the isolated VixSTORY E03 calibrator (``src/pipeline/vixstory_calibration.py``).

The module fits ``p(correct)`` on train300 by cross-fitting, picks the ANSWER threshold ``tau``
on dev200 by the reward ``VixStoryEnv`` actually pays, never reads gold at inference and never
touches the official validation split. These tests drive the *real* environment with synthetic
worlds (no GPU, no data) and pin those four properties plus the artifact contract.
"""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from src.pipeline.vixstory_calibration import (
    VIXSTORY_CORRECTNESS_FEATURES,
    CalibrationRow,
    VixStoryCalibrationError,
    VixStoryCorrectnessEstimator,
    artifact_identity,
    collect_rows,
    cross_fit,
    e03_action,
    feature_matrix,
    features_from_obs,
    fit_calibrator,
    reward_criterion,
    run_vixstory_calibration,
    score_table,
    select_tau,
    simulate_episode,
    tau_candidates,
)
from src.rl.vixstory_env import (
    ABSTAIN,
    ANSWER,
    N_ACTIONS,
    SEARCH_BM25,
    SEARCH_DENSE,
    VixStoryEnv,
    VixStoryEnvConfig,
    VixStoryPrivate,
    VixStoryReaderCache,
    VixStoryReaderPrediction,
    VixStoryWorld,
)

# --- synthetic worlds ------------------------------------------------------------------
CORPUS = {"d0": "Trường A toạ lạc tại Hà Nội.", "d1": "Trường A thành lập năm 1995.",
          "d2": "Trường A có 12 khoa.", "d3": "Đoạn nhiễu không liên quan."}
#: Bootstrap (BM25, 1) -> d0; the two tools then diverge to d1 / d2.
RANKINGS = {"bm25": ["d0", "d1", "d2", "d3"], "dense": ["d0", "d2", "d1", "d3"]}
GOLD = "1995"
WRONG = "một câu trả lời hoàn toàn khác"
Q_SHORT = "Trường A có đặc điểm gì?"
Q_LONG = "Trường A được thành lập vào năm nào và hiện có bao nhiêu khoa cùng viện nghiên cứu?"


def _pred(qid: str, *, text: str | None, cited: tuple[str, ...] = (), valid: bool = False):
    return VixStoryReaderPrediction(qid=qid, text=text, cited_ids=cited, citation_valid=valid)


def make_world(qid: str, *, correct: bool = True, question: str = Q_SHORT) -> VixStoryWorld:
    """A world whose cached reader answers ``GOLD`` or ``WRONG`` at every post-search state."""
    cache = VixStoryReaderCache(reader_sha="r", tokenizer_sha="t", inference_config_sha="i")
    text = GOLD if correct else WRONG
    preds = {
        ("d0",): _pred(qid, text=None),
        ("d0", "d1"): _pred(qid, text=text, cited=("d1",), valid=correct),
        ("d0", "d2"): _pred(qid, text=text, cited=("d2",), valid=correct),
        ("d0", "d1", "d2"): _pred(qid, text=text, cited=("d1", "d2"), valid=correct),
        ("d0", "d2", "d1"): _pred(qid, text=text, cited=("d1", "d2"), valid=correct),
    }
    for visible, prediction in preds.items():
        cache.put(qid, visible, prediction)
    return VixStoryWorld(
        qid=qid, question=question, split="train_strict", scenario_id=f"sc:{qid}",
        corpus=CORPUS, rankings=RANKINGS,
        private=VixStoryPrivate(qid=qid, answer_texts=(GOLD,), gold_context_ids=("d1",)),
        reader=cache.reader(qid))


def env_for(worlds) -> VixStoryEnv:
    return VixStoryEnv(list(worlds))


def train_env(n: int = 16) -> tuple[VixStoryEnv, list[str]]:
    """Half the questions answer correctly, half do not, so both label classes exist."""
    qids = [f"q{i:02d}" for i in range(n)]
    worlds = [make_world(q, correct=(i % 2 == 0)) for i, q in enumerate(qids)]
    return env_for(worlds), qids


def dev_env(n: int = 8, *, correct: bool = True, question: str = Q_SHORT):
    qids = [f"d{i:02d}" for i in range(n)]
    worlds = [make_world(q, correct=correct, question=question) for q in qids]
    return env_for(worlds), qids


def _obs_after(env: VixStoryEnv, qid: str, history):
    env.reset(options={"qid": qid})
    obs = None
    for action in history:
        obs, _, _, _, _ = env.step(int(action))
    return obs


# --- feature vector: observation-only, gold-free ---------------------------------------
def test_features_from_obs_is_fixed_width_finite_and_gold_free():
    env = env_for([make_world("q", correct=True)])
    first = features_from_obs(_obs_after(env, "q", [SEARCH_BM25]))
    assert first.shape == (len(VIXSTORY_CORRECTNESS_FEATURES),)
    assert np.all(np.isfinite(first))

    # Same visible reader output, different private gold: not a single feature may move.
    secret = make_world("q", correct=True)
    secret.private = VixStoryPrivate(qid="q", answer_texts=("bí mật",), gold_context_ids=("d3",))
    other = features_from_obs(_obs_after(env_for([secret]), "q", [SEARCH_BM25]))
    assert np.array_equal(first, other)


def test_features_from_obs_rejects_a_wrong_shaped_mask():
    obs = {"candidate": {}, "retrieval": {}, "action_mask": np.zeros(3, dtype=bool)}
    with pytest.raises(ValueError, match="action_mask"):
        features_from_obs(obs)


# --- no gold at inference ---------------------------------------------------------------
def test_score_table_and_trajectory_are_gold_free():
    def per_private(private: VixStoryPrivate):
        world = make_world("q", correct=True)
        world.private = private
        env = env_for([world])
        estimator = VixStoryCorrectnessEstimator.constant(rate=0.6)
        scores = score_table(env, estimator, ["q"])
        return scores, simulate_episode(env, "q", scores, tau=0.5)

    scores_a, episode_a = per_private(VixStoryPrivate(qid="q", answer_texts=("1995",),
                                                       gold_context_ids=("d1",)))
    scores_b, episode_b = per_private(VixStoryPrivate(qid="q", answer_texts=("bí mật",),
                                                       gold_context_ids=("d3",)))
    assert scores_a == scores_b  # identical confidence on every state
    assert (episode_a.answered, episode_a.n_searches, episode_a.terminal_action) == (
        episode_b.answered, episode_b.n_searches, episode_b.terminal_action)
    assert episode_a.correct and not episode_b.correct  # only the *measured* label moves


def test_e03_action_is_a_pure_function_of_obs_p_and_tau():
    mask = np.zeros(N_ACTIONS, dtype=bool)
    mask[ANSWER] = mask[SEARCH_BM25] = True
    obs = {"action_mask": mask}
    assert e03_action(obs, p_correct=0.9, tau=0.5) == ANSWER
    assert e03_action(obs, p_correct=0.4, tau=0.5) == SEARCH_BM25
    dense_only = np.zeros(N_ACTIONS, dtype=bool)
    dense_only[SEARCH_DENSE] = True
    assert e03_action({"action_mask": dense_only}, p_correct=0.9, tau=0.5) == SEARCH_DENSE
    none_valid = np.zeros(N_ACTIONS, dtype=bool)  # everything masked -> ABSTAIN
    assert e03_action({"action_mask": none_valid}, p_correct=0.9, tau=0.5) == ABSTAIN


# --- scaler fit on train only -----------------------------------------------------------
def test_scaler_is_fit_on_train_rows_and_never_on_dev():
    train, train_qids = train_env(16)
    dev, dev_qids = dev_env(8, question=Q_LONG)
    summary = run_vixstory_calibration(train, train_qids=train_qids, dev_env=dev, dev_qids=dev_qids)
    assert summary["status"] == "SUCCEEDED"

    train_mean = feature_matrix(collect_rows(train, train_qids)).mean(axis=0)
    dev_mean = feature_matrix(collect_rows(dev, dev_qids)).mean(axis=0)
    scaler_mean = np.asarray(summary["estimator"]["scaler_mean"])
    assert np.allclose(scaler_mean, train_mean)
    # The dev question is much longer; had dev rows leaked into the fit, this would have moved.
    q_len = VIXSTORY_CORRECTNESS_FEATURES.index("question_char_len")
    assert not np.isclose(scaler_mean[q_len], dev_mean[q_len])


# --- tau chosen on dev by the environment reward ----------------------------------------
def test_run_selects_tau_by_dev_reward_and_does_not_abstain_by_default():
    train, train_qids = train_env(16)
    dev, dev_qids = dev_env(8)
    summary = run_vixstory_calibration(train, train_qids=train_qids, dev_env=dev, dev_qids=dev_qids)
    assert summary["status"] == "SUCCEEDED"
    selection = summary["selection"]
    assert summary["tau"] == selection["selected_tau"]
    best = max(row["mean_episode_reward"] for row in selection["log"])
    assert selection["selected"]["mean_episode_reward"] == pytest.approx(best)
    # Answering a correct dev question pays 1.0 - 0.05, so a correct-answering tau beats the
    # never-answer policy (which only pays the search costs).
    assert selection["selected_is_never_answer"] is False


def test_select_tau_breaks_ties_toward_the_higher_tau_and_flags_the_sentinel():
    candidates = [0.2, 0.5, 0.8, 1.5]  # 1.5 is the max -> the never-answer sentinel
    flat = select_tau(candidates, lambda tau: {"mean_episode_reward": 0.3})
    assert flat["selected_tau"] == 1.5
    assert flat["selected_is_never_answer"] is True
    peaked = select_tau(candidates, lambda tau: {"mean_episode_reward": 1.0 if tau == 0.8 else 0.0})
    assert peaked["selected_tau"] == 0.8
    assert peaked["selected_is_never_answer"] is False


def test_simulate_episode_requires_a_score_table_covering_every_visited_state():
    dev, dev_qids = dev_env(2)
    with pytest.raises(VixStoryCalibrationError, match="was not scored"):
        simulate_episode(dev, dev_qids[0], scores={}, tau=0.5)


def test_simulate_episode_scores_the_return_the_environment_pays():
    dev, dev_qids = dev_env(4, correct=True)
    # A sure-everywhere model answers as soon as a candidate exists.
    scores = score_table(dev, VixStoryCorrectnessEstimator.constant(rate=1.0), dev_qids)
    result = simulate_episode(dev, dev_qids[0], scores, tau=0.5)
    assert result.answered and result.correct
    assert result.n_searches == 1  # the root has no candidate; one SEARCH reveals the answer
    assert result.reward == pytest.approx(1.0 - 0.05)


# --- one-class labels are refused, never faked ------------------------------------------
def test_single_class_labels_deploy_a_flagged_constant_not_a_model():
    qids = [f"q{i:02d}" for i in range(6)]
    env = env_for([make_world(q, correct=False) for q in qids])
    rows = collect_rows(env, qids)
    assert set(int(r.y) for r in rows) == {0}

    with pytest.raises(ValueError, match="both classes"):
        VixStoryCorrectnessEstimator.fit(feature_matrix(rows), [r.y for r in rows])

    with pytest.raises(ValueError, match="no fitted logistic model"):
        VixStoryCorrectnessEstimator.constant(rate=0.0).score(feature_matrix(rows))

    dev, dev_qids = dev_env(2)
    summary = run_vixstory_calibration(env, train_qids=qids, dev_env=dev, dev_qids=dev_qids)
    assert summary["status"] == "DEGRADED"
    assert summary["estimator"]["fitted"] is False
    assert summary["estimator"]["constant"] == pytest.approx(0.0)
    assert any("single-class" in blocker for blocker in summary["blockers"])


def test_fit_calibrator_skips_instead_of_fabricating():
    calibrator, blocks = fit_calibrator([0.1, 0.2], [0, 1])
    assert calibrator.mode == "none" and blocks[0]["reason"] == "too_few_oof_rows"
    calibrator, blocks = fit_calibrator([0.1] * 60, [1] * 60)
    assert calibrator.mode == "none" and blocks[0]["reason"] == "single_class_oof"
    calibrator, blocks = fit_calibrator([float("nan")] + [0.5] * 59, [0, 1] * 30)
    assert calibrator.mode == "none"
    assert blocks[0]["reason"] == "calibrator_received_non_finite_scores"


# --- cross-fit collects logit-scale scores (the calibrator input scale) ------------------
def test_cross_fit_oof_scores_are_logits_not_probabilities():
    rng = np.random.default_rng(0)
    dim = len(VIXSTORY_CORRECTNESS_FEATURES)
    rows: list[CalibrationRow] = []
    for q in range(24):
        label = q % 2
        for s in range(4):
            features = rng.normal(0.0, 0.3, size=dim)
            features[0] = float(label)
            features[2] = 2.0 * label
            rows.append(CalibrationRow(qid=f"q{q:02d}", state_hash=f"{q}-{s}", y=label,
                                       f1=float(label), features=features))
    cross = cross_fit(rows, n_folds=3, seed=0)
    oof = cross.oof_scores
    assert np.isfinite(oof).all()
    # A probability would be trapped in (0, 1); logits are not. The calibrator is fit on these
    # scores and applied to the deployed model's score, so they must be scores, not sigmoids.
    assert not np.all((oof > 0.0) & (oof < 1.0))
    assert cross.estimator.calibrator.mode == "platt"
    transformed = cross.estimator.calibrator.transform(oof)
    assert np.all((transformed >= 0.0) & (transformed <= 1.0))


# --- no validation split -----------------------------------------------------------------
def test_run_accepts_no_validation_split():
    params = inspect.signature(run_vixstory_calibration).parameters
    assert not any("valid" in name for name in params)
    assert {"train_qids", "dev_env", "dev_qids"} <= set(params)


# --- artifacts ---------------------------------------------------------------------------
def test_artifact_roundtrip_and_refusals(tmp_path):
    train, train_qids = train_env(16)
    dev, dev_qids = dev_env(8)
    summary = run_vixstory_calibration(train, train_qids=train_qids, dev_env=dev, dev_qids=dev_qids,
                                       save_dir=tmp_path)
    assert summary["status"] == "SUCCEEDED"
    selection_file = json.loads((tmp_path / "threshold_selection.json").read_text(encoding="utf-8"))
    assert selection_file["tau"] == summary["tau"]
    assert selection_file["split"] == "dev200"

    probe = np.zeros((3, len(VIXSTORY_CORRECTNESS_FEATURES)))
    loaded = VixStoryCorrectnessEstimator.load(tmp_path, expected_identity=summary["identity"])
    assert loaded.fitted and loaded.calibrated
    assert loaded.feature_names == list(VIXSTORY_CORRECTNESS_FEATURES)
    probabilities = loaded.predict(probe)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    # Reloading the same artifact reproduces the same probabilities.
    again = VixStoryCorrectnessEstimator.load(tmp_path, expected_identity=summary["identity"])
    assert np.array_equal(probabilities, again.predict(probe))

    with pytest.raises(ValueError, match="identity"):
        VixStoryCorrectnessEstimator.load(tmp_path, expected_identity="deadbeef")

    model_path = tmp_path / "calibrator_model.json"
    original = json.loads(model_path.read_text(encoding="utf-8"))

    model_path.write_text(json.dumps(dict(original, schema="something-else")), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        VixStoryCorrectnessEstimator.load(tmp_path)

    tampered = dict(original, feature_names=list(reversed(original["feature_names"])))
    model_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="feature list"):
        VixStoryCorrectnessEstimator.load(tmp_path)

    tampered = json.loads(json.dumps(original))
    tampered["meta"]["reward_criterion"]["f1_threshold"] = 0.99  # identity no longer matches
    model_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        VixStoryCorrectnessEstimator.load(tmp_path)


def test_artifact_identity_binds_the_reward_criterion():
    base, other = VixStoryEnvConfig(), VixStoryEnvConfig(search_cost=0.5)
    assert reward_criterion(base) != reward_criterion(other)
    assert artifact_identity(reward_criterion(base)) != artifact_identity(reward_criterion(other))
    assert (artifact_identity(reward_criterion(base))
            == artifact_identity(reward_criterion(VixStoryEnvConfig())))


def test_tau_candidates_include_a_never_answer_sentinel():
    grid = tau_candidates([0.1, 0.4, 0.7, 0.9], n_quantiles=4)
    assert max(grid) > 0.9  # sentinel above every observed score
    assert grid == sorted(grid)
