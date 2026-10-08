"""B5 correctness-estimator tests (`02_IMPLEMENTATION/08_...`, spec §7).

CPU-only and synthetic: no reader is trained, no real cache is read, no GPU. The end-to-end
tests patch the three boundaries the stage crosses — ``worlds.build_worlds`` (toy episodes
instead of the P1/P2/P3 caches), ``train_correctness.title_groups_for`` (title groups
without the label store) and ``train_correctness.{p3_reader_source,
train_fold_reader_source}`` (fixed reader predictions instead of a trained model). What the
tests hold the pipeline to:

* the estimator's input is the *observation* only — swapping the hidden gold moves the
  label and not a single feature;
* the scaler/logistic model may only be fit on ``train_strict`` and the calibrator only on
  ``val_strict``; the wrong split raises, in the library and through the pipeline;
* ``test_strict`` is never requested from ``build_worlds``;
* the fit rows come from the *fold* reader, not from the deployed reader (the reader
  optimism guard of spec 08 §3 is real, not a comment);
* a calibrated estimator is only called calibrated when a calibrator was actually fit;
  otherwise the run is named B5-RAW and no raw margin is ever reported as a probability.
"""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.eval import correctness as C
from src.eval import splitting
from src.pipeline import train_correctness as tc
from src.pipeline import worlds
from src.pipeline.worlds import WorldBundle
from src.qa.reader import ReaderPrediction
from src.rl.env import ABSTAIN, ANSWER, SEARCH_BM25, SEARCH_DENSE, AdaptiveQAEnv, EnvConfig, EpisodeWorld
from src.rl.features import FeatureScaler
from src.rl.reward import EpisodePrivate, RewardCalculator, RewardConfig
from src.utils.config import load_config, resolve
from src.utils.io import read_json, read_jsonl, write_yaml
from tests.toy_world import DOC_A, DOC_B, make_reader, span_prediction

#: Reader margins the fixtures serve: (wrong span at the bootstrap doc, gold span after a
#: search). The fold reader deliberately differs from the deployed one so a scaler fitted on
#: the deployed predictions can be told apart from the honest out-of-fold one.
TRAIN_MARGINS = (1.0, 5.0)
FOLD_MARGINS = (2.0, 6.0)
VAL_MARGINS = (1.0, 5.0)

QUESTION = "Phạm Văn Đồng giữ chức Thủ tướng Chính phủ Việt Nam từ năm nào?"
GOLD = "1976"
WRONG = "1010"


# ------------------------------------------------------------------------------ fixtures

def _cfg(root: Path) -> dict:
    """A self-contained config: every path inside ``root`` so no test writes to the repo."""
    cfg = load_config(None)
    cfg["paths"] = {key: str(root / key) for key in cfg["paths"]}
    cfg["data"] = {**cfg["data"], "profile": "smoke"}
    cfg["retrieval"] = {**cfg["retrieval"], "initial_bootstrap": "bm25",
                        "bootstrap_passages": 1, "passage_per_search_action": 1,
                        "optional_searches": 2}
    cfg["rl"] = {**cfg["rl"], "max_optional_search": 2}
    # Optional B5 knobs; defaults live in train_correctness, the real run may tune them.
    cfg["correctness"] = {"tau_quantiles": 8, "bootstrap_resamples": 32}
    return cfg


def _world(qid: str, *, split: str, margins: tuple[float, float],
           root: tuple[str, str] = ("doc_a", WRONG),
           reveal: str = "doc_b", post: tuple[str, str] = ("doc_b", GOLD),
           gold: str = GOLD) -> tuple[EpisodeWorld, dict]:
    """One episode whose answer *can* improve after a search, plus its prediction table.

    Bootstrap surfaces ``root[0]``, where the reader returns ``root[1]``; one BM25 (or dense)
    search reveals ``reveal``, after which the reader returns ``post[1]`` from ``post[0]``.
    The default fixture is wrong at the bootstrap doc (``y=0``) and right after the search
    (``y=1``), so any subset of its questions carries both label classes — which keeps the
    fixtures independent of which groups a fold happens to hold out.
    """
    docs = {"doc_a": DOC_A, "doc_b": DOC_B}
    table = {
        (root[0],): span_prediction(qid, root[0], docs[root[0]], root[1], margins[0]),
        (root[0], reveal): span_prediction(qid, post[0], docs[post[0]], post[1], margins[1]),
    }
    world = EpisodeWorld(
        qid=qid, question=QUESTION, split=split, scenario_id=f"s_{qid}",
        corpus=dict(docs),
        rankings={"bm25": [root[0], reveal], "dense": [reveal, root[0]]},
        private=EpisodePrivate(qid=qid, answer_texts=(gold,), is_impossible=False,
                               gold_doc_id=post[0], gold_in_corpus=True, gold_reachable=True),
        reader=make_reader(dict(table), docs),
    )
    # Real per-tool score maps: `with_reader` must carry them across (rank-proxy fallback
    # would silently change the retrieval features).
    world._scores_bm25 = {"doc_a": 8.0, "doc_b": 5.0}
    world._scores_dense = {"doc_a": 6.0, "doc_b": 7.0}
    return world, table


def _bundle(split: str, qids: list[str], *, margins: tuple[float, float], **kwargs
            ) -> tuple[WorldBundle, dict[str, dict]]:
    world_list, tables = [], {}
    for qid in qids:
        world, table = _world(qid, split=split, margins=margins, **kwargs)
        world_list.append(world)
        tables[qid] = table
    bundle = WorldBundle(split=split, worlds=world_list, questions={}, docs={},
                         laya_fn=None, ambiguous_qids=frozenset())
    return bundle, tables


def _source(name: str, tables: dict[str, dict], *, split: str) -> tc.ReaderSource:
    """ReaderSource over explicit ``(qid, visible set) -> prediction`` tables."""
    flat = {(qid, frozenset(visible)): pred
            for qid, table in tables.items() for visible, pred in table.items()}

    def lookup(qid: str, visible: tuple[str, ...]) -> ReaderPrediction:
        return flat[(qid, frozenset(visible))]

    return tc.ReaderSource(name=name, model_hash=f"hash_{name}", tokenizer_sha="tok_sha",
                           inference_config_sha="inf_sha", threshold=0.0, lookup=lookup,
                           meta={"split": split})


def _rows_for(cfg: dict, split: str, qids: list[str], *, margins: tuple[float, float],
              **kwargs) -> list[dict]:
    """Recompute labeled rows exactly the way the pipeline does, for an independent check."""
    bundle, tables = _bundle(split, list(qids), margins=margins, **kwargs)
    source = _source(f"recompute[{split}]", tables, split=split)
    selected, env = tc.world_set(cfg, bundle, source)
    reward = RewardCalculator(RewardConfig.from_dict(dict(cfg)))
    return tc.labeled_rows(tc.reachable_states(env, selected), reward)


def _install(tmp_path: Path, monkeypatch, *, fold_margins: tuple[float, float] = FOLD_MARGINS,
             world_kwargs: dict | None = None) -> SimpleNamespace:
    """Patch the pipeline's data/reader boundaries to toy fixtures; return the wiring."""
    cfg = _cfg(tmp_path)
    kwargs = dict(world_kwargs or {})
    train_qids = [f"train_strict:{i}" for i in range(6)]
    val_qids = [f"val_strict:{i}" for i in range(6)]
    train_bundle, train_tables = _bundle(tc.TRAIN_SPLIT, train_qids, margins=TRAIN_MARGINS,
                                         **kwargs)
    val_bundle, val_tables = _bundle(tc.VAL_SPLIT, val_qids, margins=VAL_MARGINS, **kwargs)
    sources = {
        tc.TRAIN_SPLIT: _source("p3_full_train[train_strict]", train_tables, split=tc.TRAIN_SPLIT),
        tc.VAL_SPLIT: _source("p3_full_val[val_strict]", val_tables, split=tc.VAL_SPLIT)}
    groups = {qid: f"tg:{qid}" for qid in train_qids + val_qids}
    calls: dict[str, list] = {"build_worlds": [], "folds": []}

    def fake_build_worlds(cfg_arg, split, *, limit=None,
                          laya_regime=worlds.DEFAULT_LAYA_SENTINEL):
        calls["build_worlds"].append({"split": split, "limit": limit,
                                      "laya_regime": laya_regime})
        return {tc.TRAIN_SPLIT: train_bundle, tc.VAL_SPLIT: val_bundle}[split]

    def fake_title_groups(cfg_arg, split, qids):
        missing = sorted(set(qids) - set(groups))
        if missing:
            raise KeyError(f"no title group for {missing[:3]}")
        return {qid: groups[qid] for qid in qids}

    def fake_p3_reader_source(cfg_arg, split):
        return sources[split]

    def fake_train_fold_reader_source(cfg_arg, *, profile, seed, reader_train_qids,
                                      feature_qids, out_dir, fold_index, limit=None):
        calls["folds"].append({
            "fold_index": fold_index, "seed": seed, "limit": limit,
            "reader_train_qids": sorted(reader_train_qids),
            "feature_qids": sorted(feature_qids), "out_dir": str(out_dir)})
        _, fold_tables = _bundle(tc.TRAIN_SPLIT, sorted(feature_qids), margins=fold_margins,
                                 **kwargs)
        source = _source(f"oof_fold_{fold_index}", fold_tables, split=tc.TRAIN_SPLIT)
        manifest = {"index": fold_index, "reader_dir": str(out_dir),
                    "n_reader_rows": len(reader_train_qids) * 3,
                    "n_feature_questions": len(feature_qids),
                    "feature_qids": sorted(feature_qids)}
        return source, manifest

    monkeypatch.setattr(tc.worlds, "build_worlds", fake_build_worlds)
    monkeypatch.setattr(tc, "title_groups_for", fake_title_groups)
    monkeypatch.setattr(tc, "p3_reader_source", fake_p3_reader_source)
    monkeypatch.setattr(tc, "train_fold_reader_source", fake_train_fold_reader_source)
    return SimpleNamespace(cfg=cfg, calls=calls, groups=groups, sources=sources,
                           train_bundle=train_bundle, val_bundle=val_bundle,
                           train_tables=train_tables, val_tables=val_tables,
                           train_qids=train_qids, val_qids=val_qids, fold_margins=fold_margins,
                           kwargs=kwargs, root=tmp_path)


def _registry(cfg: dict) -> list[dict]:
    return list(read_jsonl(resolve(cfg, "runs") / "index.jsonl"))


# ------------------------------------------------------------- observation safety + labels

def test_features_are_a_pure_function_of_the_observation():
    world, table = _world("feat:0", split=tc.TRAIN_SPLIT, margins=(1.0, 5.0))
    source = _source("feat", {"feat:0": table}, split=tc.TRAIN_SPLIT)
    reward = RewardCalculator(RewardConfig())
    env = AdaptiveQAEnv([tc.with_reader(world, source)], reward, EnvConfig())
    obs, _ = env.reset(seed=0, options={"qid": "feat:0"})

    vector = tc.features_from_obs(obs)
    assert vector.shape == (len(C.CORRECTNESS_FEATURES),)
    # qid and every other hidden-side quantity are not inputs.
    assert np.array_equal(vector, tc.features_from_obs({**obs, "qid": "other:qid"}))

    # The gold swap flips the label and leaves the feature vector bit-identical.
    swapped = replace(world, private=replace(world.private, answer_texts=(WRONG,)))
    base = tc.state_row(world, obs, "h", reward)
    other = tc.state_row(swapped, obs, "h", reward)
    assert np.array_equal(base["features"], other["features"])
    assert base["y"] == 0 and other["y"] == 1

    # No feature name may even suggest a gold/label/reward/Laya input (spec 08 §2).
    forbidden = ("gold", "reward", "laya", "label", "answer")
    assert not [name for name in C.CORRECTNESS_FEATURES
                if any(word in name for word in forbidden)]


def test_labels_follow_the_reward_criterion_including_provenance():
    world, table = _world("lab:0", split=tc.TRAIN_SPLIT, margins=(1.0, 5.0))
    source = _source("lab", {"lab:0": table}, split=tc.TRAIN_SPLIT)
    reward = RewardCalculator(RewardConfig())
    env = AdaptiveQAEnv([tc.with_reader(world, source)], reward, EnvConfig())
    root, _ = env.reset(seed=0, options={"qid": "lab:0"})
    after_search, _, _, _, _ = env.step(SEARCH_BM25)

    assert tc.state_row(world, root, "h0", reward)["y"] == 0
    gold_row = tc.state_row(world, after_search, "h1", reward)
    assert gold_row["y"] == 1 and gold_row["f1"] == pytest.approx(1.0)

    # Provenance invalid: the text matches gold, the criterion still says wrong.
    broken = {**after_search,
              "candidate": {**after_search["candidate"], "provenance_ok": False}}
    broken_row = tc.state_row(world, broken, "h1", reward)
    assert broken_row["f1"] == pytest.approx(1.0) and broken_row["y"] == 0

    # A state without a candidate is not a training row at all.
    empty = {**after_search, "candidate": {**after_search["candidate"], "text": None}}
    assert tc.state_row(world, empty, "h2", reward) is None


# ---------------------------------------------------------------------- split discipline

def test_model_and_calibrator_reject_the_wrong_split():
    rng = np.random.default_rng(0)
    features = np.vstack([rng.normal(size=(20, len(C.CORRECTNESS_FEATURES))),
                          rng.normal(size=(20, len(C.CORRECTNESS_FEATURES))) + 2.0])
    labels = np.array([0] * 20 + [1] * 20)

    with pytest.raises(ValueError, match="train_strict"):
        C.CorrectnessEstimator.fit(features, labels, split=C.CALIBRATION_SPLIT)
    with pytest.raises(ValueError, match="train_strict"):
        C.CorrectnessEstimator.fit(features, labels, split="test_strict")
    with pytest.raises(ValueError, match="val_strict"):
        C.fit_calibrator([0.1, 0.9], [0, 1], split=C.FIT_SPLIT)
    with pytest.raises(ValueError, match="val_strict"):
        C.fit_calibrator([0.1, 0.9], [0, 1], split="test_strict")

    # The fit-scope guard also covers the pipeline's own fit helper.
    rows = [{"features": features[i], "y": int(labels[i]), "qid": f"q{i}", "state_hash": f"h{i}",
             "f1": float(labels[i]), "provenance_ok": True, "is_impossible": False,
             "split": tc.TRAIN_SPLIT} for i in range(len(labels))]
    estimator = tc.fit_estimator(rows, seed=3, variant="B5", meta={})
    assert np.asarray(estimator.scaler.mean).shape == (len(C.CORRECTNESS_FEATURES),)
    expected = FeatureScaler.fit(tc.feature_matrix(rows))
    assert np.allclose(estimator.scaler.mean, expected.mean)
    assert np.allclose(estimator.scaler.std, expected.std)


def test_calibration_plan_boundaries_and_raw_refusal():
    assert C.calibration_plan(C.MIN_CAL_N - 1, 40, 40) == "raw"
    assert C.calibration_plan(C.MIN_CAL_N, C.MIN_CAL_CLASS - 1,
                              C.MIN_CAL_N - C.MIN_CAL_CLASS + 1) == "raw"
    assert C.calibration_plan(C.MIN_CAL_N, C.MIN_CAL_CLASS,
                              C.MIN_CAL_N - C.MIN_CAL_CLASS) == "platt"
    assert C.calibration_plan(C.ISOTONIC_MIN_N, 300, 200) == "isotonic"

    with pytest.raises(ValueError, match="B5-RAW"):
        C.fit_calibrator([0.2, 0.8], [0, 1], split=C.CALIBRATION_SPLIT, plan="raw")
    with pytest.raises(ValueError, match="both classes"):
        C.fit_calibrator([0.2, 0.8], [1, 1], split=C.CALIBRATION_SPLIT, plan="platt")
    with pytest.raises(ValueError, match="outside"):
        C.reliability_metrics([1.5, 0.5], [0, 1])


# ------------------------------------------------------------------ persistence + naming

def test_save_load_roundtrip_is_exact(tmp_path):
    rng = np.random.default_rng(1)
    features = np.vstack([rng.normal(size=(25, len(C.CORRECTNESS_FEATURES))),
                          rng.normal(size=(25, len(C.CORRECTNESS_FEATURES))) + 1.5])
    labels = np.array([0] * 25 + [1] * 25)
    estimator = C.CorrectnessEstimator.fit(features, labels, split=C.FIT_SPLIT, seed=5)
    calibrated = tc.with_calibrator(
        estimator, C.fit_calibrator(estimator.predict(features), labels,
                                    split=C.CALIBRATION_SPLIT, plan="platt"))
    assert calibrated.is_calibrated and calibrated.variant == "B5"

    path = calibrated.save(tmp_path / "estimator")
    assert (path / "model.json").is_file() and (path / "scaler.json").is_file()
    assert (path / "calibrator.json").is_file()
    loaded = C.CorrectnessEstimator.load(path)

    before = calibrated.predict(features)
    after = loaded.predict(features)
    assert np.array_equal(before, after)  # identical scores, not merely close
    assert loaded.is_calibrated and loaded.calibrator.mode == "platt" and loaded.fitted
    assert loaded.feature_names == list(C.CORRECTNESS_FEATURES)
    assert json.loads((path / "model.json").read_text(encoding="utf-8"))["variant"] == "B5"


def test_b5_raw_never_pretends_to_be_a_probability():
    rng = np.random.default_rng(2)
    features = np.vstack([rng.normal(size=(30, len(C.CORRECTNESS_FEATURES))),
                          rng.normal(size=(30, len(C.CORRECTNESS_FEATURES))) + 1.0])
    labels = np.array([0] * 30 + [1] * 30)
    raw = C.CorrectnessEstimator.fit(features, labels, split=C.FIT_SPLIT, variant="B5-RAW")
    assert raw.fitted and not raw.is_calibrated  # a model exists, no calibrator does
    assert raw.probability_from_obs({"candidate": {"null_margin": 3.5}}) == 3.5

    # Single-class OOF rows: no logistic model; predict must refuse instead of inventing.
    one_class = [{"features": features[i], "y": 1, "qid": f"q{i}", "state_hash": f"h{i}",
                  "f1": 1.0, "provenance_ok": True, "is_impossible": False,
                  "split": tc.TRAIN_SPLIT} for i in range(5)]
    dummy = tc.fit_estimator(one_class, seed=3, variant="B5-RAW", meta={})
    assert dummy.fitted is False and dummy.variant == "B5-RAW"
    assert dummy.probability_from_obs({"candidate": {"null_margin": 2.5}}) == 2.5
    with pytest.raises(ValueError, match="null_margin"):
        dummy.predict(features[:1])

    # A calibrated-variant fit on single-class rows is a hard error, not a fallback.
    with pytest.raises(ValueError, match="both classes"):
        tc.fit_estimator(one_class, seed=3, variant="B5", meta={})


# --------------------------------------------------------------------------- B5 policy

def test_b5_action_rule_respects_the_mask():
    always = {"action_mask": np.array([True, True, True, True])}
    assert tc.b5_action(always, 0.9, 0.5) == ANSWER
    assert tc.b5_action(always, 0.5, 0.5) == ANSWER  # >= tau, not > tau
    assert tc.b5_action(always, 0.49, 0.5) == SEARCH_BM25

    assert tc.b5_action({"action_mask": np.array([True, False, True, True])},
                        0.0, 0.5) == SEARCH_DENSE
    assert tc.b5_action({"action_mask": np.array([False, True, True, True])},
                        10.0, 0.5) == SEARCH_BM25
    assert tc.b5_action({"action_mask": np.array([True, False, False, True])},
                        0.1, 0.5) == ABSTAIN


def test_tau_grid_is_gold_free_and_select_tau_breaks_ties_upward():
    grid = tc.tau_candidates([1.0, 5.0], n_quantiles=8)
    assert grid[-1] == 6.0  # never-answer sentinel = max score + 1
    assert all(1.0 <= tau <= 5.0 for tau in grid[:-1])
    assert 1.0 in grid and 5.0 in grid

    selection = tc.select_tau([0.2, 0.5, 1.5],
                              lambda tau: {"mean_episode_reward":
                                           {0.2: -1.0, 0.5: 2.0, 1.5: -0.5}[tau]})
    assert selection["selected_tau"] == 0.5 and selection["selected_is_never_answer"] is False
    assert selection["criterion"] == tc.PRE_REGISTERED_CRITERION
    assert [row["tau"] for row in selection["log"]] == [0.2, 0.5, 1.5]

    tie = tc.select_tau([0.3, 0.7], lambda tau: {"mean_episode_reward": 1.0})
    assert tie["selected_tau"] == 0.7  # ties go to the higher tau

    sentinel = tc.select_tau([0.2, 0.7, 1.7],
                             lambda tau: {"mean_episode_reward": 3.0 if tau == 1.7 else 0.0})
    assert sentinel["selected_tau"] == 1.7 and sentinel["selected_is_never_answer"] is True


def test_holdout_specs_are_group_disjoint_and_cover_every_question():
    groups = {f"q{i}": f"g{i}" for i in range(12)}
    for reader_qids, feature_qids in splitting.holdout_specs(groups, 3, seed=7):
        assert set(reader_qids).isdisjoint(feature_qids)
        assert set(reader_qids) | set(feature_qids) == set(groups)

    (reader_qids, feature_qids), = splitting.holdout_specs(groups, 1, seed=7, fraction=0.25)
    assert 0 < len(feature_qids) < len(groups)
    assert set(reader_qids).isdisjoint(feature_qids)

    grouped = {f"q{i}": f"g{i % 4}" for i in range(12)}  # 4 groups of 3 questions
    for reader_qids, feature_qids in splitting.holdout_specs(grouped, 3, seed=7):
        assert {grouped[q] for q in reader_qids}.isdisjoint({grouped[q] for q in feature_qids})


# ------------------------------------------------------------------------------ dry run

def test_dry_run_cli_writes_no_estimator(tmp_path, monkeypatch, capsys):
    wiring = _install(tmp_path, monkeypatch)
    overlay = tmp_path / "overlay.yaml"
    write_yaml(overlay, wiring.cfg)

    assert tc.main(["--config", str(overlay), "--dry-run", "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["dry_run"] is True and summary["limit"] is None
    assert summary["probe"]["probe_states"] > 0
    assert summary["probe"]["oof_folds"] == 1
    assert wiring.calls["folds"] == []  # no fold reader is trained for a wiring check
    assert not tc.estimator_dir(wiring.cfg).exists()
    assert not (resolve(wiring.cfg, "reports") / tc.REPORT_FILE).exists()
    assert [row["status"] for row in _registry(wiring.cfg)] == ["RUNNING", "SUCCEEDED"]


# ----------------------------------------------------------------------------- end to end

def test_run_end_to_end_deploys_b5_raw(tmp_path, monkeypatch):
    wiring = _install(tmp_path, monkeypatch)
    cfg = wiring.cfg
    summary = tc.run(cfg=cfg, profile="smoke", oof_folds=1, seed=7)

    # Splits: train + val only, no prior, and never `test_strict`.
    assert [call["split"] for call in wiring.calls["build_worlds"]] == [
        tc.TRAIN_SPLIT, tc.VAL_SPLIT]
    assert all(call["laya_regime"] is None for call in wiring.calls["build_worlds"])

    # Variant: too few calibration rows for a trustworthy calibrator -> B5-RAW, named so.
    assert summary["variant"] == "B5-RAW" and summary["is_calibrated"] is False
    assert summary["score"] == "reader_null_margin"
    assert summary["calibration"]["plan"] == "raw" and summary["calibration"]["mode"] == "none"
    assert "100" in summary["calibration"]["reason"]

    # One fold reader, trained on the reduced train set, written under the fold directory.
    assert len(wiring.calls["folds"]) == 1
    fold = wiring.calls["folds"][0]
    assert fold["fold_index"] == 0 and fold["seed"] == 7
    assert fold["out_dir"] == str(tc.estimator_dir(cfg) / tc.OOF_READER_DIR / "fold_0")
    assert set(fold["reader_train_qids"]).isdisjoint(fold["feature_qids"])
    assert set(fold["reader_train_qids"]) | set(fold["feature_qids"]) == set(wiring.train_qids)

    # tau comes from the pre-registered grid: 1.0 answers the wrong span at the bootstrap
    # doc, 5.0 searches first and answers the gold span; ties go to the higher tau.
    assert summary["tau_answer"] == 5.0
    selected = summary["threshold_selection"]["selected"]
    assert selected["coverage"] == 1.0 and selected["selective_risk"] == 0.0
    assert selected["n_episodes"] == summary["splits"]["val_model_selection_questions"]
    assert summary["splits"]["val_model_selection_questions"] == 1  # 20% of six title groups
    assert summary["selection_ci"]["n_groups"] == 1
    assert summary["threshold_selection"]["selected_is_never_answer"] is False
    threshold_file = read_json(tc.estimator_dir(cfg) / tc.THRESHOLD_FILE)
    assert threshold_file["criterion"] == tc.PRE_REGISTERED_CRITERION
    assert threshold_file["split"] == "val_model_selection"

    # Reader optimism: the scaler is the out-of-fold one, not the deployed reader's.
    loaded = C.CorrectnessEstimator.load(tc.estimator_dir(cfg))
    fold_rows = _rows_for(cfg, tc.TRAIN_SPLIT, fold["feature_qids"], margins=FOLD_MARGINS,
                          **wiring.kwargs)
    # root + one state per search action: both searches reveal the other doc, then nothing
    # is left unseen, and every candidate-bearing state has a label.
    assert len(fold_rows) == 3 * len(fold["feature_qids"])
    assert {row["y"] for row in fold_rows} == {0, 1}
    expected = FeatureScaler.fit(tc.feature_matrix(fold_rows))
    assert np.allclose(loaded.scaler.mean, expected.mean, rtol=1e-10, atol=1e-12)
    assert np.allclose(loaded.scaler.std, expected.std, rtol=1e-10, atol=1e-12)
    deployed_rows = _rows_for(cfg, tc.TRAIN_SPLIT, fold["feature_qids"], margins=TRAIN_MARGINS,
                              **wiring.kwargs)
    assert not np.allclose(expected.mean,
                           FeatureScaler.fit(tc.feature_matrix(deployed_rows)).mean)
    assert loaded.fitted and not loaded.is_calibrated
    assert loaded.probability_from_obs({"candidate": {"null_margin": 3.5}}) == 3.5

    # Artifacts + provenance.
    out = tc.estimator_dir(cfg)
    for name in ("model.json", "scaler.json", "calibrator.json", tc.PROVENANCE_FILE,
                 tc.SPLIT_MANIFEST_FILE, tc.THRESHOLD_FILE):
        assert (out / name).is_file(), name
    provenance = read_json(out / tc.PROVENANCE_FILE)
    assert provenance["features"] == list(C.CORRECTNESS_FEATURES)
    assert provenance["seed"] == 7 and provenance["stage"] == tc.STAGE
    assert (provenance["reader"]["deployed"]["reader_model_hash"]
            == wiring.sources[tc.VAL_SPLIT].model_hash)
    assert len(provenance["training"]["train_qid_digest"]) == 64
    assert provenance["library_versions"] and provenance["env_config"]["max_optional_search"] == 2
    manifest = read_json(out / tc.SPLIT_MANIFEST_FILE)
    cal_qids, sel_qids = manifest["val_calibration_qids"], manifest["val_model_selection_qids"]
    assert set(cal_qids).isdisjoint(sel_qids)
    assert set(cal_qids) | set(sel_qids) == set(wiring.val_qids)
    assert manifest["folds"][0]["feature_qids"] == fold["feature_qids"]

    report = Path(summary["report"]).read_text(encoding="utf-8")
    assert "B5-RAW has no calibrated probability" in report
    assert tc.PRE_REGISTERED_CRITERION in report
    assert "test_strict" in report  # "was not read at any point"
    assert Path(summary["plot"]).is_file() and Path(summary["plot"]).stat().st_size > 0

    row = summary["results_row"]
    assert row["mode"] == "results_b5_row_json" and Path(row["path"]).is_file()
    assert read_json(Path(row["path"]))["row"]["variant"] == "B5-RAW"
    assert not (resolve(cfg, "reports") / tc.RESULTS_CSV_FILE).exists()

    registry = _registry(cfg)
    assert [entry["status"] for entry in registry] == ["RUNNING", "SUCCEEDED"]
    assert registry[0]["stage"] == tc.STAGE and registry[0]["oof_folds"] == 1
    artifacts = registry[-1]["artifacts"]
    assert artifacts[str(out / "model.json")]["bytes"] > 0
    assert artifacts[summary["report"]]["sha256"]  # the report lives under reports/


def test_oof_folds_three_cross_fits_every_train_question(tmp_path, monkeypatch):
    # `--oof-folds 3` is the full cross-fitting option of spec 08 §3: every train question is
    # a feature question in exactly one fold, and that fold's reader never saw it.
    wiring = _install(tmp_path, monkeypatch)
    cfg = wiring.cfg
    summary = tc.run(cfg=cfg, profile="smoke", oof_folds=3, seed=7)

    fold_calls = wiring.calls["folds"]
    assert [call["fold_index"] for call in fold_calls] == [0, 1, 2]
    feature_sets = [set(call["feature_qids"]) for call in fold_calls]
    assert all(feature_sets[i].isdisjoint(feature_sets[j])
               for i in range(len(feature_sets)) for j in range(i + 1, len(feature_sets)))
    assert set().union(*feature_sets) == set(wiring.train_qids)
    for call in fold_calls:
        assert set(call["reader_train_qids"]).isdisjoint(call["feature_qids"])
        assert set(call["reader_train_qids"]) | set(call["feature_qids"]) == set(wiring.train_qids)

    assert summary["oof_reader"]["n_folds"] == 3
    assert summary["training"]["oof_folds"] == 3
    manifest = read_json(tc.estimator_dir(cfg) / tc.SPLIT_MANIFEST_FILE)
    assert [fold["index"] for fold in manifest["folds"]] == [0, 1, 2]

    # The scaler consumed the union of all fold rows — every row out-of-fold.
    oof_rows = [row for call in fold_calls
                for row in _rows_for(cfg, tc.TRAIN_SPLIT, call["feature_qids"],
                                     margins=FOLD_MARGINS, **wiring.kwargs)]
    assert {row["y"] for row in oof_rows} == {0, 1}
    expected = FeatureScaler.fit(tc.feature_matrix(oof_rows))
    loaded = C.CorrectnessEstimator.load(tc.estimator_dir(cfg))
    assert np.allclose(loaded.scaler.mean, expected.mean, rtol=1e-10, atol=1e-12)
    assert np.allclose(loaded.scaler.std, expected.std, rtol=1e-10, atol=1e-12)
    assert loaded.fitted and not loaded.is_calibrated


def test_run_end_to_end_calibrates_on_val_only(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "MIN_CAL_N", 1)
    monkeypatch.setattr(C, "MIN_CAL_CLASS", 1)
    wiring = _install(tmp_path, monkeypatch)
    cfg = wiring.cfg
    reports = resolve(cfg, "reports")
    reports.mkdir(parents=True, exist_ok=True)
    (reports / tc.RESULTS_CSV_FILE).write_text("baseline,variant,seed,tau_answer\n",
                                               encoding="utf-8")

    summary = tc.run(cfg=cfg, profile="smoke", oof_folds=1, seed=7)

    assert summary["variant"] == "B5" and summary["is_calibrated"] is True
    assert summary["score"] == "p_correct"
    assert summary["calibration"]["plan"] == "platt"
    assert summary["calibration"]["mode"] == "platt"
    assert summary["selection_rows"] > 0
    selected = summary["threshold_selection"]["selected"]
    assert selected["coverage"] == 1.0 and selected["selective_risk"] == 0.0

    loaded = C.CorrectnessEstimator.load(tc.estimator_dir(cfg))
    assert loaded.is_calibrated and loaded.calibrator.mode == "platt" and loaded.fitted
    assert loaded.variant == "B5"

    # The scaler is fitted on the OOF train rows, the calibrator on val_calibration only:
    # a scaler fitted on the val rows (or on the deployed reader) is a different scaler.
    manifest = read_json(tc.estimator_dir(cfg) / tc.SPLIT_MANIFEST_FILE)
    calibration_rows = _rows_for(cfg, tc.VAL_SPLIT, manifest["val_calibration_qids"],
                                 margins=VAL_MARGINS, **wiring.kwargs)
    assert {row["y"] for row in calibration_rows} == {0, 1}
    assert not np.allclose(loaded.scaler.mean,
                           FeatureScaler.fit(tc.feature_matrix(calibration_rows)).mean)

    # Reliability on the held-out selection rows, and the results row appended in place.
    reliability = summary["threshold_selection"]
    assert reliability["selected_is_never_answer"] is False
    report = Path(summary["report"]).read_text(encoding="utf-8")
    assert "Brier:" in report and "ECE" in report
    assert "risk at 80% coverage" in report
    assert summary["results_row"]["mode"] == "appended_to_results_csv"
    with open(reports / tc.RESULTS_CSV_FILE, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["baseline"] == "B5" and rows[0]["variant"] == "B5" and rows[0]["seed"] == "7"
    # CSV cells round floats to 6 decimals (reporting.format_cell); the exact value is in
    # the JSON row / threshold file.
    assert float(rows[0]["tau_answer"]) == pytest.approx(summary["tau_answer"], abs=1e-6)
    assert not (reports / tc.RESULTS_ROW_FILE).exists()


def test_single_class_oof_rows_fall_back_to_raw_without_pretending(tmp_path, monkeypatch):
    # Every reachable state answers correctly: the OOF rows are single-class, so no logistic
    # model can be fit, and the run must still deploy B5-RAW honestly.
    wiring = _install(tmp_path, monkeypatch,
                      world_kwargs={"root": ("doc_b", GOLD), "reveal": "doc_a",
                                    "post": ("doc_b", GOLD)})
    summary = tc.run(cfg=wiring.cfg, profile="smoke", oof_folds=1, seed=7)

    assert summary["variant"] == "B5-RAW" and summary["is_calibrated"] is False
    assert summary["calibration"]["estimator_fitted"] is False
    assert "single-class" in summary["calibration"]["reason"]
    assert summary["training"]["n_negative"] == 0
    assert summary["tau_answer"] == 1.0  # every state already answers correctly: answer now
    assert summary["threshold_selection"]["selected"]["selective_risk"] == 0.0

    loaded = C.CorrectnessEstimator.load(tc.estimator_dir(wiring.cfg))
    assert loaded.fitted is False and not loaded.is_calibrated
    assert loaded.probability_from_obs({"candidate": {"null_margin": 2.5}}) == 2.5
    with pytest.raises(ValueError, match="null_margin"):
        loaded.predict(np.zeros((1, len(C.CORRECTNESS_FEATURES))))
    report = Path(summary["report"]).read_text(encoding="utf-8")
    assert "B5-RAW has no calibrated probability" in report
