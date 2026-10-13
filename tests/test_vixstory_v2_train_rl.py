"""Pinned V2 RL runner gates, separate from real training and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.vixstory_v2_train_rl import EXPECTED_UPDATES, run, selected_prior


def make_selection(run_dir: Path, arm: str, epoch: int = 2) -> None:
    location = run_dir / "laya" / arm
    (location / f"epoch-{epoch}").mkdir(parents=True)
    (location / f"epoch-{epoch}" / "model.safetensors").write_bytes(b"weights")
    (location / "selection.json").write_text(json.dumps({
        "arm": arm, "selected_epoch": epoch, "frozen_checkpoint_sha256": "pinned-hash",
        "candidates": [{"index": index, "n": 246,
                        "checkpoint_sha256": "pinned-hash" if index == epoch else "other"}
                       for index in (1, 2, 3)],
    }))


def test_selected_prior_pairs_e06_to_e05_and_e07_to_e04(tmp_path):
    for arm, source in (("E06", "E05"), ("E07", "E04")):
        make_selection(tmp_path, source)
        checkpoint, digest = selected_prior(tmp_path, arm)
        assert checkpoint == tmp_path / "laya" / source / "epoch-2"
        assert digest == "pinned-hash"


def test_selected_prior_refuses_hash_or_validation_mismatch(tmp_path):
    make_selection(tmp_path, "E05")
    selection = tmp_path / "laya" / "E05" / "selection.json"
    payload = json.loads(selection.read_text())
    payload["candidates"][1]["n"] = 245
    selection.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="does not match"):
        selected_prior(tmp_path, "E06")


def test_training_refuses_existing_rl_output_before_loading_prior(tmp_path):
    make_selection(tmp_path, "E05")
    (tmp_path / "rl" / "E06").mkdir(parents=True)
    with patch("scripts.vixstory_v2_train_rl.LayaAdapterV2.load") as loader:
        with pytest.raises(FileExistsError):
            run(tmp_path, "E06", "train")
    loader.assert_not_called()


def test_training_checks_full_schedule_and_prior_binding(tmp_path):
    make_selection(tmp_path, "E05")
    adapter = SimpleNamespace(model_hash=lambda: "pinned-hash", close=lambda: None,
                              predict_probs=lambda observations: [[0.5, 0.25, 0.25]])
    worlds = [SimpleNamespace(qid=f"q{index}") for index in range(2553)]
    result = {"identity": "identity", "prior_hash": "pinned-hash",
              "schedule_sha256": "schedule", "n_episodes": 7659, "n_updates": 240,
              "episodes_per_qid": {"min": 3, "max": 3, "distinct_qids": 2553}}
    with (patch("scripts.vixstory_v2_train_rl.LayaAdapterV2.load", return_value=adapter),
          patch("scripts.vixstory_v2_train_rl.load_worlds", return_value=worlds),
          patch("scripts.vixstory_v2_train_rl.VixStoryEnvV2") as environment,
          patch("scripts.vixstory_v2_train_rl.VixStoryV2ReinforceTrainer") as trainer):
        trainer.return_value.train.return_value = result
        assert run(tmp_path, "E06", "train") == result
        assert trainer.call_args.kwargs["prior_hash"] == "pinned-hash"
        assert trainer.call_args.args[1].updates == 240
        assert trainer.call_args.kwargs["run_dir"] == tmp_path / "rl" / "E06"
        environment.assert_called_once_with(worlds)
        assert trainer.return_value.train.call_args.args[0] == [world.qid for world in worlds]
        assert list(trainer.call_args.kwargs["prior"]({"action_mask": [True] * 3})) == [0.5, 0.25, 0.25]


def test_checkpoint_cadence_is_ten_distinct_updates():
    assert EXPECTED_UPDATES == [25, 50, 75, 100, 125, 150, 175, 200, 225, 240]
