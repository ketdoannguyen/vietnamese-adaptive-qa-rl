"""Shared guards reject label-bearing fields in public model inputs."""

from __future__ import annotations

from src.data.schemas import FORBIDDEN_IN_OBSERVATION, find_forbidden_keys


def test_forbidden_key_guard_covers_every_label_name():
    """The guard set is the contract: every label-bearing name stays in it."""
    assert FORBIDDEN_IN_OBSERVATION == frozenset({
        "answers", "answer_texts", "answer_starts", "is_impossible",
        "plausible_answers", "gold_answer", "gold_doc_id", "reward",
        "expected_action", "oracle_q", "teacher_action", "gold_distribution",
    })


def test_find_forbidden_keys_scans_nested_containers():
    assert find_forbidden_keys({"is_impossible": True}) == ["is_impossible"]
    assert find_forbidden_keys({"candidate": {"reward": 0.5}}) == ["candidate.reward"]
    assert find_forbidden_keys({"docs": [{"gold_answer": "x"}]}) == ["docs[0].gold_answer"]
    assert find_forbidden_keys(("public", {"teacher_action": "ANSWER"})) == ["[1].teacher_action"]
    clean = {"qid": "q1", "question": "Thủ đô của Việt Nam là gì?", "candidate": {"text": "Hà Nội"}}
    assert find_forbidden_keys(clean) == []
