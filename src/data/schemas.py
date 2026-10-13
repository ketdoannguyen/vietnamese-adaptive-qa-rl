"""Shared guards that keep label-bearing fields out of visible model inputs."""

from __future__ import annotations

from typing import Any

#: Fields that must never appear inside a visible state / observation / prompt.
FORBIDDEN_IN_OBSERVATION = frozenset({
    "answers", "answer_texts", "answer_starts", "is_impossible",
    "plausible_answers", "gold_answer", "gold_doc_id", "reward",
    "expected_action", "oracle_q", "teacher_action", "gold_distribution",
})


def find_forbidden_keys(obj: Any, *, _path: str = "") -> list[str]:
    """Recursively scan a nested structure for label-bearing keys.

    Used by tests to assert that observations, info dicts and serialized states
    contain no gold side-channel.
    """
    hits: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in FORBIDDEN_IN_OBSERVATION:
                hits.append(f"{_path}.{key}" if _path else key)
            hits.extend(find_forbidden_keys(value, _path=f"{_path}.{key}" if _path else str(key)))
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            hits.extend(find_forbidden_keys(value, _path=f"{_path}[{i}]"))
    return hits
