"""Atomic JSON writes preserve the previous artifact if replacement fails."""

from __future__ import annotations

import json
import os

import pytest

from src.utils.io import atomic_write_json


def test_failed_json_rewrite_preserves_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "artifact.json"
    atomic_write_json(path, {"qid": "first"})
    original = path.read_bytes()

    def fail_replace(source, target):
        raise OSError("interrupted replace")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted replace"):
        atomic_write_json(path, {"qid": "replacement"})

    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_json_rewrite_replaces_previous_file(tmp_path):
    path = tmp_path / "nested" / "artifact.json"
    atomic_write_json(path, {"qid": "first"})
    atomic_write_json(path, {"question": "Thủ đô của Việt Nam là gì?"})

    text = path.read_text(encoding="utf-8")
    assert json.loads(text) == {"question": "Thủ đô của Việt Nam là gì?"}
    assert "Thủ đô" in text
    assert list(path.parent.iterdir()) == [path]
