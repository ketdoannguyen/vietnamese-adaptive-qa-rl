"""Vietnamese-safe normalization, exact match and token-F1 contracts."""

from __future__ import annotations

import pytest

from src.qa.metrics import exact_match, f1_score, normalize_answer


def test_vietnamese_normalization_keeps_diacritics():
    assert normalize_answer("  Hà   Nội!! ") == "hà nội"
    # Diacritics must survive: these are different words.
    assert normalize_answer("hoà") != normalize_answer("hoa")
    assert normalize_answer("Đà Nẵng.") == "đà nẵng"


def test_exact_match_and_f1():
    assert exact_match("Hà Nội", ["Hà Nội"]) == 1.0
    assert exact_match("hà nội", ["Hà Nội"]) == 1.0
    assert exact_match(None, ["Hà Nội"]) == 0.0
    assert f1_score("Phạm Văn Đồng", ["Phạm Văn Đồng"]) == pytest.approx(1.0)
    assert f1_score("Phạm Văn", ["Phạm Văn Đồng"]) == pytest.approx(2 * (2 / 2) * (2 / 3) / (2 / 2 + 2 / 3))
    assert f1_score("hoàn toàn sai", ["Phạm Văn Đồng"]) == 0.0
    assert f1_score(None, ["x"]) == 0.0


def test_f1_repeated_tokens_are_capped_at_reference_multiplicity():
    # "a a a" vs "a": overlap must be counted once, so precision 1/3, recall 1.
    assert f1_score("a a a", ["a"]) == pytest.approx(2 * (1 / 3) * 1 / (1 / 3 + 1))
