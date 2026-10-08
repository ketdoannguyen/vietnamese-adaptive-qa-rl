"""Reward contract tests (`01_SPECIFICATIONS/06_REWARD_ORACLE_AND_CALIBRATION.md`)."""

from __future__ import annotations

import pytest

from src.qa.metrics import answer_is_correct, exact_match, f1_score, normalize_answer
from src.rl.reward import EpisodePrivate, RewardCalculator, RewardConfig


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


def test_correctness_threshold_is_f1_080():
    assert answer_is_correct("Phạm Văn Đồng", ["Phạm Văn Đồng"], 0.80)
    assert not answer_is_correct("hoàn toàn khác", ["Phạm Văn Đồng"], 0.80)


def _private(impossible: bool, answers=()) -> EpisodePrivate:
    return EpisodePrivate(qid="q", answer_texts=tuple(answers),
                          is_impossible=impossible, gold_doc_id="d")


def test_terminal_utilities_match_the_pre_registered_spec():
    calc = RewardCalculator(RewardConfig())
    correct = calc.terminal_utility(action="ANSWER", candidate_text="1976",
                                    provenance_ok=True, private=_private(False, ["1976"]),
                                    n_bm25=1, n_dense=0)
    assert correct.terminal_utility == pytest.approx(2.0)
    assert correct.total == pytest.approx(2.0 - 0.10)

    wrong = calc.terminal_utility(action="ANSWER", candidate_text="1945",
                                  provenance_ok=True, private=_private(False, ["1976"]),
                                  n_bm25=1, n_dense=0)
    assert wrong.terminal_utility == pytest.approx(-2.0)

    good_abstain = calc.terminal_utility(action="ABSTAIN", candidate_text=None,
                                         provenance_ok=False, private=_private(True),
                                         n_bm25=1, n_dense=0)
    assert good_abstain.terminal_utility == pytest.approx(0.8)

    bad_abstain = calc.terminal_utility(action="ABSTAIN", candidate_text=None,
                                        provenance_ok=False, private=_private(False, ["1976"]),
                                        n_bm25=1, n_dense=0)
    assert bad_abstain.terminal_utility == pytest.approx(-1.0)


def test_provenance_failure_forces_wrong_answer():
    calc = RewardCalculator(RewardConfig(force_provenance=True))
    bd = calc.terminal_utility(action="ANSWER", candidate_text="1976", provenance_ok=False,
                               private=_private(False, ["1976"]), n_bm25=1, n_dense=0)
    assert bd.terminal_utility == pytest.approx(-2.0)
    assert "provenance_invalid" in bd.notes


def test_search_costs_are_deducted_per_call():
    calc = RewardCalculator(RewardConfig())
    bd = calc.terminal_utility(action="ANSWER", candidate_text="1976", provenance_ok=True,
                               private=_private(False, ["1976"]), n_bm25=3, n_dense=2)
    assert bd.search_cost_total == pytest.approx(3 * 0.10 + 2 * 0.25)
    assert bd.total == pytest.approx(2.0 - (3 * 0.10 + 2 * 0.25))


def test_cost_sensitivity_changes_the_comparison():
    """Spec 06 §7: dense cost is an ablation knob and must genuinely change returns."""
    cheap = RewardCalculator(RewardConfig(dense_cost=0.10))
    dear = RewardCalculator(RewardConfig(dense_cost=0.50))
    args = dict(action="ANSWER", candidate_text="1976", provenance_ok=True,
                private=_private(False, ["1976"]), n_bm25=1, n_dense=1)
    assert cheap.terminal_utility(**args).total == pytest.approx(2.0 - 0.20)
    assert dear.terminal_utility(**args).total == pytest.approx(2.0 - 0.60)


def test_reward_spec_hash_changes_with_configuration():
    assert RewardConfig().spec_hash() == RewardConfig().spec_hash()
    assert RewardConfig().spec_hash() != RewardConfig(dense_cost=0.99).spec_hash()


def test_non_terminal_action_is_rejected():
    calc = RewardCalculator(RewardConfig())
    with pytest.raises(ValueError):
        calc.terminal_utility(action="SEARCH_BM25", candidate_text=None, provenance_ok=False,
                              private=_private(True), n_bm25=1, n_dense=0)


def test_unreachable_gold_is_flagged_but_does_not_change_the_penalty():
    """Spec 06 §1: unreachable gold is reported for fairness, not silently forgiven."""
    calc = RewardCalculator(RewardConfig())
    reachable = _private(False, ["1976"])
    unreachable = EpisodePrivate(qid="q", answer_texts=("1976",), is_impossible=False,
                                 gold_doc_id="d", gold_reachable=False)
    a = calc.terminal_utility(action="ABSTAIN", candidate_text=None, provenance_ok=False,
                              private=reachable, n_bm25=1, n_dense=0)
    b = calc.terminal_utility(action="ABSTAIN", candidate_text=None, provenance_ok=False,
                              private=unreachable, n_bm25=1, n_dense=0)
    assert a.terminal_utility == b.terminal_utility == pytest.approx(-1.0)
    assert "gold_unreachable" in b.notes and "gold_unreachable" not in a.notes
