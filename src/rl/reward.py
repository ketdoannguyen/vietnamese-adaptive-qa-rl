"""Episode reward.

Spec sources: `01_SPECIFICATIONS/06_REWARD_ORACLE_AND_CALIBRATION.md` §1 and
`01_SPECIFICATIONS/05_MDP_ENVIRONMENT_CONTRACT.md` §2.

    G = terminal_utility - bm25_cost * n_BM25 - dense_cost * n_DENSE

The forced bootstrap retrieval counts for every policy, so it is subtracted from the
episode return exactly like an agent-chosen BM25 search (Gym's ``reset`` cannot return
a reward, so the cost is applied to the single terminal reward). Retrieval costs appear
only here — never in a transition reward that the policy could mistake for a signal
about correctness.

This module is the only place in the codebase allowed to read gold labels for scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.qa.metrics import answer_is_correct, f1_score
from src.utils.hashing import sha256_json

ACTIONS = ("ANSWER", "SEARCH_BM25", "SEARCH_DENSE", "ABSTAIN")
ACTION_INDEX = {name: i for i, name in enumerate(ACTIONS)}


@dataclass(frozen=True)
class RewardConfig:
    correct_answer: float = 2.0
    wrong_answer: float = -2.0
    correct_abstain: float = 0.8
    unnecessary_abstain: float = -1.0
    bm25_cost: float = 0.10
    dense_cost: float = 0.25
    bootstrap_bm25_cost: float = 0.10
    candidate_f1_threshold: float = 0.80
    force_provenance: bool = True
    gamma: float = 1.0

    @classmethod
    def from_dict(cls, cfg: dict) -> "RewardConfig":
        reward = cfg.get("reward", cfg)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in reward.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}

    def spec_hash(self) -> str:
        """Hash of the reward specification. Any change invalidates teacher labels."""
        return sha256_json(self.to_dict())


@dataclass
class EpisodePrivate:
    """Gold side of one episode. Constructed by the environment, never serialized out."""

    qid: str
    answer_texts: tuple[str, ...]
    is_impossible: bool
    gold_doc_id: str | None
    #: True when the gold context is present in this episode's controlled corpus.
    gold_in_corpus: bool = True
    #: Set by the runner when the retriever cannot surface the gold doc within budget.
    #: Diagnostic only — it does not change the reward formula (spec 06 §1 fairness note).
    gold_reachable: bool | None = None


@dataclass
class RewardBreakdown:
    terminal_action: str
    terminal_utility: float
    correctness_f1: float
    correctness_em: float
    n_bm25_searches: int
    n_dense_searches: int
    search_cost_total: float
    total: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "terminal_action": self.terminal_action,
            "terminal_utility": self.terminal_utility,
            "correctness_f1": self.correctness_f1,
            "correctness_em": self.correctness_em,
            "n_bm25_searches": self.n_bm25_searches,
            "n_dense_searches": self.n_dense_searches,
            "search_cost_total": self.search_cost_total,
            "total": self.total,
        }
        if self.notes:
            d["notes"] = list(self.notes)
        return d


class RewardCalculator:
    """Private, gold-aware scoring. The policy never sees anything this class returns."""

    def __init__(self, config: RewardConfig) -> None:
        self.config = config

    # -- costs ---------------------------------------------------------------
    def search_cost(self, action: str) -> float:
        if action == "SEARCH_BM25":
            return self.config.bm25_cost
        if action == "SEARCH_DENSE":
            return self.config.dense_cost
        return 0.0

    def episode_return(self, breakdown: RewardBreakdown) -> float:
        """G = terminal_utility - search costs. Bootstrap is folded into n_bm25."""
        return breakdown.total

    # -- terminal utility ----------------------------------------------------
    def terminal_utility(
        self,
        *,
        action: str,
        candidate_text: str | None,
        provenance_ok: bool,
        private: EpisodePrivate,
        n_bm25: int,
        n_dense: int,
    ) -> RewardBreakdown:
        cfg = self.config
        notes: list[str] = []
        f1 = 0.0
        em = 0.0
        utility = 0.0

        if action == "ANSWER":
            if private.is_impossible:
                # Answering a negative: is the candidate a *plausible* answer? Still wrong.
                utility = cfg.wrong_answer
                notes.append("answered_negative")
            elif candidate_text is None:
                utility = cfg.wrong_answer
                notes.append("answer_without_candidate")
            else:
                f1 = f1_score(candidate_text, list(private.answer_texts))
                em = 1.0 if answer_is_correct(
                    candidate_text, list(private.answer_texts), threshold=1.0) else 0.0
                correct = f1 >= cfg.candidate_f1_threshold - 1e-12
                if cfg.force_provenance and not provenance_ok:
                    correct = False
                    notes.append("provenance_invalid")
                utility = cfg.correct_answer if correct else cfg.wrong_answer
                if not correct:
                    notes.append("incorrect_answer")

        elif action == "ABSTAIN":
            if private.is_impossible:
                utility = cfg.correct_abstain
            else:
                utility = cfg.unnecessary_abstain
                notes.append("abstained_on_answerable")
                if private.gold_reachable is False:
                    notes.append("gold_unreachable")

        else:
            raise ValueError(f"terminal_utility called with non-terminal action {action!r}")

        cost = n_bm25 * cfg.bm25_cost + n_dense * cfg.dense_cost
        return RewardBreakdown(
            terminal_action=action,
            terminal_utility=utility,
            correctness_f1=f1,
            correctness_em=em,
            n_bm25_searches=n_bm25,
            n_dense_searches=n_dense,
            search_cost_total=cost,
            total=utility - cost,
            notes=notes,
        )
