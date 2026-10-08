"""Core data contracts.

The single most important rule in this file: **label-bearing fields never leave
the private side of a record.** ``QuestionRecord.private`` holds gold answers and
answerability; :meth:`QuestionRecord.public_view` is the only thing that may be
serialized into an observation, a Laya state string, or a policy feature vector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Fields that must never appear inside a visible state / observation / prompt.
FORBIDDEN_IN_OBSERVATION = frozenset({
    "answers", "answer_texts", "answer_starts", "is_impossible",
    "plausible_answers", "gold_answer", "gold_doc_id", "reward",
    "expected_action", "oracle_q", "teacher_action", "gold_distribution",
})


@dataclass(frozen=True)
class DocumentRecord:
    """A retrievable passage. Content only — carries no labels."""

    doc_id: str            # sha256 of canonically-normalized content
    title_group: str       # normalized group key used for split isolation
    text: str              # NFC-normalized, whitespace-collapsed
    source_split: str      # which raw dataset split it came from

    def to_public_dict(self) -> dict[str, Any]:
        return {"doc_id": self.doc_id, "title_group": self.title_group,
                "text": self.text, "source_split": self.source_split}


@dataclass(frozen=True)
class QuestionPrivate:
    """Gold-side payload. Must never reach a model or a log that a policy reads."""

    answer_texts: tuple[str, ...]
    answer_starts: tuple[int, ...]
    is_impossible: bool
    plausible_texts: tuple[str, ...] = ()
    gold_doc_id: str | None = None
    offset_issue: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer_texts": list(self.answer_texts),
            "answer_starts": list(self.answer_starts),
            "is_impossible": self.is_impossible,
            "plausible_texts": list(self.plausible_texts),
            "gold_doc_id": self.gold_doc_id,
            "offset_issue": self.offset_issue,
        }


@dataclass(frozen=True)
class QuestionRecord:
    qid: str                  # "<source_split>:<raw id>"
    source_split: str
    title_group: str
    question: str
    private: QuestionPrivate
    strict_split: str | None = None   # train_strict | val_strict | test_strict | None

    def public_view(self) -> dict[str, Any]:
        """The ONLY serialization allowed to cross into observations/prompts."""
        return {"qid": self.qid, "question": self.question}

    def to_row(self) -> dict[str, Any]:
        row = {"qid": self.qid, "source_split": self.source_split,
               "title_group": self.title_group, "question": self.question,
               "strict_split": self.strict_split}
        row.update(self.private.to_dict())
        return row


@dataclass(frozen=True)
class ScenarioRecord:
    """One episodic controlled corpus for a question."""

    scenario_id: str
    qid: str
    split: str
    corpus_doc_ids: tuple[str, ...]      # includes the gold context when answerable
    scenario_seed: int
    gold_doc_id: str | None              # private
    reachable_bm25: bool | None = None   # private diagnostic
    reachable_dense: bool | None = None
    difficulty: dict[str, Any] = field(default_factory=dict)

    def public_meta(self) -> dict[str, Any]:
        return {"scenario_id": self.scenario_id, "qid": self.qid, "split": self.split,
                "corpus_size": len(self.corpus_doc_ids), "scenario_seed": self.scenario_seed}

    def to_row(self) -> dict[str, Any]:
        row = self.public_meta()
        row.update({
            "corpus_doc_ids": list(self.corpus_doc_ids),
            "gold_doc_id": self.gold_doc_id,
            "reachable_bm25": self.reachable_bm25,
            "reachable_dense": self.reachable_dense,
            "difficulty": self.difficulty,
        })
        return row


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
