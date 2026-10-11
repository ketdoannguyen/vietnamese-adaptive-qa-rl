"""VixSTORY free-form episodic QA environment — an *isolated* adapter.

This is the VixSTORY counterpart of :mod:`src.rl.env`. It keeps the same four-action
MDP shape so :func:`src.rl.reinforce.rollout` and :mod:`src.eval.policies` can drive it
unchanged, and differs from ViQuAD exactly where ViQuAD's *extractive, single-doc* contract
does not hold (plan ``.claude/plans/vixstory-pilot-300-e01-e07.md`` §3):

    action 0 ANSWER | 1 SEARCH_BM25 | 2 SEARCH_DENSE | 3 ABSTAIN

* **Free-form reader.** A frozen reader returns ``(text, cited_ids, citation_valid)`` per
  ordered visible-document tuple. There is no ``null_score``/``non_null_score`` and no span
  offset, so the observation carries none of them: inventing a null margin would fabricate
  calibration that was never measured. The mask/policy interface
  (:func:`src.eval.policies.legal_actions`, :func:`src.eval.policies.choose_tool`) works
  unchanged. The stock ``src.rl.features.FeatureExtractor`` *requires* the two score fields,
  so this module instead ships :class:`VixStoryFeatureExtractor` — the canonical on-policy
  adapter the plan (§3) permits — which derives features from visible state alone and feeds
  the *unchanged* :func:`src.rl.reinforce.rollout` / :func:`src.rl.reinforce.reinforce_loss`
  loop without fabricating any reader score. It subclasses the base extractor only to reuse
  the shared ``dim`` / ``matrix`` helpers; every feature it emits comes from VixSTORY state.
* **ID validity is not entailment.** ``citation_valid`` only says the cited IDs are unique
  and drawn from the visible passages; it never claims the answer follows from them.
* **Every VixSTORY question is answerable**, so ``ABSTAIN`` scores ``0.0`` — a real floor,
  not a fabricated negative.
* **Reward**, pre-registered in ``scripts/vixstory_reader_eval.py`` and frozen by config::

      R_ANSWER  = 1.0 if token-F1 >= f1_threshold (and citation_gate => valid IDs) else 0.0
      R_ABSTAIN = 0.0
      R_SEARCH  = -search_cost per optional SEARCH (0.05), charged exactly once per action
      G         = R_terminal - search_cost * n_optional_search   (the BM25 bootstrap is free)

* **No inference.** The reader is a lookup into a precomputed cache keyed by
  ``(qid, ordered visible doc ids)`` bound to the reader/tokenizer/inference fingerprints;
  a miss raises instead of inventing a candidate.
* The gold free-form answer and the gold context IDs live only in the private world and are
  read only by :meth:`VixStoryEnv.private_outcome`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import gymnasium as gym
import numpy as np

from src.qa.metrics import exact_match, f1_score, tokenize
from src.qa.reader import reader_cache_key
from src.rl.features import FeatureExtractor
from src.utils.hashing import sha256_json

N_ACTIONS = 4
ANSWER, SEARCH_BM25, SEARCH_DENSE, ABSTAIN = 0, 1, 2, 3
ACTIONS = ("ANSWER", "SEARCH_BM25", "SEARCH_DENSE", "ABSTAIN")
TOOLS = ("bm25", "dense")
DEFAULT_SEARCH_COST = 0.05


# --------------------------------------------------------------------------- reader
@dataclass(frozen=True)
class VixStoryReaderPrediction:
    """One frozen-reader output for a single ordered visible-document state.

    ``text`` is the free-form answer the reader produced (``None``/blank = it declined to
    answer). ``cited_ids`` are the passage IDs the reader claimed; ``citation_valid`` is the
    provenance check only — the IDs are non-empty, unique and a subset of the visible
    passages. ``citation_valid`` is **not** an entailment verdict.
    """

    qid: str
    text: str | None
    cited_ids: tuple[str, ...] = ()
    citation_valid: bool = False
    error: str | None = None

    @property
    def answer_present(self) -> bool:
        """Whether there is a non-blank answer to submit. This is the ANSWER legality rule."""
        return bool(self.text and self.text.strip())

    def to_dict(self) -> dict[str, Any]:
        return {"qid": self.qid, "text": self.text, "cited_ids": list(self.cited_ids),
                "citation_valid": self.citation_valid, "answer_present": self.answer_present,
                "error": self.error}


NULL_PREDICTION = VixStoryReaderPrediction(qid="", text=None)


class VixStoryReaderCache:
    """In-memory view over precomputed free-form reader predictions.

    Keyed by ``(qid, ordered visible doc ids)`` and pinned to the reader/tokenizer/inference
    fingerprints through :func:`src.qa.reader.reader_cache_key`, so a prediction produced by
    a different reader revision can never be served by accident. A miss is a hard error: the
    environment holds no model and must never fabricate a candidate.
    """

    def __init__(self, *, reader_sha: str, tokenizer_sha: str, inference_config_sha: str,
                 entries: Mapping[str, VixStoryReaderPrediction] | None = None) -> None:
        self.reader_sha = reader_sha
        self.tokenizer_sha = tokenizer_sha
        self.inference_config_sha = inference_config_sha
        self._entries: dict[str, VixStoryReaderPrediction] = dict(entries or {})

    def __len__(self) -> int:
        return len(self._entries)

    def key(self, qid: str, visible_doc_ids: Sequence[str]) -> str:
        return reader_cache_key(qid, list(visible_doc_ids), self.reader_sha,
                                self.tokenizer_sha, self.inference_config_sha)

    def put(self, qid: str, visible_doc_ids: Sequence[str],
            prediction: VixStoryReaderPrediction) -> None:
        self._entries[self.key(qid, visible_doc_ids)] = prediction

    def lookup(self, qid: str, visible_doc_ids: Sequence[str]) -> VixStoryReaderPrediction:
        key = self.key(qid, visible_doc_ids)
        hit = self._entries.get(key)
        if hit is None:
            raise KeyError(
                f"VixSTORY reader cache miss for qid {qid!r} with {len(list(visible_doc_ids))} "
                f"visible doc(s) (key {key[:16]}...); the environment performs no inference and "
                "will not fabricate a candidate")
        return hit

    def reader(self, qid: str) -> Callable[[tuple[str, ...]], VixStoryReaderPrediction]:
        """A state -> prediction callable for one question, backed by this cache."""

        def _reader(seen: tuple[str, ...]) -> VixStoryReaderPrediction:
            return self.lookup(qid, seen)

        return _reader


# ----------------------------------------------------------------------- episode inputs
@dataclass
class VixStoryPrivate:
    """Gold side of one VixSTORY episode. Never serialized into an observation or info."""

    qid: str
    answer_texts: tuple[str, ...]
    gold_context_ids: tuple[str, ...]
    hop_level: int | None = None
    question_type: str | None = None
    gold_in_corpus: bool = True


@dataclass
class VixStoryWorld:
    """Immutable per-episode inputs assembled from the shared corpus, rankings and cache.

    ``corpus`` is the shared 3,462-passage mapping (``context_id -> text``); ``rankings`` are
    the real BM25/Dense orderings over it; ``scores`` are optional real retrieval scores used
    only for the (off-by-default) retrieval preview. ``private`` holds the gold free-form
    answer and the gold context IDs. ``reader`` maps the ordered visible-document tuple to a
    cached prediction.
    """

    qid: str
    question: str
    split: str
    scenario_id: str
    corpus: Mapping[str, str]
    rankings: Mapping[str, Sequence[str]]
    private: VixStoryPrivate
    reader: Callable[[tuple[str, ...]], VixStoryReaderPrediction]
    scores: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    scenario_version: str = "v1"
    #: sha256 of the frozen reader checkpoint whose cached predictions this world serves
    #: (empty when the world was assembled without a pinned reader). Provenance only: it is
    #: never a feature and never enters an observation.
    reader_sha: str = ""

    def __post_init__(self) -> None:
        if not self.qid or not self.question:
            raise ValueError("VixSTORY world requires a non-empty qid and question")
        if not self.private.answer_texts:
            raise ValueError(
                f"{self.qid}: VixSTORY questions are answerable; private.answer_texts is empty")
        for tool, ranking in self.rankings.items():
            unknown = [d for d in ranking if d not in self.corpus]
            if unknown:
                raise ValueError(
                    f"{self.qid}: ranking for tool {tool!r} references {len(unknown)} doc(s) "
                    f"outside the shared corpus, e.g. {unknown[0]!r}")
        missing = [d for d in self.private.gold_context_ids if d not in self.corpus]
        if missing:
            raise ValueError(f"{self.qid}: {len(missing)} gold context id(s) missing from the corpus")
        # The corpus is the same for every question; fingerprint it once.
        self._corpus_sha = sha256_json(sorted(self.corpus.items()))

    @property
    def corpus_sha(self) -> str:
        return self._corpus_sha


# --------------------------------------------------------------------------- config
@dataclass(frozen=True)
class VixStoryEnvConfig:
    bootstrap_tool: str = "bm25"
    bootstrap_passages: int = 1
    passage_per_search_action: int = 1
    max_optional_search: int = 2
    #: Cost charged for each optional SEARCH action (0.05), exactly once per action.
    search_cost: float = DEFAULT_SEARCH_COST
    #: Token-F1 threshold for a correct ANSWER (free-form answers are single string spans).
    f1_threshold: float = 0.5
    #: When True, a correct-F1 answer still scores 0 if its citations are not valid IDs.
    #: Fixed by configuration, never tuned on the evaluation split.
    citation_gate: bool = True
    #: Expose the top retrieval score of the next unseen passage. Off by default: the cached
    #: rankings carry ids only, so there is nothing honest to preview (no fabricated score).
    preview_unseen_scores: bool = False
    #: Watchdog only. Reaching it produces truncated=True, never a fake terminal reward.
    max_steps: int = 16
    #: Dev/test guard: raise instead of tolerating an illegal action.
    strict_invalid_actions: bool = True

    def __post_init__(self) -> None:
        if self.bootstrap_tool not in TOOLS:
            raise ValueError(f"unknown bootstrap tool {self.bootstrap_tool!r}; known: {list(TOOLS)}")
        for name in ("bootstrap_passages", "passage_per_search_action", "max_optional_search"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.search_cost < 0.0:
            raise ValueError("search_cost must be non-negative")
        if not 0.0 <= self.f1_threshold <= 1.0:
            raise ValueError("f1_threshold must lie in [0, 1]")
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")


# --------------------------------------------------------------------------- env
class VixStoryEnv(gym.Env):
    """The VixSTORY RL environment. One instance per worker; ``reset`` selects the episode."""

    metadata = {"render_modes": []}

    def __init__(self, worlds: Sequence[VixStoryWorld],
                 config: VixStoryEnvConfig | None = None) -> None:
        super().__init__()
        if not worlds:
            raise ValueError("VixStoryEnv requires at least one VixStoryWorld")
        self._worlds = list(worlds)
        self._by_qid = {w.qid: w for w in self._worlds}
        if len(self._by_qid) != len(self._worlds):
            raise ValueError("duplicate qid in VixSTORY worlds")
        self.config = config or VixStoryEnvConfig()
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.observation_space = gym.spaces.Dict({
            "question": gym.spaces.Text(max_length=4096),
            "action_mask": gym.spaces.MultiBinary(N_ACTIONS),
        })
        self._world: VixStoryWorld | None = None
        self._seen: list[str] = []
        self._optional_used = 0
        self._n_bm25 = 0
        self._n_dense = 0
        self._history: list[int] = []
        self._candidate: VixStoryReaderPrediction = NULL_PREDICTION
        self._t = 0
        self._rng: np.random.Generator = np.random.default_rng(0)
        self._outcome: dict[str, Any] | None = None
        self._terminal_reason: str | None = None

    # ---------------------------------------------------------------- reset
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}
        qid = options.get("qid")
        if qid is not None:
            if qid not in self._by_qid:
                raise KeyError(f"qid {qid!r} is not part of this environment's worlds")
            world = self._by_qid[qid]
        else:
            world = self._worlds[int(self._rng.integers(0, len(self._worlds)))]
        self._world = world

        self._seen = []
        self._optional_used = 0
        self._n_bm25 = 0
        self._n_dense = 0
        self._history = []
        self._candidate = NULL_PREDICTION
        self._t = 0
        self._outcome = None
        self._terminal_reason = None

        # Forced bootstrap: one retrieval before the agent's first decision. It is free here
        # (the pre-registered VixSTORY reward charges only optional SEARCH actions), but it is
        # counted for every policy so the root state is identical across policies.
        if self.config.bootstrap_tool == "bm25":
            self._n_bm25 += 1
        else:
            self._n_dense += 1
        self._seen.extend(self._next_unseen(self.config.bootstrap_tool,
                                            self.config.bootstrap_passages))
        self._refresh_candidate()
        return self._observation(), self._safe_info()

    # ----------------------------------------------------------------- step
    def step(self, action: int):
        if self._world is None:
            raise RuntimeError("step() called before reset()")
        action = int(action)
        if not 0 <= action < N_ACTIONS:
            raise ValueError(f"action {action} outside [0, {N_ACTIONS})")

        mask = self._action_mask()
        if not mask[action]:
            if self.config.strict_invalid_actions:
                raise ValueError(
                    f"invalid action {ACTIONS[action]} at t={self._t} "
                    f"(mask={mask.astype(int).tolist()}); valid={self._valid_action_names()}")
            # Production tolerance path: documented, logged, never silent.
            self._history.append(action)
            self._t += 1
            info = self._safe_info()
            info["invalid_action_penalised"] = True
            return self._observation(), 0.0, False, False, info

        self._history.append(action)
        self._t += 1

        if action == ANSWER:
            return self._terminal(ANSWER)
        if action == ABSTAIN:
            return self._terminal(ABSTAIN)

        # Search: charge the cost once, reveal one previously unseen passage.
        tool = "bm25" if action == SEARCH_BM25 else "dense"
        if tool == "bm25":
            self._n_bm25 += 1
        else:
            self._n_dense += 1
        self._optional_used += 1
        self._seen.extend(self._next_unseen(tool, self.config.passage_per_search_action))
        self._refresh_candidate()

        truncated = self._t >= self.config.max_steps
        info = self._safe_info()
        if truncated:
            self._terminal_reason = "watchdog_max_steps"
            info["truncated_reason"] = "watchdog_max_steps"
            self._outcome = self._make_outcome("ABSTAIN")
        return self._observation(), -self.config.search_cost, False, truncated, info

    # ------------------------------------------------------------- internals
    def _terminal(self, action: int):
        assert self._world is not None
        name = ACTIONS[action]
        utility = self._terminal_utility(action)
        self._outcome = self._make_outcome(name)
        self._terminal_reason = name
        # Search costs were paid on their own steps and the bootstrap is free, so the terminal
        # step carries the utility alone; the episode return is utility - every search cost.
        info = self._safe_info()
        info["terminal_action"] = name
        return self._observation(), utility, True, False, info

    def _terminal_utility(self, action: int) -> float:
        assert self._world is not None
        if action == ABSTAIN:
            return 0.0
        candidate = self._candidate
        if not candidate.answer_present:
            return 0.0
        f1 = f1_score(candidate.text, list(self._world.private.answer_texts))
        if f1 < self.config.f1_threshold - 1e-12:
            return 0.0
        if self.config.citation_gate and not candidate.citation_valid:
            return 0.0
        return 1.0

    def _make_outcome(self, action: str) -> dict[str, Any]:
        assert self._world is not None
        gold = self._world.private
        candidate = self._candidate
        f1 = em = 0.0
        if action == "ANSWER" and candidate.answer_present:
            f1 = f1_score(candidate.text, list(gold.answer_texts))
            em = exact_match(candidate.text, list(gold.answer_texts))
        utility = self._terminal_utility(ANSWER) if action == "ANSWER" else 0.0
        cost = self._optional_used * self.config.search_cost
        gold_set = set(gold.gold_context_ids)
        recall = (len(gold_set & set(self._seen)) / len(gold_set)) if gold_set else float("nan")
        return {
            "qid": self._world.qid,
            "terminal_action": action,
            "correct": bool(action == "ANSWER" and utility >= 1.0),
            "f1": float(f1),
            "em": float(em),
            "citation_valid": bool(candidate.citation_valid),
            "answer_present": bool(candidate.answer_present),
            "n_bm25_searches": self._n_bm25,
            "n_dense_searches": self._n_dense,
            "n_optional_searches": self._optional_used,
            "search_cost_total": float(cost),
            "terminal_utility": float(utility),
            "total": float(utility - cost),
            "gold_context_count": len(gold.gold_context_ids),
            "gold_evidence_recall": float(recall),
        }

    def _next_unseen(self, tool: str, n: int) -> list[str]:
        assert self._world is not None
        ranking = self._world.rankings.get(tool, ())
        seen = set(self._seen)
        out: list[str] = []
        for doc_id in ranking:
            if doc_id not in seen and doc_id in self._world.corpus:
                out.append(doc_id)
                seen.add(doc_id)
                if len(out) >= n:
                    break
        return out

    def _has_unseen(self, tool: str) -> bool:
        assert self._world is not None
        seen = set(self._seen)
        return any(d not in seen for d in self._world.rankings.get(tool, ()))

    def _refresh_candidate(self) -> None:
        assert self._world is not None
        self._candidate = self._world.reader(tuple(self._seen))

    def _action_mask(self) -> np.ndarray:
        assert self._world is not None
        searchable = self._optional_used < self.config.max_optional_search
        mask = np.zeros(N_ACTIONS, dtype=bool)
        # ANSWER is legal whenever the reader produced a non-blank answer. Citation validity is
        # a reward gate, not a legality gate: an uncited answer is submittable and scores 0.
        mask[ANSWER] = self._candidate.answer_present
        mask[SEARCH_BM25] = searchable and self._has_unseen("bm25")
        mask[SEARCH_DENSE] = searchable and self._has_unseen("dense")
        mask[ABSTAIN] = True  # always legal
        return mask

    def _valid_action_names(self) -> list[str]:
        return [ACTIONS[i] for i, ok in enumerate(self._action_mask()) if ok]

    def _retrieval_summary(self) -> dict[str, float]:
        assert self._world is not None

        def top(tool: str) -> tuple[float, float]:
            ranking = list(self._world.rankings.get(tool, ()))
            if not ranking:
                return 0.0, 0.0
            scores = self._world.scores.get(tool)
            if scores:
                vals = [float(scores.get(d, 0.0)) for d in ranking]
                gap = (vals[0] - vals[1]) if len(vals) > 1 else 0.0
                return float(vals[0]), float(gap)
            # Rank-only worlds: expose the rank of the next unseen doc, never a fake score.
            seen = set(self._seen)
            next_unseen = next((i for i, d in enumerate(ranking) if d not in seen), None)
            return (float(-next_unseen) if next_unseen is not None else 0.0), 0.0

        if self.config.preview_unseen_scores:
            bm25_top, bm25_gap = top("bm25")
            dense_top, dense_gap = top("dense")
        else:
            bm25_top = bm25_gap = dense_top = dense_gap = 0.0
        return {"bm25_top_score": bm25_top, "bm25_top_gap": bm25_gap,
                "dense_top_score": dense_top, "dense_top_gap": dense_gap}

    def _observation(self) -> dict[str, Any]:
        assert self._world is not None
        cand = self._candidate
        return {
            "qid": self._world.qid,
            "question": self._world.question,
            "seen_doc_ids": tuple(self._seen),
            "seen_doc_texts": tuple(self._world.corpus[d] for d in self._seen),
            "candidate": {
                "text": cand.text,
                "cited_ids": tuple(cand.cited_ids),
                "citation_valid": bool(cand.citation_valid),
                "answer_present": bool(cand.answer_present),
            },
            "retrieval": self._retrieval_summary(),
            "searches_used": self._n_bm25 + self._n_dense,
            "optional_used": self._optional_used,
            "remaining_optional_searches": self.config.max_optional_search - self._optional_used,
            "last_action": int(self._history[-1]) if self._history else -1,
            "history": tuple(int(a) for a in self._history),
            "timestep": self._t,
            "action_mask": self._action_mask(),
        }

    def _safe_info(self) -> dict[str, Any]:
        """Training-visible metadata. No gold answer, no gold IDs, no hop, no oracle action."""
        assert self._world is not None
        return {
            "qid": self._world.qid,
            "split": self._world.split,
            "scenario_id": self._world.scenario_id,
            "timestep": self._t,
            "state_hash": self.state_hash(),
        }

    # --------------------------------------------------------------- public
    def state_hash(self) -> str:
        assert self._world is not None
        return sha256_json({
            "qid": self._world.qid,
            "scenario_id": self._world.scenario_id,
            "seen_doc_ids": self._seen,
            "optional_used": self._optional_used,
            "history": self._history,
            "state_version": self._world.scenario_version,
        })

    def private_outcome(self) -> dict[str, Any] | None:
        """Terminal metrics for diagnostics. **Never** wire this into policy features."""
        if self._outcome is None:
            return None
        out = dict(self._outcome)
        out["terminal_reason"] = self._terminal_reason
        return out

    def trace_row(self) -> dict[str, Any]:
        """Canonical per-step trace row — safe fields only."""
        assert self._world is not None
        return {
            "qid": self._world.qid,
            "scenario_id": self._world.scenario_id,
            "corpus_sha": self._world.corpus_sha,
            "timestep": self._t,
            "state_hash": self.state_hash(),
            "visible_doc_ids": list(self._seen),
            "candidate": self._candidate.to_dict(),
            "action_mask": self._action_mask().astype(int).tolist(),
            "last_action": int(self._history[-1]) if self._history else None,
        }

    @property
    def candidate(self) -> VixStoryReaderPrediction:
        return self._candidate

    @property
    def seen_doc_ids(self) -> tuple[str, ...]:
        return tuple(self._seen)

    @property
    def n_bm25(self) -> int:
        return self._n_bm25

    @property
    def n_dense(self) -> int:
        return self._n_dense


# ----------------------------------------------------------------------- features
_VIX_FEATURE_NAMES = (
    "candidate_answer_present", "candidate_citation_valid",
    "candidate_token_len", "candidate_char_len",
    "bm25_top_score", "bm25_top_gap", "dense_top_score", "dense_top_gap",
    "seen_count", "optional_used", "remaining_optional", "searches_used", "timestep",
    "last_action_ANSWER", "last_action_SEARCH_BM25", "last_action_SEARCH_DENSE",
    "last_action_ABSTAIN",
    "mask_answer", "mask_bm25", "mask_dense", "mask_abstain",
    "question_token_len", "question_char_len",
    "laya_p_answer", "laya_p_bm25", "laya_p_dense", "laya_p_abstain", "laya_missing",
)


def vixstory_feature_names() -> list[str]:
    """Raw feature names of the canonical VixSTORY observation. No extractive score appears."""
    return list(_VIX_FEATURE_NAMES)


class VixStoryFeatureExtractor(FeatureExtractor):
    """The canonical on-policy feature provider for the VixSTORY REINFORCE loop (plan §3).

    It reuses the base :class:`src.rl.features.FeatureExtractor` only for the shared
    ``dim`` / ``matrix`` helpers, and overrides the feature vector: the stock base reads the
    *extractive* reader's ``non_null_score`` / ``null_margin``, which a free-form reader never
    produces, and this adapter refuses to invent them. ``dim`` / ``vector(obs, laya_probs)`` /
    ``matrix`` are the interface :func:`src.rl.reinforce.rollout` and
    :class:`src.eval.policies.RLPolicy` call, so those classes drive :class:`VixStoryEnv`
    unchanged, using **visible state only**: the candidate's presence/citation flag, real
    retrieval preview scores (when configured), the search budget and the mask. No gold answer,
    id, hop or oracle ever enters a feature.
    """

    def __init__(self) -> None:
        super().__init__()
        self.names = vixstory_feature_names()

    def vector(self, obs: Mapping[str, Any],
               laya_probs: Sequence[float] | None = None) -> np.ndarray:
        if "action_mask" not in obs:
            raise ValueError("VixSTORY observation is missing 'action_mask'")
        mask = np.asarray(obs["action_mask"], dtype=bool)
        if mask.shape != (N_ACTIONS,):
            raise ValueError(f"action_mask must have shape ({N_ACTIONS},); got {mask.shape}")
        candidate = obs.get("candidate", {})
        retrieval = obs.get("retrieval", {})
        text = candidate.get("text")
        cand_tokens = tokenize(text) if text else []
        if laya_probs is None:
            p = [0.0, 0.0, 0.0, 0.0]
            missing = 1.0
        else:
            p = [float(x) for x in laya_probs]
            if len(p) != N_ACTIONS:
                raise ValueError(
                    f"VixSTORY Laya prior must be a Sequence[{N_ACTIONS}]; got length {len(p)}")
            if not all(np.isfinite(x) and x >= 0.0 for x in p):
                raise ValueError(f"VixSTORY Laya prior must be finite and non-negative: {p}")
            missing = 0.0
        last_action = int(obs.get("last_action", -1))
        last_onehot = [1.0 if last_action == i else 0.0 for i in range(N_ACTIONS)]
        question = str(obs.get("question", ""))
        q_tokens = tokenize(question)

        def preview(key: str) -> float:
            value = float(retrieval.get(key) or 0.0)
            return value if np.isfinite(value) else 0.0

        vec = [
            1.0 if candidate.get("answer_present") else 0.0,
            1.0 if candidate.get("citation_valid") else 0.0,
            float(len(cand_tokens)),
            float(len(text)) if text else 0.0,
            preview("bm25_top_score"), preview("bm25_top_gap"),
            preview("dense_top_score"), preview("dense_top_gap"),
            float(len(obs.get("seen_doc_ids", ()))),
            float(obs.get("optional_used", 0)),
            float(obs.get("remaining_optional_searches", 0)),
            float(obs.get("searches_used", 0)),
            float(obs.get("timestep", 0)),
            *last_onehot,
            *[1.0 if m else 0.0 for m in mask],
            float(len(q_tokens)),
            float(len(question)),
            *p,
            missing,
        ]
        out = np.asarray(vec, dtype=np.float32)
        if out.shape[0] != self.dim:
            raise AssertionError(f"feature width {out.shape[0]} != declared {self.dim}")
        return out
