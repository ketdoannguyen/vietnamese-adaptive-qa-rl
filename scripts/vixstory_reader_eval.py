"""VixSTORY reader evaluation and reward-based search-advantage measurement.

Two stages, both train-only (the official test split is never read):

``--stage gold``
    Score the cloud reader on all 500 locked train questions *given their gold passages*.
    This is a diagnostic upper bound on the reader: it says whether the reader can turn
    available evidence into a correct answer at all. Gold passages are supplied to the
    reader only here; they never enter a retrieval ranking or a policy observation.

``--stage states``
    Enumerate the *real* reachable visible-document states of the episodic retrieval tree
    (BM25/Dense rankings over the full corpus, no gold injection), generate one answer per
    state, and compute reader-based Q values. Only this stage can produce
    ``positive_search_advantage_rate``: gold-passage reachability is evidence, not reward.

The reward is pre-registered here and applied identically to every state:

    R_ANSWER  = 1.0 if the answer text token-F1 >= tau (citation-gated by default)
    R_ABSTAIN = 0.0
    SEARCH    = -search_cost, plus the value of the successor state

``R_ANSWER`` combines two independent signals. The *answer* signal scores the returned text
against the reference (token-F1). The *citation* signal only checks that the cited IDs are
unique and were among the passages the reader was shown: a valid citation ID proves the
reader named a passage it saw and nothing more — it does **not** prove that the passage
entails the answer. The two metrics are reported separately, and the reward is reported both
with the citation gate (locked) and without it (sensitivity).

Every VixSTORY question is answerable, so ``R_ABSTAIN = 0`` is a genuine floor rather than
a fabricated negative.

Cloud generation is cached by exact prompt in an envelope bound to the provider model,
endpoint, decoding and corpus/rankings inputs. A legacy or mismatched cache is refused. A
lookup miss on the locked evaluation path raises; only an explicit warm-up makes requests.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import httpx
import pyarrow.parquet as pq

from src.qa.metrics import exact_match, f1_score
from src.retrieval.states import next_unseen, reachable_states
from src.utils.hashing import sha256_file
from src.utils.io import atomic_write_json

REVISION = "19613082c7ef128647d561472d1e150ca878fb22"
BASE_URL = "https://api.ai-box.vn/v1"
MODEL = "ds/deepseek-flash"
MODEL_REVISION = "provider-managed-alias"
GO_BASE_URL = "https://opencode.ai/zen/go/v1"
GO_MODEL = "deepseek-v4.1-flash"
GO_KEY_ENV = "OPENCODE_GO_API_KEY"
MESSAGE_VERSION = "openai-chat-user-v1"
MAX_REQUESTS = 50
# Two additional attempts authorized after three prompts exhausted transport retries.
MAX_ATTEMPTS_PER_PROMPT = 5
SNAP = (Path.home() / ".cache/huggingface/hub/datasets--ura-hcmut--VixSTORY"
        / "snapshots" / REVISION)
CORPUS_PATH = SNAP / "corpus/corpus.parquet"
TRAIN_PATH = SNAP / "data/train.parquet"
SUITABILITY = Path("outputs/experiments/2026-10-11-vixstory-suitability")
DEFAULT_OUT = Path("outputs/experiments/2026-10-11-vixstory-reader-eval")
DEFAULT_RANKINGS = (Path("outputs/experiments/2026-10-11-vixstory-diagnosis")
                    / "train500_rankings_top300.json")
TOOLS = ("bm25", "dense")

SEARCH_COST = 0.05
# The provider counts reasoning against max_tokens; two reachable prompts exhausted 4096
# and 8192, respectively, then completed in isolated probes at 8192 and 16384.
MAX_NEW_TOKENS = 16384
MAX_PROMPT_CHARS = 12_000
# The pilot allows longer natural-retrieval states; both warm-up and world assembly use this
# cap, while the reader checks the actual tokenized model-context budget before generation.
PILOT_MAX_PROMPT_CHARS = 20_000
REWARD_TAU = 0.5

# Bump this whenever ``reader_prompt`` changes; it is part of the cache identity.
PROMPT_VERSION = "v3-complete-concise"
# Envelope format for the generation cache. Legacy files (a bare prompt->response dict) do
# not carry it and are refused.
CACHE_VERSION = 2
# Versioned default filename so the new envelope never clobbers the legacy caches
# (``generation-cache.json`` / ``shared-cache.json``).
DEFAULT_CACHE_NAME = "reader-cache-v2.json"

# Result files a stage writes. They are historical measurements and must never be clobbered by
# a re-run, so the stage refuses to start when any of them already exists.
GOLD_RESULT_NAMES = ("reader-gold-train500.json", "reader-gold-train500-rows.json")


def result_paths(stage: str, out: Path, tag: str = "") -> tuple[Path, ...]:
    """The result files a stage would write, so the overwrite guard can check them up front."""
    if stage == "gold":
        return tuple(out / name for name in GOLD_RESULT_NAMES)
    return (out / f"search-advantage-train500{tag}.json",
            out / f"search-advantage-train500{tag}-rows.json")


def refuse_overwrite(paths: Sequence[Path]) -> None:
    """Fail before any generation when a result file already exists.

    Called at the top of every stage (and by the CLI before the reader is built), so a historical
    measurement is never overwritten and no compute is spent only to fail at the write.
    """
    existing = [str(p) for p in paths if p.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing result file(s): "
                              + ", ".join(existing))


@dataclass(frozen=True)
class CacheIdentity:
    """Every input a cached generation depends on. Any mismatch invalidates the cache."""

    model: str
    model_revision: str
    base_url: str
    prompt_version: str
    decoding: str
    max_new_tokens: int
    corpus_sha256: str
    rankings_sha256: str
    message_version: str

    def as_dict(self) -> dict:
        return asdict(self)


class GenerationCache:
    """Versioned, identity-bound store of ``prompt -> generated response``.

    The on-disk envelope is ``{"cache_version", "identity", "entries"}``. Loading a legacy
    prompt-only dict, an unknown version, or a different identity raises instead of reusing
    generations that were produced by a different reader configuration.
    """

    def __init__(self, path: str | Path, identity: CacheIdentity) -> None:
        self.path = Path(path)
        self.identity = identity
        self.attempts = 0
        self.failures: dict[str, str] = {}
        self.retries: dict[str, int] = {}
        self.provider_observed = False
        self.provider_model: str | None = None
        self.provider_fingerprint: str | None = None
        self.entries: dict[str, str] = self._load()

    def _load(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "cache_version" not in data:
            raise ValueError(
                f"{self.path} is a legacy prompt-only cache; refusing to reuse it. "
                f"Point --cache at a new path (default {DEFAULT_CACHE_NAME}).")
        if data["cache_version"] != CACHE_VERSION:
            raise ValueError(f"{self.path} has cache_version {data['cache_version']!r}, "
                             f"expected {CACHE_VERSION}")
        if data.get("identity") != self.identity.as_dict():
            raise ValueError(f"{self.path} was written under a different reader identity; "
                             f"refusing to reuse it. Stored {data.get('identity')} vs run "
                             f"{self.identity.as_dict()}")
        entries = data.get("entries")
        if not isinstance(entries, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in entries.items()):
            raise ValueError(f"{self.path} has a malformed entries block")
        attempts = data.get("attempts", 0)
        if not isinstance(attempts, int) or attempts < 0:
            raise ValueError(f"{self.path} has an invalid request count")
        self.attempts = attempts
        failures = data.get("failures", {})
        if not isinstance(failures, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in failures.items()):
            raise ValueError(f"{self.path} has malformed failure records")
        self.failures = failures
        retries = data.get("retries", {})
        if not isinstance(retries, dict) or not all(
                isinstance(k, str) and isinstance(v, int) and v >= 0
                for k, v in retries.items()):
            raise ValueError(f"{self.path} has malformed retry counts")
        self.retries = retries
        if entries and "provider_observed" not in data:
            raise ValueError(f"{self.path} has no provider identity metadata")
        self.provider_observed = data.get("provider_observed", False)
        self.provider_model = data.get("provider_model")
        self.provider_fingerprint = data.get("provider_fingerprint")
        if not isinstance(self.provider_observed, bool) or any(
                value is not None and not isinstance(value, str) for value in
                (self.provider_model, self.provider_fingerprint)):
            raise ValueError(f"{self.path} has malformed provider identity")
        return dict(entries)

    def __contains__(self, prompt: str) -> bool:
        return prompt in self.entries

    def __len__(self) -> int:
        return len(self.entries)

    def get(self, prompts: Sequence[str]) -> list[str]:
        """Read-only lookup for the locked path: any miss is an error, never a fresh sample."""
        missing = [p for p in dict.fromkeys(prompts) if p not in self.entries]
        if missing:
            raise KeyError(
                f"{len(missing)} prompt(s) absent from the frozen cache at {self.path}; "
                f"refusing to generate during locked evaluation")
        return [self.entries[p] for p in prompts]

    def put(self, prompt: str, response: str) -> None:
        self.entries[prompt] = response

    def flush(self) -> None:
        atomic_write_json(self.path, {"cache_version": CACHE_VERSION,
                                      "identity": self.identity.as_dict(),
                                      "attempts": self.attempts,
                                      "failures": self.failures,
                                      "retries": self.retries,
                                      "provider_observed": self.provider_observed,
                                      "provider_model": self.provider_model,
                                      "provider_fingerprint": self.provider_fingerprint,
                                      "entries": self.entries})


def prompt_token_budget(model_context: int, max_new_tokens: int) -> int:
    """Input-token budget = context window minus the reserved output tokens."""
    budget = model_context - max_new_tokens
    if budget <= 0:
        raise ValueError(f"max_new_tokens {max_new_tokens} leaves no room in a "
                         f"{model_context}-token context")
    return budget


def assert_prompts_fit(lengths: Sequence[int], *, model_context: int,
                       max_new_tokens: int) -> None:
    """Fail fast when a prompt would have to be truncated; evidence is never silently cut."""
    budget = prompt_token_budget(model_context, max_new_tokens)
    over = [n for n in lengths if n > budget]
    if over:
        raise ValueError(f"prompt of {max(over)} tokens exceeds the {budget}-token input budget "
                         f"(context {model_context} - max_new_tokens {max_new_tokens}); "
                         f"refusing to truncate evidence")


def answer_reward(f1: float, citation_valid: bool, *, gate_citation: bool = True,
                  tau: float = REWARD_TAU) -> float:
    """Pre-registered ANSWER reward, from two independent signals.

    ``f1`` scores the answer *text* against the reference. ``citation_valid`` is a
    provenance check: the cited IDs are unique and were among the passages the reader was
    shown. A valid citation ID only proves the reader named a passage it saw — it does **not**
    prove the passage entails the answer. The locked reward gates on citations;
    ``gate_citation=False`` is the sensitivity arm used to report how much the reward depends
    on that gate.
    """
    if gate_citation and not citation_valid:
        return 0.0
    return 1.0 if f1 >= tau - 1e-12 else 0.0


def reader_prompt(question: str, passages: Sequence[Mapping]) -> str:
    """Prompt built from *visible* passages only; never carries a reference answer."""
    rendered = "\n\n".join(f"[{p['context_id']}] {p['text']}" for p in passages)
    return ("Đọc các đoạn văn bản và trả lời câu hỏi bằng tiếng Việt. Trong answer, nêu "
            "đầy đủ các thực thể, sự kiện, quan hệ và điều kiện được hỏi bằng câu văn tự nhiên; "
            "không chỉ trả lời một cụm từ khi câu hỏi đòi hỏi giải thích hoặc nhiều ý. "
            "Chỉ dùng thông tin trong các đoạn đã cho, không thêm diễn giải hoặc chi tiết "
            "ngoài câu hỏi. Nếu thiếu thông tin, chỉ nêu phần có căn cứ hoặc để answer rỗng. "
            'Trả về đúng JSON: {"answer":"<câu trả lời>","citations":["<ID đoạn đã dùng>"]}. '
            "Không viết gì ngoài JSON.\n"
            f"Câu hỏi: {question}\nĐoạn văn:\n{rendered}")


def _citation_ids(citations: object) -> tuple[str, ...] | None:
    """Usable citation IDs: a single string or a list of strings. Anything else (``null``, a
    number, an object, a mixed list) is not a citation the reader can be credited for."""
    if isinstance(citations, str):
        return (citations,)
    if isinstance(citations, (list, tuple)) and all(isinstance(item, str) for item in citations):
        return tuple(citations)
    return None


def parse_reader_output(raw: str, seen: Sequence[str]) -> tuple[str | None, tuple[str, ...], bool]:
    """Tolerant *single-object* JSON extraction. ``valid`` means the citation *provenance* is
    usable.

    Only the first JSON object is decoded (from the first ``{``), so trailing prose or stray
    braces after a complete object do not defeat the parse. ``valid`` only checks that the cited
    IDs are unique and are IDs the reader was shown; it says nothing about whether a cited
    passage entails the answer. The answer text is returned even when citations are missing,
    ``null`` or not strings, because the returned text is still what a scoring function has to
    judge — dropping it would turn a provenance failure into an invisible answer-rate failure.
    """
    start = raw.find("{")
    if start == -1:
        return None, (), False
    try:
        parsed, _end = json.JSONDecoder().raw_decode(raw, start)
    except json.JSONDecodeError:
        return None, (), False
    if not isinstance(parsed, dict):
        return None, (), False
    cited = _citation_ids(parsed.get("citations", parsed.get("citation", [])))
    valid = (cited is not None and bool(cited)
             and len(set(cited)) == len(cited) and set(cited) <= set(seen))
    if cited is None:
        cited = ()
    text = parsed.get("answer")
    if not isinstance(text, str) or not text.strip():
        return None, cited, False
    return text.strip(), cited, valid


def select_scored_rows(rows: Sequence[Mapping], docs: Mapping[str, Mapping], *,
                       max_prompt_passage_chars: int = MAX_PROMPT_CHARS
                       ) -> tuple[list[Mapping], list[str]]:
    """Split rows into scorable / over-budget. Over-budget rows are reported, not truncated."""
    kept, excluded = [], []
    for row in rows:
        total = sum(len(docs[doc_id]["text"]) for doc_id in row["context_ids"])
        (kept if total <= max_prompt_passage_chars else excluded).append(row)
    return kept, [row["id"] for row in excluded]


def search_advantage(branch_q: Mapping[str, float], terminal_q: Mapping[str, float], *,
                     search_cost: float) -> dict:
    """Compare the best search branch against the best terminal action.

    Branch Q values already include search cost, so ``search_cost`` is not re-applied here; it
    is part of the signature so call sites and the regression test state the "cost applied
    exactly once" contract explicitly. A tie is not a reason to search, so terminal actions win
    ties.
    """
    best_terminal = max(terminal_q.values())
    best_search = max((value for name, value in branch_q.items() if name.startswith("SEARCH_")),
                      default=float("-inf"))
    pool = dict(branch_q) | dict(terminal_q)
    optimal = max(pool, key=lambda name: (pool[name], not name.startswith("SEARCH_"), name))
    return {"best_terminal": best_terminal, "best_search": best_search,
            "advantage": best_search - best_terminal,
            "positive": best_search > best_terminal,
            "optimal": optimal}


def q_values(state: tuple[str, ...], rankings: Mapping[str, Sequence[str]],
             score: "callable", *, remaining: int, search_cost: float,
             passage_per_search_action: int = 1) -> tuple[float, dict[str, float]]:
    """Recursive state-action values over the reachable tree, from the frozen reader.

    Each SEARCH reveals ``passage_per_search_action`` new passages, matching the budget used by
    :func:`reachable_states` to build the state set — otherwise the recursion would visit states
    that were never pre-generated and the locked cache read would fail.
    """
    answer, abstain = score(state)
    q = {"ANSWER": answer, "ABSTAIN": abstain}
    if remaining:
        for tool in TOOLS:
            addition = next_unseen(rankings[tool], state, passage_per_search_action)
            if not addition:
                continue
            future, _ = q_values(state + tuple(addition), rankings, score,
                                 remaining=remaining - 1, search_cost=search_cost,
                                 passage_per_search_action=passage_per_search_action)
            q[f"SEARCH_{tool.upper()}"] = future - search_cost
    return max(q.values()), q


def _load() -> tuple[dict[str, Mapping], list[Mapping]]:
    docs = {r["context_id"]: r for r in pq.read_table(CORPUS_PATH).to_pylist()}
    train = {r["id"]: r for r in pq.read_table(TRAIN_PATH).to_pylist()}
    qids = json.loads((SUITABILITY / "train_sample_qids.json").read_text(encoding="utf-8"))
    return docs, [train[qid] for qid in qids]


@dataclass(frozen=True)
class CloudAttempt:
    text: str | None = None
    model: str | None = None
    fingerprint: str | None = None
    usage: dict | None = None
    reason: str | None = None
    retryable: bool = False


class CloudReader:
    """Explicit, bounded cache warm-up; locked evaluation only reads the cache."""

    def __init__(self, cache_path: Path, *, identity: CacheIdentity,
                 max_requests: int = MAX_REQUESTS, client: httpx.Client | None = None) -> None:
        if (identity.model, identity.base_url) not in (
                (MODEL, BASE_URL), (GO_MODEL, GO_BASE_URL)):
            raise ValueError("cloud reader identity does not match a configured endpoint/model")
        self.key_env = GO_KEY_ENV if identity.base_url == GO_BASE_URL else "AIBOX_API_KEY"
        if max_requests < 0:
            raise ValueError("max_requests must not be negative")
        self.cache = GenerationCache(cache_path, identity)
        self.identity = identity
        self.max_new_tokens = identity.max_new_tokens
        self.max_requests = max_requests
        self.requests = 0
        self.generated = 0
        self.usage: list[dict] = []
        self.client = client or httpx.Client(timeout=45, transport=httpx.HTTPTransport(retries=0))

    def _attempt(self, prompt: str, key: str) -> CloudAttempt:
        reason = "invalid response shape"
        retryable = False
        try:
            response = self.client.post(
                f"{self.identity.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": self.identity.model, "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0, "max_tokens": self.max_new_tokens},
            )
            if response.status_code != 200:
                raise RuntimeError(f"cloud reader HTTP {response.status_code}")
            payload = response.json()
            choice = payload["choices"][0]
            text = choice["message"]["content"]
            if choice.get("finish_reason") == "length":
                raise ValueError("cloud response exhausted output budget")
            if choice.get("finish_reason") not in ("stop", "end_turn"):
                raise ValueError("cloud response did not complete normally")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("empty cloud reader response")
            model = payload.get("model")
            fingerprint = payload.get("system_fingerprint")
            if model is not None and (not isinstance(model, str) or not model):
                raise ValueError("invalid cloud model echo")
            if model is not None and model not in (
                    self.identity.model, self.identity.model.rsplit("/", 1)[-1]):
                raise ValueError("cloud model identity changed during run")
            if fingerprint is not None and not isinstance(fingerprint, str):
                raise ValueError("invalid cloud fingerprint echo")
            usage = payload.get("usage")
            record = None
            if isinstance(usage, dict):
                record = {k: v for k, v in usage.items()
                          if k in ("prompt_tokens", "completion_tokens", "total_tokens")
                          and isinstance(v, int)}
                for source, fields in (("prompt_tokens_details", ("cached_tokens",)),
                                       ("completion_tokens_details", ("reasoning_tokens",))):
                    details = usage.get(source)
                    if isinstance(details, dict):
                        record.update({field: details[field] for field in fields
                                       if isinstance(details.get(field), int)})
            return CloudAttempt(text=text, model=model, fingerprint=fingerprint, usage=record)
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError, RuntimeError) as exc:
            if isinstance(exc, httpx.HTTPError):
                reason = "transport error"
            elif isinstance(exc, RuntimeError) and str(exc).startswith("cloud reader HTTP "):
                reason = str(exc)
                retryable = reason in ("cloud reader HTTP 408", "cloud reader HTTP 429",
                                       "cloud reader HTTP 500", "cloud reader HTTP 502",
                                       "cloud reader HTTP 503", "cloud reader HTTP 504")
            elif isinstance(exc, ValueError) and str(exc) in (
                    "empty cloud reader response", "cloud response did not complete normally",
                    "cloud response exhausted output budget",
                    "invalid cloud model echo", "invalid cloud fingerprint echo",
                    "cloud model identity changed during run"):
                reason = str(exc)
            else:
                reason = "invalid response shape"
            retryable = retryable or reason in (
                "transport error", "empty cloud reader response",
                "cloud response did not complete normally", "invalid response shape")
            return CloudAttempt(reason=reason, retryable=retryable)

    def generate(self, prompts: Sequence[str], *, workers: int = 1) -> list[str]:
        if not 1 <= workers <= 10:
            raise ValueError("cloud workers must be between 1 and 10")
        missing = [p for p in dict.fromkeys(prompts) if p not in self.cache]
        if missing and not os.environ.get(self.key_env):
            raise RuntimeError(f"{self.key_env} is not set; cloud generation skipped")
        for prompt in missing:
            if len(prompt) > PILOT_MAX_PROMPT_CHARS:
                raise ValueError("cloud reader prompt exceeds character budget")
            if prompt in self.cache.failures:
                raise RuntimeError(f"cloud prompt has a recorded failure ({self.cache.failures[prompt]}); "
                                   "not retrying without a new reader identity")
            if self.cache.retries.get(prompt, 0) >= MAX_ATTEMPTS_PER_PROMPT:
                raise RuntimeError("cloud prompt exhausted its retry limit; cache incomplete")
        pending = deque((prompt, 0.0) for prompt in missing)
        active = {}
        error = None
        terminal_error = False
        cap_reached = False
        key = os.environ.get(self.key_env, "")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while pending or active:
                ready = []
                if error is None and not cap_reached:
                    now = time.monotonic()
                    for _ in range(len(pending)):
                        prompt, due = pending.popleft()
                        # Establish the provider echo from one response before parallel dispatch.
                        limit = workers if self.cache.provider_observed else 1
                        if len(active) + len(ready) < limit and due <= now:
                            if self.cache.attempts >= self.max_requests:
                                cap_reached = True
                                pending.appendleft((prompt, due))
                                break
                            self.requests += 1
                            self.cache.attempts += 1
                            self.cache.retries[prompt] = self.cache.retries.get(prompt, 0) + 1
                            ready.append(prompt)
                        else:
                            pending.append((prompt, due))
                    if ready:
                        # Persist every reservation before any worker can send its HTTP request.
                        self.flush()
                        for prompt in ready:
                            active[pool.submit(self._attempt, prompt, key)] = prompt
                if not active:
                    if error is not None or cap_reached:
                        break
                    if pending:
                        time.sleep(max(0, min(due for _, due in pending) - time.monotonic()))
                    continue
                timeout = (max(0, min(due for _, due in pending) - time.monotonic())
                           if pending and error is None and not cap_reached
                           and len(active) < workers else None)
                try:
                    completed, _ = wait(active, timeout=timeout, return_when=FIRST_COMPLETED)
                except KeyboardInterrupt:
                    # Let in-flight requests finish, then persist their results before stopping.
                    completed = set(active)
                    error = "cloud warm-up interrupted after in-flight requests completed"
                for future in completed:
                    prompt = active.pop(future)
                    result = future.result()
                    if result.reason is not None:
                        if result.retryable and self.cache.retries[prompt] < MAX_ATTEMPTS_PER_PROMPT:
                            if error is None:
                                delay = min(4, 2 ** (self.cache.retries[prompt] - 1))
                                pending.append((prompt, time.monotonic() + delay))
                        else:
                            if not result.retryable:
                                self.cache.failures[prompt] = result.reason
                            if error is None or (not result.retryable and not terminal_error):
                                error = (f"cloud reader request failed ({result.reason}); "
                                         "response and credentials withheld")
                                terminal_error = not result.retryable
                    elif self.cache.provider_observed and (
                            result.model != self.cache.provider_model or
                            result.fingerprint != self.cache.provider_fingerprint):
                        self.cache.failures[prompt] = "cloud model identity changed during run"
                        if not terminal_error:
                            error = ("cloud reader request failed (cloud model identity changed "
                                     "during run); response and credentials withheld")
                            terminal_error = True
                    else:
                        self.cache.provider_observed = True
                        self.cache.provider_model = result.model
                        self.cache.provider_fingerprint = result.fingerprint
                        self.cache.put(prompt, result.text)
                        self.generated += 1
                        if result.usage is not None:
                            self.usage.append(result.usage)
                if completed:
                    self.flush()
            if error is not None:
                raise RuntimeError(error)
            if cap_reached:
                raise RuntimeError("cloud request cap reached; cache contains only completed prompts")
        return self.cache.get(prompts)

    def flush(self) -> None:
        self.cache.flush()


def _score_factory(reader: CloudReader, docs: Mapping[str, Mapping], question: str,
                   reference: str, *, gate_citation: bool = True,
                   tau: float = REWARD_TAU) -> "callable":
    """Score one visible state; identical function for every state and every arm."""
    def score(state: tuple[str, ...]) -> tuple[float, float]:
        if not state:
            return 0.0, 0.0
        passages = [{"context_id": d, "text": docs[d]["text"]} for d in state]
        prompt = reader_prompt(question, passages)
        if len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError("state prompt exceeds the pre-registered character budget")
        raw = reader.cache.get([prompt])[0]  # locked read: a miss raises, it never samples
        text, _cited, valid = parse_reader_output(raw, state)
        f1 = f1_score(text, [reference]) if text else 0.0
        return answer_reward(f1, valid, gate_citation=gate_citation, tau=tau), 0.0
    return score


def cache_metadata(reader: CloudReader) -> dict:
    return {"cache_version": CACHE_VERSION, "identity": reader.cache.identity.as_dict()}


def _aggregate(records: Sequence[Mapping], *, key: str | None = None) -> dict:
    """Gold-stage aggregation: answer-text quality and citation-ID validity kept apart."""
    def block(rows: Sequence[Mapping]) -> dict:
        n = len(rows)
        if not n:
            return {"n": 0}
        return {"n": n,
                "answer_rate": round(sum(r["answer_present"] for r in rows) / n, 4),
                # Provenance metric: cited IDs unique and seen. Not an entailment check.
                "citation_valid_rate": round(sum(r["citation_valid"] for r in rows) / n, 4),
                "mean_f1": round(sum(r["f1"] for r in rows) / n, 4),
                "em_count": sum(r["em"] for r in rows),
                "f1_ge_0_8_count": sum(r["f1"] >= 0.8 for r in rows),
                "f1_ge_0_6_count": sum(r["f1"] >= 0.6 for r in rows),
                "f1_ge_0_5_count": sum(r["f1"] >= 0.5 for r in rows),
                # Reward sensitivity: citation-gated (locked) vs text-only.
                "reward_gated_rate": round(
                    sum(answer_reward(r["f1"], r["citation_valid"]) for r in rows) / n, 4),
                "reward_text_only_rate": round(
                    sum(answer_reward(r["f1"], r["citation_valid"], gate_citation=False)
                        for r in rows) / n, 4)}
    out = {"overall": block(records)}
    if key:
        groups: dict[str, list[Mapping]] = defaultdict(list)
        for row in records:
            groups[str(row[key])].append(row)
        out["by_" + key] = {name: block(rows) for name, rows in sorted(groups.items())}
    return out


def _aggregate_advantage(records: Sequence[Mapping], *, key: str | None = None) -> dict:
    """State-stage aggregation: reader-based Q values and search advantage."""
    def block(rows: Sequence[Mapping]) -> dict:
        n = len(rows)
        if not n:
            return {"n": 0}
        searches = [r["best_search"] for r in rows if r["best_search"] is not None]
        return {"n": n,
                "positive_search_advantage_count": sum(r["positive_search_advantage"] for r in rows),
                "positive_search_advantage_rate": round(
                    sum(r["positive_search_advantage"] for r in rows) / n, 4),
                "positive_search_advantage_rate_text_only": round(
                    sum(r["positive_search_advantage_text_only"] for r in rows) / n, 4),
                "mean_best_terminal": round(sum(r["best_terminal"] for r in rows) / n, 4),
                "mean_best_search": round(sum(searches) / len(searches), 4) if searches else None,
                "mean_full_episode_return": round(sum(r["full_episode_return"] for r in rows) / n, 4)}
    out = {"overall": block(records)}
    if key:
        groups: dict[str, list[Mapping]] = defaultdict(list)
        for row in records:
            groups[str(row[key])].append(row)
        out["by_" + key] = {name: block(rows) for name, rows in sorted(groups.items())}
    return out


def stage_gold(reader: CloudReader, docs: Mapping[str, Mapping], rows: Sequence[Mapping],
               *, out: Path) -> dict:
    # Refuse before any generation: these are historical measurements, never clobbered.
    refuse_overwrite(result_paths("gold", out))
    scorable, excluded = select_scored_rows(rows, docs)
    prompts, index = [], []
    for row in scorable:
        passages = [{"context_id": d, "text": docs[d]["text"]} for d in row["context_ids"]]
        prompts.append(reader_prompt(row["question"], passages))
        index.append(row)
    raw = reader.generate(prompts)
    records = []
    for row, prompt, response in zip(index, prompts, raw):
        text, _cited, valid = parse_reader_output(response, row["context_ids"])
        f1 = f1_score(text, [row["answer"]]) if text else 0.0
        records.append({"qid": row["id"], "gold_context_count": len(row["context_ids"]),
                        "question_type": row["question_type"], "hop_level": int(row["hop_level"]),
                        "answer_present": text is not None, "citation_valid": valid,
                        "f1": round(f1, 4), "em": exact_match(text, [row["answer"]]) if text else 0.0})
    summary = {"stage": "gold", "n_scored": len(records), "n_excluded_over_budget": len(excluded),
               "excluded_qids": excluded, "config": {
                   "model": MODEL, "model_revision": MODEL_REVISION,
                   "max_new_tokens": MAX_NEW_TOKENS, "decoding": "temperature=0",
                   "max_prompt_chars": MAX_PROMPT_CHARS,
                   "note": "gold passages supplied to the reader for diagnosis only",
                   "reward_note": "citation_valid is a provenance check (unique IDs the reader "
                                  "was shown), not entailment; answer F1 and citation validity are "
                                  "reported separately, plus reward_gated_rate vs "
                                  "reward_text_only_rate",
                   "cache": cache_metadata(reader)},
               **{k: v for k, v in _aggregate(records).items() if k == "overall"},
               "by_gold_context_count": _aggregate(records, key="gold_context_count")["by_gold_context_count"],
               "by_question_type": _aggregate(records, key="question_type")["by_question_type"]}
    atomic_write_json(out / "reader-gold-train500.json", summary)
    atomic_write_json(out / "reader-gold-train500-rows.json", records)
    return summary


def stage_states(reader: CloudReader, docs: Mapping[str, Mapping], rows: Sequence[Mapping],
                 *, out: Path, rankings_path: Path, bootstrap_passages: int = 1,
                 passage_per_search_action: int = 1, max_optional_search: int = 2,
                 tag: str = "") -> dict:
    # Refuse before any generation: these are historical measurements, never clobbered.
    refuse_overwrite(result_paths("states", out, tag))
    rankings = json.loads(rankings_path.read_text(encoding="utf-8"))
    records, all_states, skipped = [], [], []
    planned: list[tuple[Mapping, list[tuple[str, ...]], list[str]]] = []
    for row in rows:
        state_set = reachable_states({"bm25": rankings[row["id"]]["bm25"],
                                      "dense": rankings[row["id"]]["dense"]},
                                     bootstrap_passages=bootstrap_passages,
                                     passage_per_search_action=passage_per_search_action,
                                     max_optional_search=max_optional_search)
        prompts = [reader_prompt(row["question"], [{"context_id": d, "text": docs[d]["text"]}
                                                   for d in state]) for state in state_set]
        if any(len(prompt) > MAX_PROMPT_CHARS for prompt in prompts):
            skipped.append(row["id"])
            continue
        planned.append((row, state_set, prompts))
        all_states.append(len(state_set))
    # One batched pass fills the cache; the recursive walk below then re-reads it, so the
    # recursion costs no further generation.
    if planned:
        reader.generate([prompt for _row, _states, prompts in planned for prompt in prompts])
        reader.flush()
    for row, state_set, _prompts in planned:
        root = tuple(state_set[0])
        score = _score_factory(reader, docs, row["question"], row["answer"])
        root_value, root_q = q_values(root, rankings[row["id"]], score,
                                      remaining=max_optional_search, search_cost=SEARCH_COST,
                                      passage_per_search_action=passage_per_search_action)
        terminal = {"ANSWER": root_q["ANSWER"], "ABSTAIN": root_q["ABSTAIN"]}
        branches = {k: v for k, v in root_q.items() if k.startswith("SEARCH_")}
        advantage = search_advantage(branches, terminal, search_cost=SEARCH_COST)
        # Sensitivity: same cached generations, reward without the citation gate.
        score_text = _score_factory(reader, docs, row["question"], row["answer"],
                                    gate_citation=False)
        _, root_q_text = q_values(root, rankings[row["id"]], score_text,
                                  remaining=max_optional_search, search_cost=SEARCH_COST,
                                  passage_per_search_action=passage_per_search_action)
        branches_text = {k: v for k, v in root_q_text.items() if k.startswith("SEARCH_")}
        advantage_text = search_advantage(
            branches_text, {"ANSWER": root_q_text["ANSWER"], "ABSTAIN": root_q_text["ABSTAIN"]},
            search_cost=SEARCH_COST)
        records.append({"qid": row["id"], "gold_context_count": len(row["context_ids"]),
                        "question_type": row["question_type"], "hop_level": int(row["hop_level"]),
                        "root_q": {k: round(v, 4) for k, v in root_q.items()},
                        "best_terminal": round(advantage["best_terminal"], 4),
                        "best_search": None if advantage["best_search"] == float("-inf")
                        else round(advantage["best_search"], 4),
                        "search_advantage": None if advantage["best_search"] == float("-inf")
                        else round(advantage["advantage"], 4),
                        "positive_search_advantage": advantage["positive"],
                        "positive_search_advantage_text_only": advantage_text["positive"],
                        "optimal_action": advantage["optimal"],
                        "full_episode_return": round(root_value, 4)})
    scored = [r for r in records]
    summary = {"stage": "states",
               "config": {"model": MODEL, "model_revision": MODEL_REVISION,
                          "max_new_tokens": MAX_NEW_TOKENS, "decoding": "temperature=0",
                          "tag": tag, "bootstrap_passages": bootstrap_passages,
                          "passage_per_search_action": passage_per_search_action,
                          "max_optional_search": max_optional_search,
                          "tools": list(TOOLS), "rankings_path": str(rankings_path),
                          "cache": cache_metadata(reader)},
               "reward": {"definition": "R_ANSWER=1 if citation_valid and F1>=tau else 0; "
                                        "R_ABSTAIN=0; SEARCH pays search_cost",
                          "tau_used": REWARD_TAU, "search_cost": SEARCH_COST,
                          "citation_semantics": "citation_valid checks cited IDs are unique and "
                                                "were shown; it is provenance, not entailment",
                          "sensitivity": "positive_search_advantage_rate_text_only drops the "
                                         "citation gate and keeps answer F1 only",
                          "note": "every VixSTORY question is answerable, so ABSTAIN=0 is a real floor"},
               "n_questions": len(scored), "n_skipped_over_budget": len(skipped),
               "skipped_qids": skipped,
               "states_per_question_mean": round(sum(all_states) / len(all_states), 2) if all_states else 0,
               "states_per_question_max": max(all_states) if all_states else 0,
               "generations": reader.generated, "unique_prompts": len(reader.cache),
               "positive_search_advantage_rate": round(
                   sum(r["positive_search_advantage"] for r in scored) / len(scored), 4) if scored else None,
               "positive_search_advantage_rate_text_only": round(
                   sum(r["positive_search_advantage_text_only"] for r in scored) / len(scored), 4)
               if scored else None,
               "optimal_action_distribution": dict(Counter(r["optimal_action"] for r in scored)),
               "mean_full_episode_return": round(
                   sum(r["full_episode_return"] for r in scored) / len(scored), 4) if scored else None,
               "mean_answer_at_root": round(
                   sum(r["best_terminal"] for r in scored) / len(scored), 4) if scored else None,
               "mean_best_search_branch": round(
                   sum(r["best_search"] for r in scored if r["best_search"] is not None)
                   / max(1, sum(r["best_search"] is not None for r in scored)), 4) if scored else None,
               "by_gold_context_count": _aggregate_advantage(scored, key="gold_context_count")["by_gold_context_count"],
               "by_question_type": _aggregate_advantage(scored, key="question_type")["by_question_type"]}
    atomic_write_json(out / f"search-advantage-train500{tag}.json", summary)
    atomic_write_json(out / f"search-advantage-train500{tag}-rows.json", records)
    return summary


def build_identity(*, corpus_path: Path = CORPUS_PATH,
                   rankings_path: Path = DEFAULT_RANKINGS,
                   provider: str = "aibox") -> CacheIdentity:
    """Bind cloud generation to the provider, message contract and visible-data artifacts."""
    if not Path(rankings_path).exists():
        raise FileNotFoundError(f"rankings artifact required to bind the cache identity: "
                                f"{rankings_path}")
    if provider not in ("aibox", "opencode-go"):
        raise ValueError(f"unknown cloud provider: {provider}")
    model, base_url = ((GO_MODEL, GO_BASE_URL) if provider == "opencode-go"
                       else (MODEL, BASE_URL))
    return CacheIdentity(model=model, model_revision=MODEL_REVISION, base_url=base_url,
                         prompt_version=PROMPT_VERSION, decoding="temperature=0",
                         max_new_tokens=MAX_NEW_TOKENS,
                         corpus_sha256=sha256_file(corpus_path),
                         rankings_sha256=sha256_file(rankings_path),
                         message_version=MESSAGE_VERSION)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("gold", "states"), required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache", type=Path, default=None,
                        help=f"identity-bound generation cache (defaults to <out>/{DEFAULT_CACHE_NAME}); "
                             "legacy prompt-only caches are refused")
    parser.add_argument("--rankings", type=Path, default=DEFAULT_RANKINGS)
    parser.add_argument("--max-requests", type=int, default=MAX_REQUESTS)
    parser.add_argument("--bootstrap-passages", type=int, default=1)
    parser.add_argument("--passage-per-search-action", type=int, default=1)
    parser.add_argument("--max-optional-search", type=int, default=2)
    parser.add_argument("--tag", default=None, help="output filename suffix for this config")
    args = parser.parse_args()
    tag = args.tag or ""
    # Refuse to clobber a historical result before loading data or spending any generation.
    refuse_overwrite(result_paths(args.stage, args.out, tag))
    args.out.mkdir(parents=True, exist_ok=True)
    docs, rows = _load()
    # The ranking fingerprint is recorded for every run (gold included) so one cache file cannot
    # be reused under a different retrieval artifact.
    identity = build_identity(rankings_path=args.rankings)
    from src.utils.runs import OutputNamespaceLock

    cache_path = args.cache or (args.out / DEFAULT_CACHE_NAME)
    started = time.time()
    with OutputNamespaceLock(cache_path.with_name(cache_path.name + ".lock")):
        reader = CloudReader(cache_path, identity=identity, max_requests=args.max_requests)
        if args.stage == "gold":
            result = stage_gold(reader, docs, rows, out=args.out)
        else:
            result = stage_states(
                reader, docs, rows, out=args.out, rankings_path=args.rankings,
                bootstrap_passages=args.bootstrap_passages,
                passage_per_search_action=args.passage_per_search_action,
                max_optional_search=args.max_optional_search, tag=tag)
        reader.flush()
    result["generated_this_run"] = reader.generated
    result["requests_this_run"] = reader.requests
    result["usage"] = reader.usage
    result["seconds"] = round(time.time() - started, 1)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("excluded_qids", "skipped_qids")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
