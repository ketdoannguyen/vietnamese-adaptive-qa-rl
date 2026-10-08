"""Strict, research-grade, group-disjoint splitting of the source ``train`` split.

Policy (``01_SPECIFICATIONS/04_DATASET_VALIDITY_AND_LEAKAGE.md`` §4):

* only the original HF ``train`` split is split, into ``train_strict`` / ``val_strict`` /
  ``test_strict``;
* partitioning is by ``title_group`` (never by individual QA row), target 70/15/15 by group
  with approximately stratified ``is_impossible``, fixed seed;
* near-duplicate contexts across groups are *resolved* by merging their groups before the
  final allocation, not ignored;
* a profile cap subsamples rows per strict split with a deterministic, nested ordering, so
  ``smoke ⊂ pilot ⊂ ... ⊂ full`` and caches stay valid as a run grows.
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from src.utils.hashing import stable_seed

STRICT_SPLITS = ("train", "val", "test")
DEFAULT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


@dataclass(frozen=True)
class GroupStat:
    rows: int = 0
    impossible: int = 0

    @property
    def answerable(self) -> int:
        return self.rows - self.impossible

    def __add__(self, other: "GroupStat") -> "GroupStat":
        return GroupStat(self.rows + other.rows, self.impossible + other.impossible)


@dataclass(frozen=True)
class CapInfo:
    requested: int | None
    applied: bool
    rows_out: int
    rows_pool: int
    answerable_out: int
    impossible_out: int
    answerable_pool: int
    impossible_pool: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested, "applied": self.applied,
            "rows_out": self.rows_out, "rows_pool": self.rows_pool,
            "answerable_out": self.answerable_out, "impossible_out": self.impossible_out,
            "answerable_pool": self.answerable_pool, "impossible_pool": self.impossible_pool,
        }


# --------------------------------------------------------------------------------------
# group allocation
# --------------------------------------------------------------------------------------

def allocate_groups(
    stats: Mapping[str, GroupStat],
    *,
    ratios: Mapping[str, float] = DEFAULT_RATIOS,
    splits: Sequence[str] = STRICT_SPLITS,
    seed_salt: object = 42,
) -> dict[str, str]:
    """Deterministic greedy allocation of whole groups to strict splits.

    A group goes to the split whose *relative* deficit of rows (plus half-weighted deficit
    of impossible rows, i.e. approximate stratification) is largest. Ties keep the
    canonical split order, so the result depends only on ``stats`` and ``seed_salt``.
    """
    total = sum(s.rows for s in stats.values())
    if total == 0:
        raise ValueError("cannot split an empty pool")
    impossible_total = sum(s.impossible for s in stats.values())
    impossible_ratio = impossible_total / total
    targets = {s: ratios[s] * total for s in splits}
    target_imp = {s: targets[s] * impossible_ratio for s in splits}
    order = sorted(stats.items(), key=lambda kv: (-kv[1].rows, stable_seed(seed_salt, kv[0])))
    current = {s: GroupStat() for s in splits}
    assignment: dict[str, str] = {}
    for group, stat in order:
        best_split, best_score = splits[0], -math.inf
        for s in splits:
            row_deficit = (targets[s] - current[s].rows) / max(targets[s], 1.0)
            imp_deficit = (target_imp[s] - current[s].impossible) / max(target_imp[s], 1.0)
            score = row_deficit + 0.5 * imp_deficit
            if score > best_score + 1e-12:
                best_split, best_score = s, score
        assignment[group] = best_split
        current[best_split] = current[best_split] + stat
    return assignment


def group_stats(rows: Iterable[Any], key) -> dict[str, GroupStat]:
    acc: dict[str, GroupStat] = defaultdict(GroupStat)
    for row in rows:
        g = key(row)
        acc[g] = acc[g] + GroupStat(1, 1 if row.question.private.is_impossible else 0)
    return dict(acc)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def merge_groups(
    pairs: Iterable[tuple[str, str]], groups: Iterable[str] = ()
) -> dict[str, str]:
    """Union-find over title groups joined by an edge list; returns ``group -> component``.

    Edges come from two sources: near-duplicate document pairs (two groups whose texts are
    near-identical) and documents referenced by more than one title (a shared context would
    otherwise be placed in two strict splits at once, breaking ``doc_id`` isolation).
    """
    uf = _UnionFind()
    for group in groups:
        uf.find(group)
    for a, b in pairs:
        uf.union(a, b)
    return {group: uf.find(group) for group in uf.parent}


# --------------------------------------------------------------------------------------
# row selection (caps)
# --------------------------------------------------------------------------------------

def stratified_order(rows: Sequence[Any], *, salt: object) -> list[Any]:
    """Round-robin over groups ordered by a stable hash, rows ordered by a stable hash.

    The result is independent of the requested cap, so a smaller profile's selection is
    always a prefix of a larger profile's selection.
    """
    by_group: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        by_group[row.question.title_group].append(row)
    groups = sorted(by_group, key=lambda g: (stable_seed(salt, "group", g), g))
    for g in groups:
        by_group[g].sort(key=lambda r: (stable_seed(salt, "qid", r.question.qid), r.question.qid))
    ordered: list[Any] = []
    depth = 0
    while True:
        progressed = False
        for g in groups:
            bucket = by_group[g]
            if depth < len(bucket):
                ordered.append(bucket[depth])
                progressed = True
        if not progressed:
            return ordered
        depth += 1


def cap_rows(rows: Sequence[Any], cap: int | None, *, salt: object) -> tuple[list[Any], CapInfo]:
    """Subsample rows to ``cap`` with an approximately stratified answerability mix."""
    answerable = [r for r in rows if not r.question.private.is_impossible]
    impossible = [r for r in rows if r.question.private.is_impossible]
    if cap is None or len(rows) <= cap:
        return list(rows), CapInfo(cap, False, len(rows), len(rows), len(answerable),
                                   len(impossible), len(answerable), len(impossible))
    ratio = len(answerable) / len(rows)
    n_pos = min(len(answerable), int(round(cap * ratio)))
    n_neg = min(len(impossible), cap - n_pos)
    shortfall = cap - n_pos - n_neg
    if shortfall > 0:
        take = min(shortfall, len(answerable) - n_pos)
        n_pos += take
        shortfall -= take
    if shortfall > 0:
        n_neg += min(shortfall, len(impossible) - n_neg)
    selected = stratified_order(answerable, salt=(salt, "answerable"))[:n_pos]
    selected += stratified_order(impossible, salt=(salt, "impossible"))[:n_neg]
    selected.sort(key=lambda r: r.question.qid)
    return selected, CapInfo(cap, True, len(selected), len(rows), n_pos, n_neg,
                             len(answerable), len(impossible))


# --------------------------------------------------------------------------------------
# near-duplicate detection (TF-IDF cosine)
# --------------------------------------------------------------------------------------

def _vectorize(texts: Sequence[str]):
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
    vectorizer = HashingVectorizer(analyzer="word", ngram_range=(1, 2), n_features=2 ** 20,
                                   alternate_sign=False, norm=None, lowercase=True,
                                   token_pattern=r"\w+")
    counts = vectorizer.transform(texts)
    return TfidfTransformer(sublinear_tf=True, norm="l2").fit_transform(counts)


def _collect(coo, keys_a: Sequence[str], keys_b: Sequence[str], offset: int,
             threshold: float, out: list[tuple[str, str, float]], *,
             upper_triangle: bool) -> float:
    maximum = 0.0
    for i, j, value in zip(coo.row, coo.col, coo.data):
        gi = offset + int(i)
        gj = int(j)
        if upper_triangle and gj <= gi:
            continue
        if value > maximum:
            maximum = float(value)
        if value >= threshold and len(out) < MAX_PAIRS:
            out.append((keys_a[gi], keys_b[gj], round(float(value), 6)))
    return maximum


#: hard cap on collected near-duplicate pairs, so a pathological pool cannot exhaust RAM
MAX_PAIRS = 20000


def cross_similarity(
    keys_a: Sequence[str], texts_a: Sequence[str],
    keys_b: Sequence[str], texts_b: Sequence[str],
    *, threshold: float, chunk: int = 512,
) -> dict[str, Any]:
    """Cosine similarity between two keyed text collections (TF-IDF word 1-2 grams).

    Returns the full pair list under ``"all"`` (capped at :data:`MAX_PAIRS`); callers that
    write reports must trim it.
    """
    matrix = _vectorize(list(texts_a) + list(texts_b))
    a = matrix[: len(texts_a)]
    b = matrix[len(texts_a):]
    found: list[tuple[str, str, float]] = []
    maximum = 0.0
    for start in range(0, a.shape[0], chunk):
        block = a[start:start + chunk] @ b.T
        maximum = max(maximum, _collect(block.tocoo(), keys_a, keys_b, start, threshold,
                                        found, upper_triangle=False))
    found.sort(key=lambda t: (-t[2], t[0], t[1]))
    return {"method": "tfidf_word_1_2_gram_cosine", "threshold": threshold,
            "pairs": len(found), "max_similarity": round(maximum, 6),
            "truncated": len(found) >= MAX_PAIRS, "all": found}


def self_similarity(
    keys: Sequence[str], texts: Sequence[str], *, threshold: float, chunk: int = 512,
) -> dict[str, Any]:
    """Pairwise near-duplicate detection inside one collection (upper triangle only)."""
    matrix = _vectorize(texts)
    found: list[tuple[str, str, float]] = []
    maximum = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for start in range(0, matrix.shape[0], chunk):
            block = matrix[start:start + chunk] @ matrix.T
            maximum = max(maximum, _collect(block.tocoo(), keys, keys, start, threshold,
                                           found, upper_triangle=True))
    found.sort(key=lambda t: (-t[2], t[0], t[1]))
    return {"method": "tfidf_word_1_2_gram_cosine", "threshold": threshold,
            "pairs": len(found), "max_similarity": round(maximum, 6),
            "truncated": len(found) >= MAX_PAIRS, "all": found}


def near_duplicate_report(result: Mapping[str, Any], *, limit: int = 30) -> dict[str, Any]:
    """Trim a similarity result for JSON reports (drops the full pair list)."""
    return {
        "method": result["method"], "threshold": result["threshold"],
        "pairs": result["pairs"], "max_similarity": result["max_similarity"],
        "truncated": result["truncated"],
        "examples": [{"a": a, "b": b, "cosine": s} for a, b, s in result["all"][:limit]],
    }
