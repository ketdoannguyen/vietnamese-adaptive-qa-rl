"""VixSTORY REINFORCE policy: canonical feature extractor + trainer glue (E06/E07).

VixSTORY differs from ViQuAD in two ways that make the ViQuAD feature vector and
checkpoint unusable as-is:

* answers are **free-form with multiple supporting passages** (``cited_ids``), so the
  reader is not an extractive single-doc span model: there is no honest ``non_null_score``
  and no ``null_margin``. This module therefore **never** fabricates those extractive
  features and never derives a fake span provenance from them. The one feature contract is
  :class:`~src.rl.vixstory_env.VixStoryFeatureExtractor`, re-exported here so there is no
  second, drifting definition.
* the frozen Laya prior is a ``Sequence[float]`` of exactly four action probabilities.

Everything else is reused from :mod:`src.rl.reinforce` — ``rollout`` (which *always* samples
actions from the current network, never from a cached trajectory), ``reinforce_loss``,
``evaluate`` and ``PolicyNet``/:class:`~src.rl.policy.TorchScaler`. A *fresh* policy is
initialised here; no ViQuAD checkpoint is copied.

On-policy actions, immutable cache for dynamics
-----------------------------------------------
Episodes are produced by :func:`src.rl.reinforce.rollout`, so the action at every step is
sampled from the network being trained. The immutable cache lives on the environment side
(frozen retrieval rankings and frozen reader outputs) and only ever supplies *dynamics*; no
action sequence is ever replayed from a cache. ``tests/test_vixstory_policy.py`` recomputes
each transition's ``log_prob`` under the current network to prove this.

VixSTORY observation schema
---------------------------
The single feature contract is :class:`~src.rl.vixstory_env.VixStoryFeatureExtractor`: it
consumes exactly the observation :class:`~src.rl.vixstory_env.VixStoryEnv` returns from
``reset``/``step`` — ``qid``, ``question``, ``action_mask``, the free-form ``candidate``
(``text`` / ``cited_ids`` / ``citation_valid`` / ``answer_present``, no extractive score),
the (off-by-default) ``retrieval`` preview, ``seen_doc_ids`` / ``seen_doc_texts``,
``searches_used`` / ``optional_used`` / ``remaining_optional_searches``, ``last_action`` and
``timestep``. An observation must carry **no** gold answer, gold passage ids, hop level,
oracle action or reward (``src.data.schemas.find_forbidden_keys`` must return ``[]``).
Features are raw here; standardisation happens once, at the network input, via the persisted
``FeatureScaler``.

Checkpoint isolation and identity
---------------------------------
:func:`train_vixstory_policy` trains two independently seeded arms under one caller run
root — ``E06_rl_rlcd`` (Laya RLCD prior) and ``E07_rl_ce`` (Laya CE prior). It refuses to
touch an existing arm directory unless the caller explicitly asks for a *safe resume*, and
it only ever reads its own checkpoint (never a ViQuAD one). It persists the scaler, the
per-update history (grad norms and weight deltas) and a summary carrying an identity hash.

The identity is a content hash of the *actual* frozen contracts a safe resume must agree on:
the canonical extractor's feature names, the train and dev environments' corpus / retrieval /
reader fingerprints (read off each world) and the frozen prior's outputs at every train root
state — plus the arm, seed, config and the train/dev QID manifests. A checkpoint whose
summary identity differs is refused, e.g. one left by the old feature extractor, by another
reader or another prior, so an accidental resume never extends a different run.

Limits
------
* No GPU work and no official-validation use happen here or in its tests: only the train and
  dev environments the caller builds from the frozen manifests are touched.
* Safe resume restores policy, scaler, history and identity, but **not** the Adam moment
  estimates; a resumed run is a correct continuation of the artefacts, not a bit-identical
  one. This is documented rather than hidden.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from src.rl.features import FeatureScaler
from src.rl.policy import PolicyNet, TorchScaler
from src.rl.reinforce import (
    RLConfig,
    ReinforceTrainer,
    _param_vector,
    evaluate,
    reinforce_loss,
    rollout,
)
from src.rl.vixstory_env import VixStoryFeatureExtractor, vixstory_feature_names
from src.utils.hashing import sha256_json, stable_seed
from src.utils.io import append_jsonl, atomic_write_json

__all__ = [
    "ARM_EXPERIMENT",
    "VIXSTORY_ARMS",
    "VixStoryArmResult",
    "VixStoryFeatureExtractor",
    "VixStoryReinforceTrainer",
    "train_vixstory_policy",
    "vixstory_feature_names",
]

#: ``E06`` trains against the Laya RLCD prior, ``E07`` against the Laya CE prior.
ARM_EXPERIMENT = {"rlcd": "E06", "ce": "E07"}
VIXSTORY_ARMS = ("rlcd", "ce")
HISTORY_FILE = "history.json"
SUMMARY_FILE = "summary.json"
TRAIN_LOG_FILE = "train_log.jsonl"


def _extractor_fingerprint() -> str:
    """The canonical feature contract. Changes whenever the feature vector changes."""
    return sha256_json({"features": vixstory_feature_names()})


def _env_fingerprint(env: Any) -> str:
    """Fingerprint of an assembled environment's frozen corpus / retrieval / reader inputs.

    Read off every world: its corpus hash, its BM25/Dense rankings and the frozen reader
    checkpoint hash it serves (empty when the world was assembled without a pinned reader).
    Returns ``""`` for an environment that exposes no worlds (nothing honest to fingerprint).
    """
    worlds = getattr(env, "_worlds", None) or getattr(env, "worlds", None)
    if not worlds:
        return ""
    rows = [
        {
            "qid": world.qid,
            "corpus_sha": getattr(world, "corpus_sha", ""),
            "rankings": {tool: list(ranking) for tool, ranking in sorted(world.rankings.items())},
            "reader_sha": getattr(world, "reader_sha", ""),
        }
        for world in sorted(worlds, key=lambda w: getattr(w, "qid", ""))
    ]
    return sha256_json(rows)


def _prior_fingerprint(env: Any, qids: Sequence[str],
                       prior: Callable[[dict], Sequence[float]]) -> str:
    """Fingerprint of the frozen prior: its outputs at every episode root state."""
    rows = []
    for qid in sorted(qids):
        obs, _ = env.reset(seed=0, options={"qid": qid})
        rows.append([qid, [float(x) for x in prior(obs)]])
    return sha256_json(rows)


def _arm_identity(*, arm: str, seed: int, config: RLConfig, train_qids: Sequence[str],
                  dev_qids: Sequence[str], extractor_fp: str, train_env_fp: str,
                  dev_env_fp: str, prior_fp: str, prior_ref: str | None) -> str:
    """Content hash of everything a safe resume must agree on."""
    return sha256_json({
        "extractor": extractor_fp,
        "arm": arm,
        "seed": int(seed),
        "config": dict(sorted(asdict(config).items())),
        "train_qids": sorted(train_qids),
        "dev_qids": sorted(dev_qids),
        "train_env_fp": train_env_fp,
        "dev_env_fp": dev_env_fp,
        "prior_fp": prior_fp,
        "prior_ref": prior_ref,
    })


@dataclass(frozen=True)
class VixStoryArmResult:
    """Outcome of one REINFORCE arm (E06 or E07)."""

    arm: str
    experiment: str
    status: str  # "TRAINED" | "RESUMED"
    checkpoint_dir: str
    seed: int
    n_updates: int
    best_val_reward: float | None
    best_update: int | None
    stopped_early: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VixStoryReinforceTrainer(ReinforceTrainer):
    """A :class:`ReinforceTrainer` with the canonical VixSTORY extractor injected.

    It reuses the base scaler fit, network build, save/load and evaluation, but overrides
    ``train`` so a run can start at an offset (used by safe resume) and so the arm's
    identity and history are written alongside the checkpoint.
    """

    def __init__(self, train_env: Any, dev_env: Any, config: RLConfig, *, arm: str,
                 identity: str, laya_fn: Callable[[dict], Sequence[float]],
                 train_qids: Sequence[str], prior_ref: str | None = None,
                 device: str = "cpu", run_dir: str | Path | None = None,
                 seed: int | None = None, dev_qids: Sequence[str] | None = None,
                 fixed_updates: bool = False, train_monitor: bool = False) -> None:
        if arm not in VIXSTORY_ARMS:
            raise ValueError(f"unknown VixSTORY arm {arm!r}; known: {list(VIXSTORY_ARMS)}")
        super().__init__(train_env, dev_env, config, laya_fn=laya_fn, device=device,
                         run_dir=run_dir, seed=seed, val_qids=dev_qids)
        self.extractor = VixStoryFeatureExtractor()
        self.arm = arm
        self.identity = identity
        self.prior_ref = prior_ref
        self.train_qids = tuple(train_qids)
        self.start_update = 1
        self.fixed_updates = fixed_updates
        self.train_monitor = train_monitor

    # -- resume --------------------------------------------------------------
    def _read_history(self, ckpt: Path) -> list[dict]:
        log = ckpt / TRAIN_LOG_FILE
        if not log.is_file():
            return []
        rows = []
        for line in log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def resume_from(self, ckpt_dir: str | Path, *, identity: str) -> None:
        """Restore policy, scaler and history from our own checkpoint for a safe resume."""
        ckpt = Path(ckpt_dir)
        summary_path = ckpt / SUMMARY_FILE
        if not summary_path.is_file():
            raise ValueError(f"cannot resume: {summary_path} is missing; nothing to verify")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("identity") != identity:
            raise ValueError(
                "cannot resume: checkpoint identity does not match the current "
                "arm/seed/config/qid manifest/extractor/env/prior fingerprints; refusing to "
                "extend a different run")
        policy_path = ckpt / "policy.safetensors"
        scaler_path = ckpt / "scaler.json"
        if not policy_path.is_file() or not scaler_path.is_file():
            raise ValueError(f"cannot resume: {ckpt} lacks policy.safetensors/scaler.json")
        self.net = PolicyNet.load(policy_path).to(self.device)
        self.scaler = FeatureScaler.load(scaler_path)
        self.scaler_t = TorchScaler(np.asarray(self.scaler.mean, dtype=np.float32),
                                    np.asarray(self.scaler.std, dtype=np.float32),
                                    self.scaler.clip).to(self.device)
        self.history = self._read_history(ckpt)
        self.start_update = max((int(r["update"]) for r in self.history), default=0) + 1

    # -- training ------------------------------------------------------------
    def _restore_best(self) -> tuple[float, int]:
        best_val, best_update = float("-inf"), 0
        for row in self.history:
            val = row.get("train_monitor" if self.train_monitor else "val")
            if val is not None and float(val["mean_reward"]) > best_val:
                best_val = float(val["mean_reward"])
                best_update = int(row["update"])
        return best_val, best_update

    def train(self) -> dict[str, Any]:
        if self.net is None:
            self.build()
        if self.scaler is None:
            self.fit_scaler()
        assert self.net is not None and self.scaler is not None
        self.scaler_t = self.scaler_t.to(self.device)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.lr)
        rng = np.random.default_rng(stable_seed(self.seed, "vixstory-rollouts", self.start_update))
        best_val, best_update = self._restore_best()
        patience = 0
        result: dict[str, Any] = {"updates": [], "stopped_early": False,
                                  "best_val_reward": None, "best_update": None}

        for update in range(self.start_update, self.config.max_updates + 1):
            before = _param_vector(self.net)
            episodes = [
                rollout(self.train_env, self.net, self.extractor, self.scaler_t, self.laya_fn,
                        seed=int(rng.integers(0, 2**31 - 1)), device=self.device)
                for _ in range(self.config.batch_episodes)
            ]
            loss, stats = reinforce_loss(episodes, self.net, self.scaler_t, self.config,
                                         device=self.device)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.net.parameters(), self.config.grad_clip)
            optimizer.step()

            after = _param_vector(self.net)
            row = {
                "update": update,
                "train_mean_reward": float(np.mean([e.total_reward for e in episodes])),
                "grad_norm": float(grad_norm),
                "param_delta": float(np.abs(after - before).mean()),
                **stats,
            }

            stop = False
            if update % self.config.eval_every == 0:
                kwargs: dict[str, Any] = {"episodes": self.config.eval_episodes,
                                          "device": self.device}
                if self.val_qids is not None:
                    kwargs = {"episodes": len(self.val_qids), "qids": self.val_qids,
                              "device": self.device}
                val = evaluate(self.val_env, self.net, self.extractor, self.scaler_t,
                               self.laya_fn, **kwargs)
                row["train_monitor" if self.train_monitor else "val"] = val
                if val["mean_reward"] > best_val + 1e-6:
                    best_val, best_update, patience = val["mean_reward"], update, 0
                    if self.run_dir:
                        self.save(Path(self.run_dir) / "best")
                else:
                    patience += 1
                    if not self.fixed_updates and patience >= self.config.early_stop_patience:
                        result["stopped_early"] = True
                        stop = True

            if self.run_dir:
                self.save(Path(self.run_dir) / "last")
                append_jsonl(Path(self.run_dir) / TRAIN_LOG_FILE, row)
            self.history.append(row)
            result["updates"].append(row)
            if stop:
                break

        result["best_val_reward"] = best_val if best_val > float("-inf") else None
        result["best_update"] = best_update or None
        return result

    # -- persistence ---------------------------------------------------------
    def persist(self, ckpt_dir: str | Path) -> None:
        """Write the final policy + scaler + history + summary for this arm."""
        ckpt = Path(ckpt_dir)
        ckpt.mkdir(parents=True, exist_ok=True)
        self.save(ckpt)
        atomic_write_json(ckpt / HISTORY_FILE, self.history)
        best_val, best_update = self._restore_best()
        atomic_write_json(ckpt / SUMMARY_FILE, {
            "arm": self.arm,
            "experiment": ARM_EXPERIMENT[self.arm],
            "seed": int(self.seed),
            "fixed_updates": self.fixed_updates,
            "evaluation_scope": "train_monitor" if self.train_monitor else "dev",
            "identity": self.identity,
            "prior_ref": self.prior_ref,
            "train_qids": list(self.train_qids),
            "dev_qids": list(self.val_qids or ()),
            "n_updates": len(self.history),
            "last_update": max((int(r["update"]) for r in self.history), default=0),
            "best_val_reward": None if best_val == float("-inf") else best_val,
            "best_update": best_update or None,
        })


def _validate_qids(train_qids: Sequence[str], dev_qids: Sequence[str], *,
                   train_monitor: bool = False) -> None:
    if not train_qids or not dev_qids:
        raise ValueError("train_qids and dev_qids must both be non-empty")
    for label, qids in (("train_qids", train_qids), ("dev_qids", dev_qids)):
        if len(set(qids)) != len(qids):
            raise ValueError(f"{label} contains duplicates")
    overlap = set(train_qids) & set(dev_qids)
    if train_monitor:
        if not set(dev_qids) <= set(train_qids):
            raise ValueError("train monitor QIDs must be a subset of official training QIDs")
        return
    if overlap:
        raise ValueError(
            f"train and dev qids must be disjoint; {len(overlap)} shared, e.g. {sorted(overlap)[0]!r}")


def _env_world_qids(env: Any) -> list[str] | None:
    worlds = getattr(env, "_worlds", None) or getattr(env, "worlds", None)
    if worlds is None:
        return None
    return [getattr(w, "qid", None) for w in worlds]


def _validate_env(env: Any, qids: Sequence[str], label: str) -> None:
    """Prove ``env`` holds exactly ``qids`` and can select each by ``options={"qid": ...}``."""
    world_qids = _env_world_qids(env)
    if world_qids is not None and set(world_qids) != set(qids):
        extra = sorted(set(world_qids) - set(qids))[:5]
        missing = sorted(set(qids) - set(world_qids))[:5]
        raise ValueError(
            f"{label} env does not match the qid manifest (extra={extra}, missing={missing}); "
            f"the {label} environment must contain exactly the manifest qids")
    for qid in qids:
        obs, _ = env.reset(seed=0, options={"qid": qid})
        if obs.get("qid") != qid:
            raise ValueError(f"{label} env returned qid {obs.get('qid')!r} for requested {qid!r}")


def train_vixstory_policy(train_env: Any, dev_env: Any, config: RLConfig, *,
                          priors: Mapping[str, Callable[[dict], Sequence[float]]],
                          run_root: str | Path, train_qids: Sequence[str],
                          dev_qids: Sequence[str],
                          arm_seeds: Mapping[str, int] | None = None,
                          prior_refs: Mapping[str, str] | None = None,
                          resume: bool = False, device: str = "cpu",
                          fixed_updates: bool = False,
                          train_monitor: bool = False,
                          ) -> dict[str, VixStoryArmResult]:
    """Train the two isolated VixSTORY REINFORCE arms under ``run_root``.

    ``priors`` supplies the frozen Laya prior callable per arm (``"rlcd"`` for E06,
    ``"ce"`` for E07), each returning a ``Sequence[4]`` of action probabilities. Only
    ``train_env`` (train300) and ``dev_env`` (dev200) are ever touched by default.
    ``train_monitor=True`` uses the same official-train environment for both, with
    monitor QIDs a subset of training QIDs; monitor reward is explicitly in-sample.
    The official validation split is not an input here and is never read.

    Each arm gets its own checkpoint directory ``<run_root>/E06_rl_rlcd`` / ``E07_rl_ce`` and
    its own seed, and an identity derived from the actual frozen extractor/env/prior
    contracts. If a directory already exists the run fails with :class:`FileExistsError`
    unless ``resume=True``, in which case the checkpoint is re-opened only if its identity
    matches the current request; a stale checkpoint (another extractor, env or prior) is
    refused rather than extended.
    """
    if set(priors) != set(VIXSTORY_ARMS):
        raise ValueError(f"priors must provide exactly {sorted(VIXSTORY_ARMS)}; got {sorted(priors)}")
    train_qids, dev_qids = tuple(train_qids), tuple(dev_qids)
    _validate_qids(train_qids, dev_qids, train_monitor=train_monitor)
    _validate_env(train_env, train_qids, "train")
    if train_monitor:
        if dev_env is not train_env:
            raise ValueError("train monitor must use the same train-only environment")
    else:
        _validate_env(dev_env, dev_qids, "dev")

    extractor_fp = _extractor_fingerprint()
    train_env_fp = _env_fingerprint(train_env)
    dev_env_fp = _env_fingerprint(dev_env)

    run_root = Path(run_root)
    plan: list[dict[str, Any]] = []
    for arm in VIXSTORY_ARMS:
        seed = int((arm_seeds or {}).get(arm, stable_seed(config.seed, "vixstory", arm)))
        prior_ref = (prior_refs or {}).get(arm)
        prior_fp = _prior_fingerprint(train_env, train_qids, priors[arm])
        identity = _arm_identity(
            arm=arm, seed=seed, config=config, train_qids=train_qids, dev_qids=dev_qids,
            extractor_fp=extractor_fp, train_env_fp=train_env_fp, dev_env_fp=dev_env_fp,
            prior_fp=prior_fp, prior_ref=prior_ref)
        identity = sha256_json({"arm_identity": identity, "fixed_updates": fixed_updates,
                                "train_monitor": train_monitor})
        ckpt = run_root / f"{ARM_EXPERIMENT[arm]}_rl_{arm}"
        if ckpt.exists() and not resume:
            raise FileExistsError(
                f"checkpoint directory already exists: {ckpt}; pass resume=True to continue it "
                "(or remove it) — refusing to overwrite a trained arm")
        plan.append({"arm": arm, "seed": seed, "prior_ref": prior_ref, "identity": identity,
                     "ckpt": ckpt})

    results: dict[str, VixStoryArmResult] = {}
    for entry in plan:
        trainer = VixStoryReinforceTrainer(
            train_env, dev_env, config, arm=entry["arm"], identity=entry["identity"],
            laya_fn=priors[entry["arm"]], prior_ref=entry["prior_ref"], train_qids=train_qids,
            device=device, run_dir=entry["ckpt"], seed=entry["seed"], dev_qids=dev_qids,
            fixed_updates=fixed_updates, train_monitor=train_monitor)
        status = "TRAINED"
        if entry["ckpt"].exists():
            trainer.resume_from(entry["ckpt"], identity=entry["identity"])
            status = "RESUMED"
        result = trainer.train()
        trainer.persist(entry["ckpt"])
        results[entry["arm"]] = VixStoryArmResult(
            arm=entry["arm"], experiment=ARM_EXPERIMENT[entry["arm"]], status=status,
            checkpoint_dir=str(entry["ckpt"]), seed=entry["seed"],
            n_updates=len(trainer.history),
            best_val_reward=result["best_val_reward"], best_update=result["best_update"],
            stopped_early=bool(result["stopped_early"]))
    return results
