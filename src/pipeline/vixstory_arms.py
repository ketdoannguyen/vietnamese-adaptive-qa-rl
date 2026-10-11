"""VixSTORY pilot policies over the public observation and pinned local checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.decision.laya_adapter import LayaAdapter
from src.decision.schema import ACTION_ORDER
from src.decision.vixstory_state_text import STATE_TEXT_VERSION, serialize_vixstory_state
from src.pipeline.vixstory_calibration import VixStoryCorrectnessEstimator, e03_action
from src.rl.vixstory_env import ABSTAIN, ANSWER, SEARCH_BM25, SEARCH_DENSE, VixStoryFeatureExtractor
from src.utils.hashing import sha256_json, stable_seed


def terminal(obs: Mapping[str, Any]) -> int:
    return ANSWER if obs["action_mask"][ANSWER] else ABSTAIN


class TerminalPolicy:
    def act(self, obs: Mapping[str, Any]) -> int:
        return terminal(obs)


class AbstainPolicy:
    def act(self, obs: Mapping[str, Any]) -> int:
        return ABSTAIN


class FixedPolicy:
    def __init__(self, action: int) -> None:
        if action not in (SEARCH_BM25, SEARCH_DENSE):
            raise ValueError("fixed policy requires one SEARCH action")
        self.action = action

    def act(self, obs: Mapping[str, Any]) -> int:
        return self.action if obs["action_mask"][self.action] else terminal(obs)


class RandomValidPolicy:
    def begin_episode(self, *, qid: str, seed: int) -> None:
        self.rng = np.random.default_rng(stable_seed("vixstory-random", seed, qid))

    def act(self, obs: Mapping[str, Any]) -> int:
        valid = np.flatnonzero(np.asarray(obs["action_mask"], dtype=bool))
        return int(self.rng.choice(valid))


class CalibrationPolicy:
    def __init__(self, estimator: VixStoryCorrectnessEstimator, tau: float) -> None:
        self.estimator, self.tau = estimator, tau

    def act(self, obs: Mapping[str, Any]) -> int:
        return e03_action(obs, self.estimator.probability_from_obs(obs), self.tau)


class LayaPrior:
    """Inference uses exactly the teacher's VixSTORY state renderer."""

    def __init__(self, checkpoint: str | Path, *, device: str = "cuda") -> None:
        from src.utils.io import read_json

        directory = Path(checkpoint)
        manifest = read_json(directory.parent / f"{directory.name}.manifest.json")
        if manifest.get("state_text_version") != STATE_TEXT_VERSION:
            raise ValueError(f"Laya checkpoint requires state text {STATE_TEXT_VERSION}")
        from src.utils.hashing import sha256_file

        teacher_path = manifest.get("data", {}).get("path")
        if not teacher_path or not Path(teacher_path).is_file():
            raise ValueError("Laya v2 teacher artifact is missing")
        teacher_sha = sha256_file(teacher_path)
        if manifest["data"]["sha256"] != teacher_sha:
            raise ValueError("Laya v2 teacher artifact changed since training")
        teacher_contract = manifest.get("teacher_contract")
        provenance_path = Path(teacher_path).with_suffix(".provenance.json")
        if not provenance_path.is_file() or read_json(provenance_path) != teacher_contract:
            raise ValueError("Laya v2 teacher provenance is missing or changed")
        from src.decision import vixstory_state_text

        if (not isinstance(teacher_contract, dict) or teacher_contract.get("sha256") != teacher_sha
                or teacher_contract.get("state_text_version") != STATE_TEXT_VERSION
                or teacher_contract.get("state_text_sha256") !=
                sha256_file(vixstory_state_text.__file__)
                or teacher_contract.get("tokenizer_contract_sha256") !=
                manifest.get("tokenizer_contract_sha256")
                or teacher_contract.get("max_len") != manifest.get("max_len")
                or teacher_contract.get("head_max_len") != manifest.get("head_max_len")):
            raise ValueError("Laya v2 teacher contract is missing or disagrees with checkpoint")
        self.adapter = LayaAdapter.load(directory, device=device)
        try:
            self.tokenizer = self.adapter._agent.tok
            self.max_len = int(self.adapter._agent.cfg.get("max_len", 512))
            self.head_max_len = int(self.adapter._agent.cfg.get("head_max_len", 192))
            contract = json.loads(self.tokenizer.backend_tokenizer.to_str())
            contract.pop("truncation", None)
            matched = (manifest.get("tokenizer_contract_sha256") == sha256_json(contract)
                       and manifest.get("max_len") == self.max_len
                       and manifest.get("head_max_len") == self.head_max_len
                       and manifest.get("output", {}).get("model_hash") == self.adapter.model_hash())
        except BaseException:
            self.adapter.close()
            raise
        if not matched:
            self.adapter.close()
            raise ValueError("Laya checkpoint identity disagrees with its v2 teacher manifest")

    def __call__(self, obs: Mapping[str, Any]) -> np.ndarray:
        text = serialize_vixstory_state(obs, tokenizer=self.tokenizer, max_len=self.max_len,
                                        head_max_len=self.head_max_len)
        return self.adapter.predict_probs([text])[0]

    def close(self) -> None:
        self.adapter.close()


class LayaPolicy:
    def __init__(self, prior: LayaPrior) -> None:
        self.prior = prior

    def act(self, obs: Mapping[str, Any]) -> int:
        probs = np.asarray(self.prior(obs), dtype=np.float64)
        mask = np.asarray(obs["action_mask"], dtype=bool)
        if probs.shape != (len(ACTION_ORDER),) or mask.shape != probs.shape:
            raise ValueError("Laya prior and observation mask must carry four actions")
        if not np.all(np.isfinite(probs)) or np.any(probs < 0) or probs.sum() <= 0:
            raise ValueError("Laya prior is not a valid probability vector")
        return int(np.argmax(np.where(mask, probs, -np.inf)))


class ReinforcePolicy:
    """Greedy inference from the actual arm checkpoint, with its own frozen Laya prior."""

    def __init__(self, checkpoint: str | Path, prior: LayaPrior, *, device: str = "cuda") -> None:
        import torch

        from src.rl.features import FeatureScaler
        from src.rl.policy import PolicyNet, TorchScaler

        directory = Path(checkpoint)
        self.net = PolicyNet.load(directory / "policy.safetensors").to(device).eval()
        scaler = FeatureScaler.load(directory / "scaler.json")
        self.scaler = TorchScaler(np.asarray(scaler.mean, dtype=np.float32),
                                  np.asarray(scaler.std, dtype=np.float32), scaler.clip).to(device)
        self.extractor = VixStoryFeatureExtractor()
        self.device = device
        self.prior = prior
        self.torch = torch

    def act(self, obs: Mapping[str, Any]) -> int:
        features = self.extractor.vector(obs, self.prior(obs))
        x = self.torch.as_tensor(features, dtype=self.torch.float32, device=self.device).unsqueeze(0)
        mask = self.torch.as_tensor(obs["action_mask"], dtype=self.torch.bool,
                                    device=self.device).unsqueeze(0)
        with self.torch.inference_mode():
            logits, _ = self.net(self.scaler(x))
        return int(logits.masked_fill(~mask, -1e9).argmax(dim=-1).item())


def basic_policies() -> dict[str, Any]:
    return {"E00": TerminalPolicy(), "E01": FixedPolicy(SEARCH_BM25),
            "E02": FixedPolicy(SEARCH_DENSE), "ABSTAIN": AbstainPolicy(),
            "RANDOM_VALID": RandomValidPolicy()}
