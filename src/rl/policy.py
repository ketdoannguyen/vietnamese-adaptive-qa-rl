"""Shared MLP policy head trained by REINFORCE.

The network chooses actions from visible features plus the frozen Laya prior rather
than modeling language. Laya stays frozen here; nothing in this file backpropagates
into it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

N_ACTIONS = 4


class PolicyNet(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 128, n_actions: int = N_ACTIONS) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.Tanh(),
            nn.Linear(d_hidden, d_hidden), nn.Tanh(),
        )
        self.policy_logits = nn.Linear(d_hidden, n_actions)
        self.value = nn.Linear(d_hidden, 1)
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.n_actions = n_actions

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        return self.policy_logits(h), self.value(h).squeeze(-1)

    def config(self) -> dict[str, Any]:
        return {"d_in": self.d_in, "d_hidden": self.d_hidden, "n_actions": self.n_actions}

    def save(self, path: str | Path) -> None:
        """Save as safetensors when available, else a plain local state_dict.

        Rule: no pickled model objects from untrusted sources. This file is ours.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().cpu() for k, v in self.state_dict().items()}
        try:
            from safetensors.torch import save_file
            save_file(state, str(path))
        except Exception:
            torch.save(state, path)
        path.with_suffix(path.suffix + ".config.json").write_text(
            json.dumps(self.config(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PolicyNet":
        path = Path(path)
        cfg = json.loads(path.with_suffix(path.suffix + ".config.json").read_text(encoding="utf-8"))
        net = cls(cfg["d_in"], cfg["d_hidden"], cfg["n_actions"])
        try:
            from safetensors.torch import load_file
            state = load_file(str(path))
        except Exception:
            state = torch.load(path, map_location="cpu", weights_only=True)
        net.load_state_dict(state)
        return net


class TorchScaler:
    """The feature standardiser as torch buffers, so transform is part of the graph input."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, clip: float = 10.0) -> None:
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)
        self.clip = clip

    def to(self, device: str | torch.device) -> "TorchScaler":
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = (x - self.mean) / self.std
        return torch.clamp(out, -self.clip, self.clip)
