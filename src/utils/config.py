"""Config loading: ``configs/base.yaml`` + named profile overlay.

Retained for the VixSTORY V1 tooling: ``scripts/vixstory_pilot_laya.py`` reads the ``laya``
block and the named data profile through here. An explicit ``--config`` path is deep-merged
on top of the profile.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from src.utils.io import read_yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(profile: str | None = None, *, extra: dict | None = None) -> dict:
    """Load ``configs/base.yaml``, overlay ``configs/<profile>.yaml`` if present."""
    cfg = read_yaml(CONFIG_DIR / "base.yaml")
    if profile is None:
        profile = cfg["data"]["profile"]
    profile_path = CONFIG_DIR / f"{profile}.yaml"
    if profile_path.exists():
        cfg = _deep_merge(cfg, read_yaml(profile_path))
    if profile != "full" and profile_path.exists():
        cfg["data"]["profile"] = profile
    if extra:
        cfg = _deep_merge(cfg, extra)
    return cfg


def add_profile_args(parser: Any) -> None:
    parser.add_argument("--profile", default=None,
                        choices=["smoke", "pilot", "small", "medium", "full"],
                        help="data scale profile (defaults to base.yaml)")
    parser.add_argument("--config", default=None, help="path to an extra YAML overlay")
    parser.add_argument("--seed", type=int, default=None, help="override global seed")


def config_from_args(args: Any) -> dict:
    extra = read_yaml(args.config) if getattr(args, "config", None) else None
    cfg = load_config(getattr(args, "profile", None), extra=extra)
    if getattr(args, "seed", None) is not None:
        cfg["seed"] = args.seed
    return cfg
