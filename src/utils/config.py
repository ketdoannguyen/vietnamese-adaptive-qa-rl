"""Config loading: base.yaml + profile overlay, with runtime pinning support."""

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


def resolve(cfg: dict, key: str) -> Path:
    """Resolve a path from cfg['paths'][key] against the repo root."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else REPO_ROOT / p


def snapshot_config(cfg: dict, out_path: str | Path) -> str:
    """Write the fully-resolved config next to a run and return its sha256."""
    import json
    from src.utils.hashing import sha256_text
    text = json.dumps(cfg, ensure_ascii=False, sort_keys=True, indent=2)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text + "\n", encoding="utf-8")
    return sha256_text(text)


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
