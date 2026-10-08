"""Determinism helpers.

Every stochastic component in this project derives its randomness either from
``seed_everything`` or from ``numpy.random.Generator`` instances created with a
seed computed by :func:`src.utils.hashing.stable_seed`. Python's ``hash()`` and
unseeded global RNGs are forbidden in code paths that produce persisted outputs.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, *, deterministic_torch: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def rng_from_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def capture_library_versions() -> dict:
    versions = {"python": None}
    import platform
    versions["python"] = platform.python_version()
    for mod in ("torch", "transformers", "datasets", "numpy", "sentence_transformers",
                "rank_bm25", "sklearn", "gymnasium", "pandas", "pyarrow"):
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            versions[mod] = None
    try:
        import torch
        if torch.cuda.is_available():
            versions["cuda_runtime"] = torch.version.cuda
            versions["cudnn"] = torch.backends.cudnn.version()
    except Exception:
        pass
    return versions
