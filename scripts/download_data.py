#!/usr/bin/env python
"""Thin wrapper so the pinned download can be started as ``python scripts/download_data.py``.

The implementation lives in :mod:`src.pipeline.download` (importable, testable); this file
only puts the repo root on ``sys.path`` for a direct script invocation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.download import main  # noqa: E402  (path fix must run first)

if __name__ == "__main__":
    raise SystemExit(main())
