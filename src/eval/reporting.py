"""Training-log parsing shared with the VixStory full-train stage.

``scripts/vixstory_full_train.py`` summarises the captured Laya training log through
:func:`parse_laya_losses`; nothing here aggregates, plots or reshapes results.
"""

from __future__ import annotations

import re

#: The Laya trainer's per-step log line, e.g. ``epoch 1/2 step 13000 loss 0.0009``.
LAYA_LOSS_RE = re.compile(r"^epoch (\d+)/(\d+) step (\d+) loss (-?[\d.eE+]+)\s*$")


def parse_laya_losses(text: str) -> list[tuple[int, float]]:
    """``(step, loss)`` pairs from a captured Laya training log, in log order."""
    out: list[tuple[int, float]] = []
    for line in text.splitlines():
        match = LAYA_LOSS_RE.match(line.strip())
        if match:
            out.append((int(match.group(3)), float(match.group(4))))
    return out
