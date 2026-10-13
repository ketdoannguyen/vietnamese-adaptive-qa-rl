"""Parser contract for captured Laya training logs (``src/eval/reporting.py``).

The parser is what ``scripts/vixstory_full_train.py`` summarises the training loss with, so
its contract is exact: only real ``epoch e/E step s loss l`` lines become points, everything
else in the captured stdout is ignored.
"""

from __future__ import annotations

from src.eval import reporting


def test_laya_loss_parser_extracts_real_lines_and_ignores_others():
    laya_log = ("Fetching 5 files: 100%|████| 5/5 [00:00<00:00, 1369.26it/s]\n"
                "epoch 1/2 step 100 loss 1.3838\n"
                "epoch 1/2 step 200 loss 0.0000\n"
                "epoch 2/2 step 100 loss -0.5\n")
    assert reporting.parse_laya_losses(laya_log) == [(100, 1.3838), (200, 0.0), (100, -0.5)]
    # a line without the full "epoch e/E step s loss l" shape is not a point
    assert reporting.parse_laya_losses("epoch 1/2 step 100 loss 0.1 extra\n") == []
    assert reporting.parse_laya_losses("") == []
