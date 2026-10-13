"""The head-only variant must freeze the complete encoder and train only the decision head."""

import pytest
import torch
from torch import nn

from scripts.vixstory_v2_train_laya import (
    BUCKET,
    BucketedRandom,
    freeze_all_encoder_blocks,
    frozen_parameter_digest,
)


class FakeLaya(nn.Module):
    def __init__(self, encoder_layers=22, head_layers=2):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.embeddings = nn.Linear(2, 2)
        self.encoder.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(encoder_layers)])
        self.encoder.final_norm = nn.LayerNorm(2)
        self.head = nn.Module()
        self.head.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(head_layers)])
        self.scorer = nn.Linear(2, 1)


def test_only_head_trainable():
    model = FakeLaya()
    info = freeze_all_encoder_blocks(model)
    assert info["encoder_blocks_trainable"] == []
    assert info["head_blocks_trainable"] == 2
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == (not name.startswith("encoder.")), name
    before = frozen_parameter_digest(model)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.01)
    loss = sum(parameter.square().sum() for parameter in model.parameters()
               if parameter.requires_grad)
    loss.backward()
    optimizer.step()
    assert frozen_parameter_digest(model) == before


@pytest.mark.parametrize("encoder_layers,head_layers", [(21, 2), (22, 1)])
def test_rejects_incompatible_checkpoint_shapes(encoder_layers, head_layers):
    with pytest.raises(RuntimeError, match="22-block ModernBERT and 2-block Laya head"):
        freeze_all_encoder_blocks(FakeLaya(encoder_layers, head_layers))


def test_bucketed_shuffle_groups_similar_lengths_deterministically():
    items = [{"state_ids": list(range(n))} for n in range(128)]
    order = list(items)
    BucketedRandom(7).shuffle(order)
    assert sorted(len(item["state_ids"]) for item in order) == list(range(128))
    for start in range(0, 128, 4):
        lengths = [len(item["state_ids"]) for item in order[start:start + 4]]
        assert max(lengths) - min(lengths) < BUCKET
    again = list(items)
    BucketedRandom(7).shuffle(again)
    assert again == order


def test_bucketed_shuffle_falls_back_for_plain_lists():
    values = list(range(20))
    BucketedRandom(3).shuffle(values)
    assert sorted(values) == list(range(20))
    assert values != list(range(20))
