"""Simple recursive neural model used for HFCTM-II benchmarks."""

from __future__ import annotations

import torch
from torch import nn


class RecursiveModel(nn.Module):
    """A toy model that applies a linear layer recursively."""

    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, depth: int) -> torch.Tensor:
        if depth <= 0:
            return x
        return self.forward(torch.relu(self.layer(x)), depth - 1)


def build_recursive_model(hidden_size: int = 16) -> RecursiveModel:
    model = RecursiveModel(hidden_size)
    return model
