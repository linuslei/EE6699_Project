"""Two-layer feedforward network used for all experiments.

    input (1) -> Linear(1, m) -> activation -> Linear(m, 1) -> output (1)

Default activation is ReLU. `tanh` is wired in as well for the optional
Experiment 3 (activation comparison).

We deliberately keep the architecture and initialization standard
(PyTorch defaults) so the qualitative lazy-training trend is observed
under a familiar setup. The README documents the caveat that this is not
strict NTK parameterization.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


_ACTIVATIONS: Dict[str, type] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}


class TwoLayerNet(nn.Module):
    def __init__(self, width: int, activation: str = "relu") -> None:
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise KeyError(
                f"Unknown activation '{activation}'. Choices: {list(_ACTIVATIONS)}"
            )
        if width < 1:
            raise ValueError(f"width must be >= 1, got {width}")

        self.width = int(width)
        self.activation_name = activation

        self.fc1 = nn.Linear(1, self.width)
        self.act = _ACTIVATIONS[activation]()
        self.fc2 = nn.Linear(self.width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
