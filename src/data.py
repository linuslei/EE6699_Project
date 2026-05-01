"""Synthetic 1D regression datasets.

Three target functions of increasing difficulty:
    - sin_low:  y = sin(2 pi x)        (smooth, low frequency)
    - sin_high: y = sin(8 pi x)        (smooth, high frequency)
    - abs:      y = |x|                (continuous, non-differentiable at 0)

x is sampled uniformly on [-1, 1] for training.
For the test set we use a deterministic linspace on [-1, 1] so plots are clean
and test loss is a consistent estimator across runs.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Tuple

import torch


# --- Target functions -------------------------------------------------------

def target_sin_low(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(2.0 * math.pi * x)


def target_sin_high(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(8.0 * math.pi * x)


def target_abs(x: torch.Tensor) -> torch.Tensor:
    return torch.abs(x)


TARGETS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "sin_low": target_sin_low,
    "sin_high": target_sin_high,
    "abs": target_abs,
}


# --- Dataset construction ---------------------------------------------------

def make_dataset(
    target_name: str,
    n_train: int,
    n_test: int,
    seed: int,
    device: torch.device | str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate (x_train, y_train, x_test, y_test) for a named target.

    All tensors are float32, shape (n, 1), on `device`.

    The training inputs are random Uniform[-1, 1] using a *local* generator
    seeded with `seed` (we do not mutate the global RNG here).
    The test inputs are a deterministic linspace on [-1, 1].
    """
    if target_name not in TARGETS:
        raise KeyError(
            f"Unknown target '{target_name}'. Choices: {list(TARGETS)}"
        )
    f = TARGETS[target_name]

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    x_train = (torch.rand((n_train, 1), generator=gen) * 2.0 - 1.0).to(
        device=device, dtype=torch.float32
    )
    y_train = f(x_train)

    x_test = torch.linspace(-1.0, 1.0, n_test, dtype=torch.float32).unsqueeze(1).to(device)
    y_test = f(x_test)

    # Sanity checks
    assert x_train.shape == (n_train, 1), x_train.shape
    assert y_train.shape == (n_train, 1), y_train.shape
    assert x_test.shape == (n_test, 1), x_test.shape
    assert y_test.shape == (n_test, 1), y_test.shape

    return x_train, y_train, x_test, y_test
