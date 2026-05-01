"""Metrics for the lazy-training study.

The headline quantity is the *relative* L2 movement of the parameter vector
from initialization:

    relative_movement = || theta_final - theta_initial || / || theta_initial ||

We flatten *all* trainable parameters (weights AND biases of both layers)
into one vector. Snapshotting and the final flatten must be consistent.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def flatten_params(model: nn.Module) -> torch.Tensor:
    """Return a 1D float tensor (on CPU, detached, cloned) of all parameters.

    We move to CPU and clone so the snapshot is independent of subsequent
    in-place updates and of the training device.
    """
    parts = [p.detach().to("cpu").reshape(-1).clone() for p in model.parameters()]
    return torch.cat(parts) if parts else torch.empty(0)


def snapshot_initial_params(model: nn.Module) -> torch.Tensor:
    """Take a snapshot of parameters before training. Just a named alias."""
    return flatten_params(model)


def relative_weight_movement(
    model: nn.Module,
    initial_flat: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """|| theta_final - theta_initial || / max(|| theta_initial ||, eps)."""
    final_flat = flatten_params(model)
    if final_flat.shape != initial_flat.shape:
        raise ValueError(
            f"Param shapes differ: initial {initial_flat.shape} vs final {final_flat.shape}"
        )
    init_norm = torch.linalg.norm(initial_flat).item()
    diff_norm = torch.linalg.norm(final_flat - initial_flat).item()
    rel = diff_norm / max(init_norm, eps)
    if not math.isfinite(rel):
        raise FloatingPointError(f"Non-finite relative_movement: {rel}")
    return float(rel)


@torch.no_grad()
def mse_loss_on_dataset(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """Mean squared error of `model` on (x, y). Returns a Python float."""
    was_training = model.training
    model.eval()
    try:
        pred = model(x)
        loss = torch.mean((pred - y) ** 2).item()
    finally:
        if was_training:
            model.train()
    return float(loss)
