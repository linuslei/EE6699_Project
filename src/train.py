"""Single-run training loop.

`train_one_model(config)` performs one independent run and returns:
    {
        "summary":     {one-row dict matching summary CSV columns},
        "curves":      [{"epoch":..., "train_loss":..., "test_loss":...}, ...],
        "predictions": (x_test_np, y_pred_np)   # for plotting; runner decides
                                                # whether to persist these
    }

Doing no disk I/O here keeps the function easy to test and reuse.
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from .data import make_dataset
from .metrics import mse_loss_on_dataset, relative_weight_movement, snapshot_initial_params
from .model import TwoLayerNet


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA if present).

    Note: this gives reproducible *qualitative* results. Bit-identical
    reproducibility across devices (CPU vs CUDA vs MPS) is not guaranteed
    and is not required for this study.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(preferred: str = "auto") -> torch.device:
    """Auto-select CUDA -> MPS -> CPU unless `preferred` overrides."""
    if preferred and preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_optimizer(name: str, params, lr: float) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    raise KeyError(f"Unknown optimizer '{name}'. Choices: 'adam', 'sgd'.")


def train_one_model(config: Dict[str, Any]) -> Dict[str, Any]:
    """Train one model end-to-end and return summary + curves + predictions.

    Required config keys:
        target_name, width, seed, n_train, n_test,
        optimizer, learning_rate, epochs, activation,
        device, log_every

    The run is fully self-contained: it seeds RNGs, builds data and model,
    snapshots initial parameters, trains, and computes metrics.
    """
    required = {
        "target_name", "width", "seed", "n_train", "n_test",
        "optimizer", "learning_rate", "epochs", "activation",
        "device", "log_every",
    }
    missing = required - set(config)
    if missing:
        raise KeyError(f"Missing config keys: {sorted(missing)}")

    set_seed(config["seed"])
    device = pick_device(config["device"])

    x_train, y_train, x_test, y_test = make_dataset(
        target_name=config["target_name"],
        n_train=int(config["n_train"]),
        n_test=int(config["n_test"]),
        seed=int(config["seed"]),
        device=device,
    )

    model = TwoLayerNet(
        width=int(config["width"]),
        activation=str(config["activation"]),
    ).to(device)

    init_flat = snapshot_initial_params(model)
    optim = _build_optimizer(
        config["optimizer"], model.parameters(), float(config["learning_rate"])
    )

    epochs = int(config["epochs"])
    log_every = max(1, int(config["log_every"]))
    curves: List[Dict[str, float]] = []

    # Log epoch 0 (pre-training) loss
    curves.append({
        "epoch": 0,
        "train_loss": mse_loss_on_dataset(model, x_train, y_train),
        "test_loss": mse_loss_on_dataset(model, x_test, y_test),
    })
    initial_train_loss = curves[0]["train_loss"]

    t0 = time.perf_counter()
    model.train()
    for epoch in range(1, epochs + 1):
        optim.zero_grad(set_to_none=True)
        pred = model(x_train)
        loss = torch.mean((pred - y_train) ** 2)
        loss.backward()
        optim.step()

        if epoch % log_every == 0 or epoch == epochs:
            curves.append({
                "epoch": epoch,
                "train_loss": mse_loss_on_dataset(model, x_train, y_train),
                "test_loss": mse_loss_on_dataset(model, x_test, y_test),
            })
    runtime_seconds = time.perf_counter() - t0

    final_train_loss = mse_loss_on_dataset(model, x_train, y_train)
    final_test_loss = mse_loss_on_dataset(model, x_test, y_test)
    rel_move = relative_weight_movement(model, init_flat)

    # Soft sanity warning: did the loss actually go down?
    if final_train_loss >= 0.9 * max(initial_train_loss, 1e-12):
        print(
            f"[WARN] target={config['target_name']} width={config['width']} "
            f"seed={config['seed']}: train loss barely changed "
            f"({initial_train_loss:.4g} -> {final_train_loss:.4g}). "
            "Consider more epochs or different LR."
        )

    with torch.no_grad():
        y_pred = model(x_test).detach().to("cpu").reshape(-1).numpy()
    x_test_np = x_test.detach().to("cpu").reshape(-1).numpy()

    summary = {
        "target_name": config["target_name"],
        "width": int(config["width"]),
        "seed": int(config["seed"]),
        "optimizer": config["optimizer"],
        "learning_rate": float(config["learning_rate"]),
        "epochs": epochs,
        "activation": config["activation"],
        "n_train": int(config["n_train"]),
        "n_test": int(config["n_test"]),
        "final_train_loss": float(final_train_loss),
        "final_test_loss": float(final_test_loss),
        "relative_movement": float(rel_move),
        "runtime_seconds": float(runtime_seconds),
    }

    return {
        "summary": summary,
        "curves": curves,
        "predictions": (x_test_np, y_pred),
    }


if __name__ == "__main__":
    # Tiny self-check: one short run, prints summary.
    cfg = {
        "target_name": "sin_low",
        "width": 50,
        "seed": 0,
        "n_train": 100,
        "n_test": 1000,
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "epochs": 200,
        "activation": "relu",
        "device": "auto",
        "log_every": 50,
    }
    out = train_one_model(cfg)
    print("summary:", out["summary"])
    print("curves[0]:", out["curves"][0])
    print("curves[-1]:", out["curves"][-1])
