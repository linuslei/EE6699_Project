"""Sweep runner: loops over (target, width, seed) and writes results to disk.

Usage:
    python -m src.run_experiments --config smoke
    python -m src.run_experiments --config full
    python -m src.run_experiments --config full --out-dir results

Outputs (under results/raw/):
    summary_<config_name>.csv
    curves_<config_name>.csv
    predictions_<config_name>.json
    config_<config_name>.json

Configs are Python dicts declared inline below for simplicity.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from .train import train_one_model


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

# Common training knobs shared by both configs.
_BASE: Dict[str, Any] = {
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "activation": "relu",
    "n_train": 100,
    "n_test": 1000,
    "device": "auto",
    "log_every": 50,
}

SMOKE_CONFIG: Dict[str, Any] = {
    "name": "smoke",
    "targets": ["sin_low"],
    "widths": [10, 100],
    "seeds": [0],
    "epochs": 500,
    **_BASE,
}

FULL_CONFIG: Dict[str, Any] = {
    "name": "full",
    "targets": ["sin_low", "sin_high", "abs"],
    "widths": [10, 50, 100, 500, 1000],
    "seeds": [0, 1, 2, 3, 4],
    "epochs": 5000,
    **_BASE,
}

CONFIGS: Dict[str, Dict[str, Any]] = {
    "smoke": SMOKE_CONFIG,
    "full": FULL_CONFIG,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return ""


def _select_prediction_runs(widths: List[int], seeds: List[int]) -> set:
    """Decide which (width, seed) pairs to persist predictions for.

    To keep predictions JSON small we save only:
        - smallest width, seed=seeds[0]
        - largest  width, seed=seeds[0]
    per target. (Plenty for the "learned function" plots.)
    """
    if not widths or not seeds:
        return set()
    s0 = seeds[0]
    return {(min(widths), s0), (max(widths), s0)}


def _per_run_config(cfg: Dict[str, Any], target: str, width: int, seed: int) -> Dict[str, Any]:
    return {
        "target_name": target,
        "width": int(width),
        "seed": int(seed),
        "n_train": cfg["n_train"],
        "n_test": cfg["n_test"],
        "optimizer": cfg["optimizer"],
        "learning_rate": cfg["learning_rate"],
        "epochs": cfg["epochs"],
        "activation": cfg["activation"],
        "device": cfg["device"],
        "log_every": cfg["log_every"],
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(cfg: Dict[str, Any], out_dir: Path) -> Dict[str, Path]:
    name = cfg["name"]
    targets: List[str] = cfg["targets"]
    widths: List[int] = cfg["widths"]
    seeds: List[int] = cfg["seeds"]

    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    curves_rows: List[Dict[str, Any]] = []
    pred_runs: List[Dict[str, Any]] = []
    pred_keys = _select_prediction_runs(widths, seeds)
    pred_x_test: List[float] | None = None

    combos: List[Tuple[str, int, int]] = list(product(targets, widths, seeds))
    expected = len(targets) * len(widths) * len(seeds)
    assert len(combos) == expected

    pbar = tqdm(combos, desc=f"sweep[{name}]", unit="run")
    for target, width, seed in pbar:
        run_cfg = _per_run_config(cfg, target, width, seed)
        pbar.set_postfix(target=target, m=width, seed=seed)
        out = train_one_model(run_cfg)

        summary_rows.append(out["summary"])
        for c in out["curves"]:
            curves_rows.append({
                "target_name": target,
                "width": int(width),
                "seed": int(seed),
                "epoch": int(c["epoch"]),
                "train_loss": float(c["train_loss"]),
                "test_loss": float(c["test_loss"]),
            })

        if (width, seed) in pred_keys:
            x_test_np, y_pred_np = out["predictions"]
            if pred_x_test is None:
                pred_x_test = x_test_np.tolist()
            from .data import TARGETS
            import torch as _torch
            y_true_np = TARGETS[target](_torch.from_numpy(x_test_np)).numpy()
            pred_runs.append({
                "target_name": target,
                "width": int(width),
                "seed": int(seed),
                "y_true": y_true_np.tolist(),
                "y_pred": y_pred_np.tolist(),
            })

    # --- Write outputs -----------------------------------------------------
    summary_path = raw_dir / f"summary_{name}.csv"
    curves_path = raw_dir / f"curves_{name}.csv"
    preds_path = raw_dir / f"predictions_{name}.json"
    config_path = raw_dir / f"config_{name}.json"

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(curves_rows).to_csv(curves_path, index=False)
    with open(preds_path, "w") as f:
        json.dump({"x_test": pred_x_test or [], "runs": pred_runs}, f)
    with open(config_path, "w") as f:
        json.dump(
            {
                "config": cfg,
                "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
                "git_sha": _git_sha(),
            },
            f,
            indent=2,
        )

    # Post-run sanity assertions
    assert len(summary_rows) == expected, (len(summary_rows), expected)
    for p in (summary_path, curves_path, preds_path, config_path):
        assert p.exists(), f"missing output: {p}"

    print(f"[done] wrote: {summary_path}")
    print(f"[done] wrote: {curves_path}")
    print(f"[done] wrote: {preds_path}")
    print(f"[done] wrote: {config_path}")
    return {
        "summary": summary_path,
        "curves": curves_path,
        "predictions": preds_path,
        "config": config_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run width × target × seed sweep.")
    parser.add_argument(
        "--config",
        choices=sorted(CONFIGS.keys()),
        default="smoke",
        help="Which inline config to run.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results"),
        help="Output root directory (CSV/JSON go under <out-dir>/raw/).",
    )
    args = parser.parse_args()

    cfg = CONFIGS[args.config]
    print(f"[config={cfg['name']}] targets={cfg['targets']} widths={cfg['widths']} "
          f"seeds={cfg['seeds']} epochs={cfg['epochs']}")
    run_sweep(cfg, args.out_dir)


if __name__ == "__main__":
    main()
