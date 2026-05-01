"""Plotting from saved CSV/JSON. No torch dependency on this path.

Run after `run_experiments.py` has produced files in `results/raw/`:

    python -m src.plots --config smoke
    python -m src.plots --config full

Figures are written to `results/figures/` as PNG and PDF.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _load(raw_dir: Path, name: str):
    summary = pd.read_csv(raw_dir / f"summary_{name}.csv")
    curves = pd.read_csv(raw_dir / f"curves_{name}.csv")
    with open(raw_dir / f"predictions_{name}.json") as f:
        preds = json.load(f)
    return summary, curves, preds


def _save(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out_path.with_suffix('.png')}")


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------

def _agg_by_width(summary: pd.DataFrame, value_col: str) -> pd.DataFrame:
    g = summary.groupby(["target_name", "width"])[value_col]
    return g.agg(["mean", "std", "count"]).reset_index()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_width_vs_metric(
    summary: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
    log_y: bool = False,
) -> None:
    agg = _agg_by_width(summary, value_col)
    fig, ax = plt.subplots(figsize=(6, 4))
    for target in sorted(agg["target_name"].unique()):
        sub = agg[agg["target_name"] == target].sort_values("width")
        std = sub["std"].fillna(0.0).values
        ax.errorbar(
            sub["width"].values,
            sub["mean"].values,
            yerr=std,
            marker="o",
            capsize=3,
            label=target,
        )
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("width m (log scale)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(title="target")
    _save(fig, out_path)


def plot_training_curves(
    curves: pd.DataFrame,
    target: str,
    widths: Iterable[int],
    seed: int,
    out_path: Path,
) -> None:
    widths = list(widths)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for w in widths:
        sub = curves[
            (curves["target_name"] == target)
            & (curves["width"] == w)
            & (curves["seed"] == seed)
        ].sort_values("epoch")
        if sub.empty:
            continue
        axes[0].plot(sub["epoch"], sub["train_loss"], label=f"m={w}")
        axes[1].plot(sub["epoch"], sub["test_loss"], label=f"m={w}")
    for ax, ttl in zip(axes, ["train loss", "test loss"]):
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel(ttl)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend(title="width")
    fig.suptitle(f"Training curves — target={target}, seed={seed}")
    fig.tight_layout()
    _save(fig, out_path)


def plot_learned_function(
    preds: dict,
    target: str,
    widths: Iterable[int],
    seed: int,
    out_path: Path,
) -> None:
    x_test = np.asarray(preds.get("x_test", []), dtype=float)
    runs = preds.get("runs", [])
    if x_test.size == 0:
        print(f"[skip] no x_test in predictions for target={target}")
        return

    matching = [
        r for r in runs
        if r["target_name"] == target and r["seed"] == seed and r["width"] in set(widths)
    ]
    if not matching:
        print(f"[skip] no matching prediction runs for target={target}, seed={seed}, widths={list(widths)}")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    # True curve (take from any matching run)
    y_true = np.asarray(matching[0]["y_true"], dtype=float)
    ax.plot(x_test, y_true, "k--", linewidth=2, label="true f(x)")
    for r in sorted(matching, key=lambda r: r["width"]):
        y_pred = np.asarray(r["y_pred"], dtype=float)
        ax.plot(x_test, y_pred, label=f"m={r['width']}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Learned vs true — target={target}, seed={seed}")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def make_all_figures(name: str, raw_dir: Path, fig_dir: Path) -> None:
    summary, curves, preds = _load(raw_dir, name)

    # 1-3: width vs metric
    plot_width_vs_metric(
        summary, "relative_movement",
        ylabel="relative weight movement",
        title="Lazy training: relative movement vs width",
        out_path=fig_dir / f"fig_width_vs_movement_{name}",
        log_y=True,
    )
    plot_width_vs_metric(
        summary, "final_train_loss",
        ylabel="final train loss (MSE)",
        title="Final train loss vs width",
        out_path=fig_dir / f"fig_width_vs_train_loss_{name}",
        log_y=True,
    )
    plot_width_vs_metric(
        summary, "final_test_loss",
        ylabel="final test loss (MSE)",
        title="Final test loss vs width",
        out_path=fig_dir / f"fig_width_vs_test_loss_{name}",
        log_y=True,
    )

    # 4-5: per-target curves and learned functions
    targets: List[str] = sorted(summary["target_name"].unique())
    seed0 = int(sorted(summary["seed"].unique())[0])
    widths_all = sorted(summary["width"].unique())
    # pick a few widths for curves: first, middle, last
    if len(widths_all) >= 3:
        sel_widths = [widths_all[0], widths_all[len(widths_all) // 2], widths_all[-1]]
    else:
        sel_widths = widths_all
    for t in targets:
        plot_training_curves(
            curves, target=t, widths=sel_widths, seed=seed0,
            out_path=fig_dir / f"fig_curves_{t}_{name}",
        )
        # learned-function plot uses whatever (width, seed) pairs the runner saved
        plot_learned_function(
            preds, target=t, widths=widths_all, seed=seed0,
            out_path=fig_dir / f"fig_learned_{t}_{name}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate figures from saved results.")
    parser.add_argument(
        "--config",
        default="smoke",
        help="Config name to plot (matches the filename suffix used by run_experiments.py).",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    raw_dir = args.results_dir / "raw"
    fig_dir = args.results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    make_all_figures(args.config, raw_dir, fig_dir)


if __name__ == "__main__":
    main()
