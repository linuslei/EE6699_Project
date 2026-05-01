# Width and Lazy Training in Neural Networks

**An experimental study of weight movement and generalization**

Final project for *Mathematics of Deep Learning*.

---

## TL;DR (for readers with little ML background)

A neural network is a function with a lot of tunable knobs (called **parameters** or **weights**). "Training" means nudging those knobs until the network's outputs match a target. There is a surprising recent discovery: when the network is made very **wide** (lots of knobs), it can learn the task while *barely moving the knobs at all* from their random starting values. This phenomenon is called **lazy training**, and it is the bridge between modern deep networks and classical kernel methods (via something called the **Neural Tangent Kernel**, or NTK).

This project asks a simple, concrete question:

> As we make a small neural network wider, how much do its weights actually move during training, and does that depend on how hard the target function is to learn?

We do this with a tiny, easy-to-visualize setup: 1D inputs, a 2-layer network, and three target functions of increasing difficulty.

---

## Background, in plain language

### What is a neural network here?
We use the simplest useful network — a **two-layer feedforward network**:

```
input x  ─►  Linear(1 → m)  ─►  ReLU  ─►  Linear(m → 1)  ─►  output ŷ
```

- `x` is a single number in `[-1, 1]`.
- `m` is the **width** — the number of hidden units. We will sweep `m` over `{10, 50, 100, 500, 1000}`.
- `ReLU(z) = max(0, z)` is a standard nonlinearity.
- The network's job: given training pairs `(x, y)`, learn to output `ŷ ≈ y`.

### What is "training"?
We collect all the network's parameters into one long vector `θ`. We pick a **loss function** — here, mean squared error `MSE = mean((ŷ − y)²)` — and use an optimizer (Adam) to repeatedly adjust `θ` so the loss goes down. After many steps (epochs), training stops; call the final parameters `θ_final` and the random starting parameters `θ_initial`.

### What is "lazy training"?
For very wide networks, theory and experiments suggest that during training the parameters move only a little — in a precise sense, the **relative movement**

$$\text{relative\_movement} \;=\; \frac{\lVert \theta_{\text{final}} - \theta_{\text{initial}} \rVert}{\lVert \theta_{\text{initial}} \rVert}$$

tends to **shrink as width `m` grows**, even though the network still fits the data well. In this regime the network behaves almost like a fixed linear model on top of features defined at initialization — that fixed kernel is the **Neural Tangent Kernel (NTK)**. The "lazy" name comes from this: the network learns without working hard to change itself.

If that is not yet intuitive, the one-sentence version is: *wide networks can be smart without moving much*.

### Why study this?
- It connects modern neural nets to classical, well-understood **kernel methods**.
- It helps explain why huge over-parameterized networks generalize at all (a major puzzle in deep learning theory).
- It also has limits: laziness is easier to achieve for "easy" target functions and may break down for harder ones. That limit is exactly what our extension tests.

---

## Research questions

For 2-layer ReLU networks of varying width `m`, trained on synthetic 1D regression data:

1. How does **width** affect the **training loss**?
2. How does **width** affect the **test loss** (generalization)?
3. How does **width** affect the **relative weight movement** from initialization?
4. **(Extension)** Do the answers depend on how hard the target function is?

---

## Experiments

### Experiment 1 — Reproduction: lazy training shows up as width grows
- **Target function:** `y = sin(2πx)` (smooth, low-frequency, easy).
- **Sweep:** widths `m ∈ {10, 50, 100, 500, 1000}`, with **5 random seeds** per width.
- **Measure:** final train loss, final test loss, and relative weight movement.
- **Expected qualitative trend:** as `m` grows, the network still fits the data, but the **relative movement decreases** — the lazy-training signature.

### Experiment 2 — Extension: does it depend on the target?
Repeat the same width sweep for three targets of increasing difficulty:

| Name        | Function       | Why it matters                          |
| ----------- | -------------- | --------------------------------------- |
| `sin_low`   | `sin(2πx)`     | smooth, low-frequency (easy)            |
| `sin_high`  | `sin(8πx)`     | smooth but high-frequency (harder)      |
| `abs`       | `|x|`          | continuous but **not differentiable** at 0 (different kind of hard) |

**Hypothesis.** Harder targets (high-frequency or non-smooth) may need *more* relative movement to fit, and may show worse test loss — i.e., laziness has limits.

### Experiment 3 (optional, time permitting)
Compare ReLU vs tanh activation on the same sweeps. Wired in code, not required.

---

## Setup

### Data (synthetic, 1D)
- Inputs `x ∈ [-1, 1]`.
- `n_train = 100` training points sampled uniformly.
- `n_test = 1000` test points on a regular grid (good for clean plots).
- Targets `y = f(x)` with `f` from the table above. No noise (clean regression).

### Model
- `TwoLayerNet(width=m, activation="relu")` — exactly the architecture above.

### Training
- **Loss:** mean squared error.
- **Optimizer:** Adam, learning rate `1e-3`.
- **Epochs:** ~5000 (tuned after a smoke test).
- **Seeds:** 5 per setting, so we can plot mean ± std.

### Main metric: relative weight movement
Defined above. Computed by flattening *all* parameters (both layers' weights and biases) into one vector and comparing the L2 norms.

### Other metrics
- Final train loss, final test loss.
- Train/test loss curves over epochs.
- Learned prediction curve vs the true target.

---

## What the plots will look like

1. **Width vs relative movement** — the headline figure. One line per target function. Expectation: lines slope downward as width increases (for easy targets at least).
2. **Width vs final train loss** — should be low across all widths (the network can fit 100 points).
3. **Width vs final test loss** — generalization story.
4. **Training curves** — loss vs epoch at a few selected widths.
5. **Learned function vs true function** — visual sanity check at the smallest and largest widths, for each target.

All figures saved as PNG and PDF under [results/figures/](results/figures/).

---

## How the code is organized

```
project/
  README.md
  requirements.txt
  src/
    data.py              # target functions + dataset generation
    model.py             # TwoLayerNet
    metrics.py           # flatten params, relative movement, MSE
    train.py             # train_one_model(config) -> summary + curves
    plots.py             # all plots, runnable from saved CSV/JSON
    run_experiments.py   # smoke + full configs, runs the sweep
  results/
    figures/             # PNG + PDF output
    tables/              # any human-readable summary tables
    raw/                 # CSV + JSON of all runs (source of truth)
  paper_notes/
    outline.md           # write-up outline
    related_work.md      # short literature review
```

The runner saves:
- `summary_<config>.csv` — one row per run (target, width, seed, losses, movement, runtime).
- `curves_<config>.csv` — one row per (run, epoch) for the loss curves.
- `predictions_<config>.json` — saved predictions for selected widths/seeds (for the "learned function" plots).
- `config_<config>.json` — the exact config used, plus a timestamp.

`plots.py` reads these files only — so plots can be regenerated **without retraining**.

---

## How to run (once implemented)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Quick smoke test (a few minutes)
python -m src.run_experiments --config smoke

# 3. Full sweep (longer)
python -m src.run_experiments --config full

# 4. Regenerate all figures from saved results
python -m src.plots
```

The code auto-selects GPU (CUDA or Apple MPS) if available, else CPU.

---

## Reproducibility

- Every run sets a random seed (`numpy`, `torch`, Python `random`).
- The exact config used is saved alongside results.
- 5 seeds per setting → reported numbers are mean ± std.
- Note: results are reproducible *qualitatively* across machines, but bit-identical reproducibility across devices (CPU vs GPU vs MPS) is not guaranteed — this is normal in deep learning.

---

## Honest caveats (worth knowing up front)

- **Adam vs theory.** The cleanest lazy-training theory uses small-step gradient descent with a specific weight scaling (NTK parameterization, `1/√m`). We use Adam and standard PyTorch initialization for stability. The qualitative *relative-movement-shrinks-with-width* trend should still appear, but exact numerical predictions from NTK theory would not.
- **Tiny problem.** 1D regression with 100 points is intentionally toy — it is what makes everything visualizable in a single afternoon and lets us sweep widths up to 1000 without a cluster.
- **No claim of new theory.** This is an experimental study: we reproduce a known phenomenon and probe one of its limits.

---

## What we hope to learn

- Concrete evidence (plots + numbers) that wider networks fit while moving less, on at least the easy target.
- A clearer picture of *when* this stops being true — does laziness survive a high-frequency or non-smooth target, or does the network have to "wake up" and move more?
- A clean, reproducible code artifact that someone else could clone and rerun.

---

## References (to be expanded in `paper_notes/related_work.md`)

- Jacot, Gabriel, Hongler. *Neural Tangent Kernel: Convergence and Generalization in Neural Networks.* NeurIPS 2018.
- Chizat, Oyallon, Bach. *On Lazy Training in Differentiable Programming.* NeurIPS 2019.
- Lee et al. *Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent.* NeurIPS 2019.
- Arora et al. *On Exact Computation with an Infinitely Wide Neural Net.* NeurIPS 2019.
