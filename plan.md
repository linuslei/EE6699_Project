# Project Plan — Future Work

Status as of writing: scaffold + smoke run complete. Pipeline verified end
to end on `SMOKE_CONFIG`. The full scientific sweep has not been run yet,
and the write-up has not been started.

This document tracks what is left to do, in roughly the order it should
happen.

---

## Milestone A — Run the full sweep

**Goal:** Produce the actual data the project's claims will rest on.

1. **Pilot one expensive run** to project total wall time.
   `python -m src.run_experiments --config smoke` already times width=100.
   Add a one-off pilot of (sin_low, width=1000, seed=0, 5000 epochs) and
   record runtime. If a single width=1000 run is, say, ≥ 60 s, the full
   sweep (15 × 1000-runs out of 75 total) will dominate; budget
   accordingly.
2. **Decide on the final width grid.**
   - Default: `[10, 50, 100, 500, 1000]`.
   - Fallback if too slow on the chosen device:
     `[10, 50, 100, 300, 500]` and/or drop `epochs` from 5000 → 3000.
3. **Run the full sweep** and confirm the post-run sanity assertions
   (`len(summary) == 75`, all four output files exist):
   ```bash
   python -m src.run_experiments --config full
   ```
4. **Generate all figures from the saved files**:
   ```bash
   python -m src.plots --config full
   ```
5. **Sanity-check the artifacts.** Open
   `results/raw/summary_full.csv` and confirm:
   - no NaN values in `final_train_loss`, `final_test_loss`, `relative_movement`;
   - for each `(target, width)` group, all 5 seeds are present;
   - relative movement is finite and on the same order of magnitude as the smoke run.

**Done when:** `results/raw/summary_full.csv`, `curves_full.csv`,
`predictions_full.json`, `config_full.json` exist and all five figure
families are in `results/figures/`.

---

## Milestone B — Read the results

**Goal:** Decide what story the data tells before writing about it.

1. **Look at `fig_width_vs_movement_full.png` first.** Three lines, one per target.
   - Does relative movement decrease as width grows?
   - Does the slope differ between `sin_low`, `sin_high`, `abs`?
2. **Look at `fig_width_vs_test_loss_full.png`.**
   - Does test loss go down with width on each target?
   - Where is the plateau? Is it different across targets?
3. **Look at the per-target figures (`fig_curves_*` and `fig_learned_*`).**
   - Did all widths actually converge? If a width's loss curve is still
     dropping at the last logged epoch, mark it as "epoch-budget bound"
     and consider rerunning that subset with more epochs.
   - Do the learned-function plots show the expected behavior
     (good fit at large widths, smoothing for `abs`, possible
     spectral-bias underfit on `sin_high`)?
4. **Write a 1-page interpretation** (informally, before formal write-up):
   what the data confirms, what it doesn't, and what surprised us.
   Append to [experiment.md](experiment.md) or create
   `paper_notes/findings.md`.

**Done when:** the interpretation matches the figures and we can state
each conclusion in one sentence.

---

## Milestone C — Write up paper notes

**Goal:** Convert the data into the deliverable the course expects.

1. `paper_notes/related_work.md` — short literature notes on:
   - Jacot, Gabriel, Hongler 2018 (NTK).
   - Chizat, Oyallon, Bach 2019 (Lazy training).
   - Lee et al. 2019 (wide nets ≈ linearized).
   - (Optional) Arora et al. 2019; Rahaman et al. 2019 ("spectral bias")
     — relevant to why `sin_high` should be harder than `sin_low`.
2. `paper_notes/outline.md` — 1–2 page outline of the final write-up:
   abstract → background → methods → reproduction (Exp 1) →
   extension (Exp 2) → discussion → limitations → references.
3. **Draft the write-up** using the outline. Embed final figures from
   `results/figures/`. Cite specific rows from
   `results/raw/summary_full.csv` for any quantitative claim.

**Done when:** a coherent draft exists and references the canonical
artifacts in `results/`.

---

## Milestone D — Optional / stretch experiments

Pick at most one or two; do not let these block the write-up.

1. **Activation comparison (Experiment 3).** Already wired in `model.py`.
   Add a `FULL_TANH_CONFIG` to `run_experiments.py` and rerun on the
   widths that matter (probably just `[100, 1000]` × all targets × 5
   seeds to keep cost down). Plot tanh vs ReLU side-by-side on the
   movement and test-loss figures.
2. **SGD vs Adam.** Add `OPTIM_CONFIG` with `optimizer="sgd"` at a
   small LR (e.g. 1e-2 or 1e-3 depending on stability). The lazy-training
   theory is cleanest under SGD; an SGD vs Adam comparison directly
   addresses the "honest caveat" already in the README.
3. **Hidden-feature change metric.** Add a function in `metrics.py` that
   measures `||φ_final(x) - φ_initial(x)|| / ||φ_initial(x)||` for
   `φ(x) = act(W₁ x + b₁)`. Lazy training predicts this stays small for
   wide networks. Plot vs width.
4. **NTK parameterization.** Reinitialize the model with the
   `1/√m`-scaled output layer used in NTK theory and rerun
   `sin_low` only at all widths. Compare the relative-movement slope
   to the default-init version. This is the most theoretically faithful
   variant.
5. **Noisy targets.** Add Gaussian noise to `y_train` and observe how
   relative movement and the train/test gap respond as width grows
   — connects to generalization in the overparameterized regime.

---

## Milestone E — Polish

1. **Reproducibility check.** On a clean clone, run:
   ```bash
   pip install -r requirements.txt
   python -m src.run_experiments --config smoke
   python -m src.plots --config smoke
   ```
   confirm the same numbers and figures appear (qualitatively; bit-exact
   not guaranteed across devices).
2. **Add a top-level `Makefile` or `tasks.md`** with the canonical
   commands (`smoke`, `full`, `plots`, `clean`). Optional but nice.
3. **Tag a release** in git once the full sweep is locked in
   (e.g. `v1.0-sweep`) so figures in the write-up can be linked to a
   commit SHA.

---

## Checklist (compact)

- [ ] Pilot a width=1000 run; record timing.
- [ ] Run `--config full`.
- [ ] Run `python -m src.plots --config full`.
- [ ] Sanity-check `summary_full.csv` (no NaN, 75 rows, 5 seeds per cell).
- [ ] Note interpretation in `experiment.md` or `paper_notes/findings.md`.
- [ ] Write `paper_notes/related_work.md`.
- [ ] Write `paper_notes/outline.md`.
- [ ] Draft final write-up.
- [ ] (Optional) Run one stretch experiment from Milestone D.
- [ ] Reproducibility check on a clean clone.

---

## Risks worth re-checking before the full run

- **Width=1000 × 5000 epochs × 75 runs may be slow on CPU.**
  Mitigations are in place (auto MPS/CUDA selection, fallback widths).
- **MPS op gaps.** If MPS errors, add `--device cpu` support to the runner
  (currently `device="auto"` in the configs; we'd extend
  `run_experiments.py` to accept `--device`).
- **Adam ≠ NTK.** Documented in README and `experiment.md`. If the
  expected lazy trend does *not* appear, this is the first thing to
  flag in the discussion section, and Milestone D.4 (NTK
  parameterization) becomes the natural follow-up.
- **Disk usage.** `predictions_full.json` only stores selected
  (width, seed) pairs; the full curves CSV with `log_every=50`,
  5000 epochs, 75 runs is ~7500 rows — trivial. No disk concern expected.
