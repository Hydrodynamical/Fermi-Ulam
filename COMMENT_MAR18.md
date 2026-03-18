# Summary Of Recent Changes

This update adds a full real-data hazard-learning workflow on top of the collision-free Fermi-map calibration path, then extends the repo with a dedicated deterministic-map sweep for estimating the global-stochasticity threshold `u_s`.

## Code changes

- Added real-data wall-bank and hazard-learning support:
  - `calibration/export_wall_bank.py`
  - `toy/build_real_survival_dataset.py`
  - `toy/train_hazard_model.py`
  - `calibration/evaluate_real_survival.py`
- Added repo-local hazard models in `calibration/hazard_models.py` so training/evaluation can run in the current Python 3.15 alpha environment without `scikit-learn`.
- Added real-data diagnostics and visualization:
  - `calibration/plot_real_wall_dashboard.py`
  - `calibration/compare_empirical_targets.py`
  - `calibration/plot_empirical_hazard_ratio.py`
- Added refined target-definition tools in `calibration/core_mask.py`:
  - entry-core target
  - retention-core target
- Extended label handling and dataset utilities in `calibration/survival_dataset.py`.
- Added deterministic-map global-stochasticity threshold tools:
  - `calibration/export_us_sweep.py`
  - `calibration/estimate_us.py`
- Updated `README.md` to document the new workflows, diagnostics, and current findings.
- Added `joblib` explicitly to `requirements.txt`.

## Real-data hazard-learning results

We ran the real-data wall-bank pipeline end to end:

1. export collision-free wall bank
2. build row-wise hazard dataset
3. train baseline `h(u)` and age-structured `h(u,a)` models
4. evaluate learned survival against empirical first-passage curves

Main result: target refinement improved the event definition substantially, but the state `(u, a)` still did **not** beat the scalar baseline on survival objects.

### Target definitions tested

- `proxy`: entry into the coarse chaos/proxy mask
- `entry-core`: entry into a stricter age-conditioned core
- `retention-core`: entry into a large-age, in-proxy, short-horizon retention core

### Target-side conclusion

The retention-core target with

- `H = 1`
- `retention_threshold = 0.6`
- `retention_cv_threshold = 0.5`

was the first refined target that looked physically credible.

Compared with the earlier targets:

- proxy target was too broad and boundary-contaminated
- entry-core target was too sparse / degenerate
- retention-core target landed in the useful middle ground

### Model-side conclusion

Even on the frozen retention-core target, the age-structured model only improved held-out hazard-style metrics slightly, while the scalar baseline still won on the physically important survival metrics (`RMST`, plateau, survival MAE).

Interpretation:

- age dependence is real
- target quality mattered and was improved
- but `(u, a)` is still an incomplete predictive state

This now justifies adding one more local observable in the next iteration, rather than changing the target again.

## Empirical diagnostics added

- Real wall dashboard for raw wall-bank / dataset inspection
- Side-by-side empirical target comparison (`proxy`, `entry-core`, `retention-core`)
- Empirical hazard-ratio diagnostic `h(u,a) / h_bar(u)`

The hazard-ratio result is important: the retention-core target shows strong, coherent age dependence, especially at high energy. So the scalar model is not winning because the process is truly age-independent. It is winning because `(u, a)` does not capture enough local state.

## Global-stochasticity threshold `u_s`

Added a dedicated deterministic-map sweep over stratified initial conditions `(u0, psi0)` to estimate the global-stochasticity threshold directly from the collision-free map, rather than from the bank-induced long-time measure.

### Method

- Run a stratified sweep over log-spaced `u0` and uniform `psi0`
- For each `u0`, compute per-seed tail diagnostics:
  - phase entropy
  - visited-phase coverage
  - lag-1 ACF of `cos(psi)`
- Mark seeds as trapped/bad if entropy or coverage falls below threshold
- Estimate `u_s` from the largest contiguous low-energy prefix satisfying:
  - `entropy_floor(u0) >= 0.95`
  - `coverage_floor(u0) >= 0.95`
  - `p_trap(u0) <= 0.01`

with `entropy_floor` and `coverage_floor` defined as the `q = 0.01` floor across phase seeds. The estimator was updated to use this robust low-quantile floor instead of literal worst-seed minima, which were too sensitive to isolated outliers.

### Current coarse-screen estimate

From the large screening sweep:

```text
u_s ≈ 11.95
```

This should be treated as a coarse screening estimate, not the final threshold. The next step is a refinement sweep in a narrower window around roughly `u0 in [8, 18]` with more seeds and longer trajectories.

## Current state of the project

The repo now supports:

- deterministic-map calibration and diagnostics
- real-data hazard-learning with multiple target definitions
- empirical target comparison and age-dependence diagnostics
- direct deterministic-map threshold estimation for `u_s`

The main scientific conclusions at this point are:

1. the scalar `tau_mix(v)` closure is not adequate
2. the best event definition so far is a short-horizon retention-based interior core
3. age dependence is real, but `(u, a)` is not enough
4. the current coarse estimate of the global-stochasticity threshold is `u_s ≈ 11.95`
5. the next justified step is to refine `u_s` and add exactly one local observable to the hazard model