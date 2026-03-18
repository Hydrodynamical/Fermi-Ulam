# Fermi-Ulam PDMP — proof of concept

A proof-of-concept particle simulation for a **Piecewise Deterministic Markov
Process (PDMP)** model of wall heating in a 1D Fermi-Ulam gas.

---

## Physical setup

N particles move ballistically in a slab [0, L], punctuated by stochastic jumps
at wall-hit and collision events.

- **Left wall (x = 0):** fixed, elastic reflection.
- **Right wall (x = L):** driven. Each particle carries a binary mixing state σ.
  - **σ = 0 (unmixed):** deterministic Fermi-Ulam reflection, `v⁺ = 2 A ω cos(ω t) − v`.
  - **σ = 1 (mixed):** stochastic wall-heating kernel (see below).
- **Bulk collisions:** Bird/DSMC-style mean-field Kac collisions with local rate
  `ν_i = λ₀ n_loc(xᵢ)(|vᵢ − v̄_loc| + v_min)`, resetting σ = 0 and age = 0
  for both particles.

---

## The mixing-state model

A particle transitions from σ = 0 to σ = 1 after a **mixing time**

```text
τ_mix(v) = m₀ + m₁ / (|v| + v_floor)
```

Once mixed, right-wall hits apply a **reflected Ornstein-Uhlenbeck step**:

```text
s⁺ = s⁻ + κ_w (V* − s⁻) Δτ_w  +  √(2 D_w Δτ_w) ζ,   ζ ~ N(0,1)
s⁺ = |s⁺|,   v⁺ = −s⁺
```

This discretises the Fokker-Planck generator

```text
L_wall f = −∂_s[ κ_w (V* − s) f ]  +  D_w ∂²_s f
```

---

## Core scientific question

> Wall heating is throttled by the probability that a particle survives long
> enough between bulk collisions to enter the mixed/chaotic wall regime.

The survival factor is

```text
χ = exp(−ν τ_mix)
```

and the equilibrium energy is suppressed roughly as E_inf(λ₀) ~ χ · E_inf(0).
Experiment B confirms this suppression; the threshold-clock formula
`exp(−ν_i τ_mix(vᵢ))` tracks the simulation well, though a remaining gap points
to the need for a full g⁰/g¹ two-population closure.

---

## Project structure

```text
config/
  base.yaml                  Default PDMP simulator parameters
  calibration_base.yaml      Calibration experiment defaults
  experiment_*.yaml          Per-experiment overrides
src/
  particles.py        State arrays: x, v, sigma, age
  walls.py            Left elastic wall; right Fermi wall (det.) and OU wall (mixed)
  collisions.py       DSMC collision step + chi measurement from collision events
  mixing.py           Deterministic threshold mixing
  simulator.py        Fixed-dt main loop
  observables.py      E(t), p_mix(t), chi_direct(t)
  diagnostics.py      Domain and energy sanity checks
  plotting.py         Standard plot helpers
  experiment_a_fp.py  Experiment A: wall-only heating, validate FP/OU operator
  experiment_b_fp.py  Experiment B: collision suppression sweep, test chi law
calibration/
  map.py                   Exact discrete Fermi-Ulam map (u,ψ) → (u′,ψ′,T) per bounce
  trajectories.py          Ensemble generation; inherits A, ω, L from base.yaml
  export_wall_bank.py      Export long real trajectory banks with proxy labels M_n
  core_mask.py             Refined core-mask diagnostics: entry-core and retention-core targets
  phase_mixing.py          ACF-based phase mixing time τ_lag(u)
  kramers_moyal.py         Kramers-Moyal drift b(u) and diffusion a(u) estimation
  fokker_planck.py         FP steady-state solver and empirical comparison
  plotting.py              Poincaré section, ACF, τ_mix, b/a, and FP comparison plots
  mixing_diagnostics.py    τ_int(u), TV distance, phase entropy H(u)
  increment_diagnostics.py Δu moments, small-jump ratio, Var vs lag
  markov_tests.py          Phase-conditioned drift map, semigroup test, lag-1 autocorr
  chaos_mask.py            Entropy-based KAM mask; masked KM/FP re-estimation
  fp_diagnostics.py        KS distance, log-residual r(u), forward FP validation
  diagnostic_report.py     8-panel dashboard figure
  ll_validation.py         Lichtenberg-Lieberman transport identity + stationary-density tests
  first_passage.py         Residual waiting-time law; survival S(u,a), hazard h(u,a), τ_rmst/τ_exp(u)
  survival_dataset.py      Convert labeled real trajectories into row-wise hazard data
  hazard_models.py         Repo-local hazard models with predict_proba API (no sklearn dependency)
  evaluate_real_survival.py   Compare learned real-data survival against empirical first-passage curves
  compare_empirical_targets.py Compare empirical first-passage objects across target definitions
  plot_real_wall_dashboard.py  Visual dashboard for wall-bank and row-wise hazard data
  plot_empirical_hazard_ratio.py Empirical h(u,a) / h_bar(u) age-dependence diagnostic
  export_us_sweep.py         Stratified (u0, psi0) sweep for estimating the global-stochasticity threshold u_s
  estimate_us.py             Worst-seed threshold diagnostics and Poincare summaries for u_s
  run_calibration.py       Top-level runner (calibration + diagnostic pass)
results/              Output npz files and figures (git-ignored)
```

---

## Real survival-learning data pipeline

The toy hazard-learning scripts can now be driven by real collision-free Fermi-map data.

1. Export a labeled wall trajectory bank:

```text
python calibration/export_wall_bank.py --config config/calibration_medium.yaml --out-dir results/real_wall_bank
```

This writes `u_n`, `psi_n`, per-bounce proxy labels `M_n`, energy-bin indices, the chaos mask, and the local diagnostics used to define that proxy.

2. Build the row-wise hazard dataset:

```text
python toy/build_real_survival_dataset.py --data results/real_wall_bank/wall_bank.npz --out-dir results/real_wall_hazard
```

The output format matches the toy pipeline: `X_full = [u, log(1+a)]`, `X_base = [u]`, `y = 1[M_{n+1}=1]`, plus `seg_id` so train/test splitting can be done by non-proxy excursion rather than by individual rows.

3. Train the baseline and age-structured hazard models:

```text
python toy/train_hazard_model.py --data results/real_wall_hazard/dataset.npz --out-dir results/real_wall_hazard/models
```

4. Evaluate learned survival against empirical first-passage curves:

```text
python calibration/evaluate_real_survival.py \
  --bank results/real_wall_bank/wall_bank.npz \
  --models results/real_wall_hazard/models \
  --out-dir results/real_wall_hazard/eval
```

The current workspace uses Python 3.15 alpha, where prebuilt `scikit-learn`
wheels are unavailable. The real-data hazard-learning path therefore uses the
repo-local models in `calibration/hazard_models.py`, while preserving the same
`predict_proba` interface and the same training/evaluation harness.

---

## Refined Target Workflow

The main new development is target refinement on the wall-bank side. Three
first-passage targets are now available:

- `proxy` — first entry into the coarse entropy-based chaotic mask.
- `entry-core` — first entry into a stricter age-conditioned core estimated from out-of-proxy entry statistics.
- `retention-core` — first entry into a refined interior core estimated from large-age, in-proxy short-horizon retention.

The retention-core target is built with `calibration/core_mask.py`, for example:

```text
python calibration/core_mask.py \
  --bank results/real_wall_bank_medium/wall_bank.npz \
  --mode retention \
  --a-star 10 \
  --horizon 1 \
  --n-psi-bins 32 \
  --min-count 500 \
  --entropy-threshold 0.9 \
  --hazard-cv-threshold 0.5 \
  --retention-threshold 0.6 \
  --tag H1_r0p6_cv0p5
```

That command writes a distinct mask artifact such as:

```text
results/real_wall_bank_medium/diagnostics/core_mask_retention_H1_r0p6_cv0p5.npz
results/real_wall_bank_medium/diagnostics/core_mask_retention_H1_r0p6_cv0p5.png
```

The empirical comparison across target definitions is produced with:

```text
python calibration/compare_empirical_targets.py \
  --bank results/real_wall_bank_medium/wall_bank.npz \
  --entry-mask results/real_wall_bank_medium/diagnostics/core_mask.npz \
  --retention-mask results/real_wall_bank_medium/diagnostics/core_mask_retention_H1_r0p6_cv0p5.npz \
  --out-dir results/real_wall_bank_medium/diagnostics/target_compare_H1_r0p6_cv0p5
```

---

## Global-Stochasticity Threshold Sweep

The repository now also contains a dedicated deterministic-map path for estimating
the global-stochasticity threshold `u_s`. This is separate from the wall-bank /
first-passage pipeline.

The primary estimator is a stratified sweep over initial conditions `(u0, psi0)`:

```text
python calibration/export_us_sweep.py \
  --out-dir results/us_sweep_screen \
  --n-u 80 \
  --n-psi 192 \
  --n-hits 20000 \
  --burn-in 5000 \
  --thin-stride 50 \
  --n-phase-bins 64
```

This does not sample the long-time dynamical measure. Instead it probes the
deterministic collision-free Fermi map uniformly over phase at each initial
energy and saves per-seed summary diagnostics:

- tail phase histograms,
- visited-phase coverage,
- normalized phase entropy,
- mean / std of tail energy,
- lag-1 ACF of `cos(psi)`,
- a thinned tail Poincare sample for representative plots.

The corresponding estimator is:

```text
python calibration/estimate_us.py \
  --data results/us_sweep_screen/us_sweep.npz \
  --out-dir results/us_sweep_screen/estimate \
  --entropy-threshold 0.95 \
  --coverage-threshold 0.95 \
  --trap-fraction-threshold 0.01 \
  --floor-quantile 0.01
```

It computes per-`u0` diagnostics over phase seeds:

- `entropy_min(u0)`
- `coverage_min(u0)`
- `p_trap(u0)`
- `lag1_acf_max(u0)`

For thresholding, the current estimator does **not** use the literal worst seed
directly, because isolated outlier seeds can collapse the prefix too early. It
uses a robust low-quantile floor instead:

```text
entropy_floor(u0)  = q_0.01[H_psi(u0, psi0)]
coverage_floor(u0) = q_0.01[coverage(u0, psi0)]
```

and defines `u_s` as the top of the largest contiguous low-energy prefix where:

```text
entropy_floor(u0)  >= 0.95
coverage_floor(u0) >= 0.95
p_trap(u0)         <= 0.01
```

### Large-screen diagnostic outputs

The first large screening run now writes:

```text
results/us_sweep_large_screen/us_sweep.npz
results/us_sweep_large_screen/estimate/us_diagnostics.npz
results/us_sweep_large_screen/estimate/us_diagnostics.png
results/us_sweep_large_screen/estimate/us_poincare_sections.png
```

The two main figures have distinct roles:

- `us_diagnostics.png` is the threshold figure. It shows worst-seed phase entropy,
  trapped-seed fraction, worst-seed visited-phase coverage, and the final prefix-based
  threshold rule used to estimate `u_s`.
- `us_poincare_sections.png` is the geometric cross-check. It shows representative
  thinned Poincare sections below, near, and above the estimated threshold so the
  transition can be inspected visually.

This `u_s` sweep is intended to be the primary deterministic-map estimator of the
global-stochasticity threshold. The wall-bank core/proxy and first-passage tools are
still useful, but mainly as secondary cross-checks near the candidate threshold.

### Current coarse-screen estimate

From the large screening sweep in `results/us_sweep_large_screen/`, the current
numerical estimate is:

```text
u_s ≈ 11.95
```

This value comes from the robust prefix rule above, applied to the large-screen
run and saved in `results/us_sweep_large_screen/estimate/us_diagnostics.npz`.
It should be treated as a **coarse screening estimate**, not the final value.
The next step is to refine the energy window around roughly `u0 in [8, 18]` with
more seeds and longer trajectories.

---

## Real-Data Findings

The real-data hazard-learning campaign has now progressed far enough to separate
three distinct questions:

1. Is the event definition physically meaningful?
2. Is age dependence present empirically?
3. Is `(u, a)` alone a sufficient predictive state?

### 1. Coarse proxy target was too broad

The original target, first entry into the coarse entropy-based chaos mask,
produced a usable dataset and allowed training to run end to end. However, the
empirical target comparison showed clear pathology:

- the hazard field had strong horizontal banding,
- the survival curves collapsed too quickly at many energies,
- the proxy mask was probably contaminated by sticky/boundary structure.

On the medium bank (`200 × 50k` hits), the proxy target used `26` mask bins.
It gave the largest target support but the least clean first-passage geometry.

### 2. Entry-core target was too degenerate

The first refined target, based on large-age out-of-proxy entry statistics,
was too sparse. With the initial thresholds it was empty; with relaxed settings
it produced only `3` bins. This made classification easier, but not the survival
objects. The target was too close to a specific boundary-crossing event to define
a stable Markov-valid interior.

### 3. Retention-core target is the first encouraging target-side result

The best target so far is the retention-based core, defined by large-age,
in-proxy, short-horizon retention with:

```text
H = 1
retention_threshold = 0.6
retention_cv_threshold = 0.5
```

This target sits in the right middle ground:

| Target | Mask bins | Supported start bins | Interpretation |
| --- | ---: | ---: | --- |
| proxy | 26 | 14 | too broad, boundary-contaminated |
| entry-core | 3 | 37 | too sparse / degenerate |
| retention-core (`H=1`) | 10 | 30 | first plausible interior-sea target |

Empirically, the retention-core target has:

- a much smoother hazard surface,
- clearer separation of survival curves by energy,
- a monotone, physically plausible RMST curve,
- a cleaner plateau structure at high energy.

This is the first target that is both statistically usable and physically
credible as a short-horizon locally mixing interior.

### 4. Age dependence is real, but `(u, a)` is still incomplete

The hazard-ratio diagnostic `h(u,a) / h_bar(u)` for the frozen retention-core
target shows strong empirical age dependence:

- median ratio across supported bins is about `2.15` at age `0`,
- stays above `1` for the first few bounces,
- falls below `1` by roughly age `10`,
- decays to about `0.11` by age `199`.

So the scalar baseline is not winning because the process is truly age-independent.
Age dependence is present and coherent. The strongest age dependence appears in
the high-energy bins with the largest `tau_rmst` and the largest survival plateau,
which is exactly where sticky structure remains most important.

### 5. Hazard training improved, but survival objects still did not flip

Across all three target definitions, the age-structured model typically improved
held-out hazard metrics slightly, but it still did not beat the scalar baseline
on the physically important survival quantities.

Medium-bank survival evaluation summary (`max_age = 200`):

| Target | RMST MAE full | RMST MAE base | Hazard MAE full | Hazard MAE base | Survival MAE full | Survival MAE base |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| proxy | 11.70 | 9.14 | 0.598 | 0.379 | 0.0596 | 0.0477 |
| entry-core | 14.40 | 13.09 | 0.159 | 0.082 | 0.0764 | 0.0670 |
| retention-core (`H=1`) | 27.28 | 26.51 | 0.170 | 0.158 | 0.1368 | 0.1330 |

This is a consistent negative result: target refinement alone was not enough.
The best interpretation is that the predictive state `(u, a)` is still missing
one local observable, even after the event definition is improved.

### 6. Current conclusion

The project has established three robust conclusions:

- the original scalar `tau_mix(v)` closure is not adequate,
- the right event definition is closer to short-horizon locally mixing chaos than to a broad global chaos mask,
- the pair `(u, a)` alone is still too compressed to predict the refined first-passage law accurately.

The next justified step is to keep the frozen retention-core target and add one
physically local feature, such as a short-window diffusivity or stickiness proxy,
while leaving the training/evaluation harness unchanged.

---

## Calibration experiment (non-interacting)

Before reintroducing bulk collisions, the calibration experiment derives the
**intrinsic wall-heating coefficients** b(u) and a(u) empirically from the
deterministic Fermi-Ulam map alone — the drift and diffusion of the effective
Markov process in energy u = ½s² after phase mixing, which directly parameterise

```text
L_wall φ(u) = b(u) φ′(u) + ½ a(u) φ″(u)
```

**What it runs.**  The exact discrete map at right-wall hit events — no
fixed-dt loop, no σ/age machinery, no collisions:

```text
U_w      = A ω cos(ψ_n)
s_{n+1}  = |2 U_w − s_n|
T_n      = 2L / s_{n+1}          (round-trip time)
ψ_{n+1}  = (ψ_n + ω T_n) mod 2π
```

For 500 particles × 10⁵ hits the full run takes ~10–30 min.

**What it measures.**

1. **Phase mixing time** τ_lag(u) — the smallest lag k where the phase ACF
   `C(k; u) = ⟨cos(ψ_{n+k} − ψ_n) | u_n ≈ u⟩` drops below a threshold.

2. **Kramers-Moyal coefficients** b̂(u), â(u) at lag m(u) = ceil(3 τ_lag(u)):

   ```text
   b̂(u) = E[Δu] / m,    â(u) = Var[Δu] / m
   ```

3. **FP validation** — the analytic steady-state

   ```text
   f_FP(u) ∝ exp(2 ∫ b/a du) / a(u)
   ```

   is compared to the empirical histogram; L1 and KL divergence are reported.

**Diagnostic pass** (runs automatically unless `--no-diagnostics`):

| Module | Question answered |
| --- | --- |
| `mixing_diagnostics` | Is the phase actually uniform? τ_int vs τ_lag |
| `increment_diagnostics` | Is Δu Gaussian? Is diffusion the right closure? |
| `markov_tests` | Does drift depend on phase ψ? (Markov failure test) |
| `chaos_mask` | Are bad-fitting bins in KAM islands? |
| `fp_diagnostics` | KS distance, log-residual r(u), forward FP convergence |
| `ll_validation` | Does D ∝ u? Is B = ½D′? Is P(u) flat on the chaotic sea? |
| `first_passage` | Residual waiting-time into binwise Markov proxy; τ_rmst, τ_exp, non-exponentiality |

**Outputs** (in `results/calibration/`): `calibration_results.npz` plus five
plots — `poincare_section.png`, `phase_acf.png`, `tau_mix.png`,
`km_coefficients.png`, `fp_comparison.png` — and a `diagnostics/` subdirectory
containing eight `.npz` files, `diagnostic_report.png` (8-panel dashboard),
`ll_validation.png` (Lichtenberg-Lieberman 4-panel), and `first_passage.png`
(residual waiting-time 4-panel).

---

## Calibration findings

Results below are from the **medium run**: 200 particles × 50 k hits,
parameters A = 0.05, ω = 20, L = 1 (`config/calibration_medium.yaml`).

### Phase mixing and KM coefficients

τ_lag resolved for all 40 energy bins, range [1, 10] bounces; coarse lag
m(u) ∈ [5, 30]. b̂(u), â(u) estimated for all 40 bins. FP steady-state vs
empirical histogram: L1 = 0.71, KL = 0.28 — a large residual indicating the
random-phase FP closure is a poor fit at these parameters.

26 of 40 bins pass the entropy-based chaos mask (H_norm ≥ 0.8).

### Lichtenberg-Lieberman transport validation

`ll_validation.py` tests one-bounce per-bounce transport against the LL
random-phase Hamiltonian predictions D(u) ∝ u, B(u) = const, B = ½D′:

| Quantity | LL prediction | Measured |
| -------- | ------------- | -------- |
| D(u) functional form | ∝ u (slope 1 on log-log) | ≈ linear, c_D ≈ 4.52 |
| B(u) functional form | constant | c_B ≈ 0.99 |
| Landau ratio c_B / (½ c_D) | 1.0 | ≈ 0.44 |
| P_emp(u) vs flat in u (L1) | 0 | ≈ 0.67 |

The Landau identity B = ½D′ fails by a factor of ~2.3. Compared to the
low-ω run (ll_ratio ≈ 0.21), faster driving (higher ω) brings the system
closer to the LL limit but does not recover it. D(u) is approximately linear
(correct functional form), so the diffusion channel is operating; the deficit
is in the drift B, pointing to residual KAM trapping within nominally chaotic bins.

### Residual waiting-time law into the binwise Markov proxy

`first_passage.py` measures, from every non-chaotic starting bounce, the
number of additional bounces until first entry into a chaotic energy bin
(the *residual waiting time*, RWT). This is the empirical replacement for
the ad-hoc scalar τ_mix(v) in the PDMP mixing clock.

Key results (τ_rmst = restricted mean survival time up to 500 bounces):

| Energy range | Bins | τ_rmst (bounces) | Censored | Interpretation |
| ------------ | ---- | ---------------- | -------- | -------------- |
| u ≲ 10 | 3, 4, 19, 25–32 | 1.4 – 2.9 | < 0.1% | Near-instant entry; 1–2 bounces to chaotic set |
| u ≈ 24 | 36 | 17 | 0.8% | Onset of slow mixing |
| u ≈ 29–45 | 37–39 | 88 – 187 | 13–31% | Slow entry; heavy KAM trapping |

The exponential-fit R² is negative for all non-chaotic bins, confirming that
the survival function is not geometric: at low energy it is a near-step function
(entry in 1–2 bounces, not exponential decay), and at high energy it has a heavy
tail beyond max_age = 500. The coefficient of variation h_cv > 1 at high energy
indicates that the discrete hazard h(u, a) varies strongly with age — the
single-exponential-clock reduction is not valid in either regime.

**Consequence for the PDMP closure.** The scalar τ_mix(v) can serve as a rough
order-of-magnitude guide only at intermediate energies. At low energy, entry into
the chaotic wall regime is essentially instantaneous (τ_rmst ~ 1–3 bounces); at
high energy, it is slow, non-exponential, and strongly energy-dependent. A faithful
closure requires retaining the full age-dependent hazard h(u, a) or at minimum
the energy-stratified RMST τ_rmst(u).

### Open questions

- **Why does ll_ratio fall short of 1?** The Landau identity requires B = ½D′ for
  a Hamiltonian system with uniform phase. The deficit (ratio ≈ 0.44) is likely a
  combination of (a) sticky KAM orbits that inflate D while suppressing B, and (b)
  incomplete phase mixing within the coarse-lag window. A phase-conditioned drift map
  B(u, ψ) would distinguish the two.
- **Can the FP mismatch (L1 = 0.71) be reduced?** The mismatch is large even after
  masking non-chaotic bins, suggesting the chaotic sea is not ergodic on the
  timescale of the simulation. Longer trajectories or a stricter chaos mask are needed.
- **What is the correct effective τ_mix(u) for the PDMP?** The first-passage
  diagnostic shows τ_rmst(u) varies by two orders of magnitude across the energy
  range. Replacing the scalar τ_mix with the energy-stratified τ_rmst(u) is the
  natural next step in the PDMP closure.

---

## Running

```bash
pip install -r requirements.txt

# Single run (base config)
python src/main.py --out-dir results/run1

# Experiment A: wall heating only (4 OU wall strengths + deterministic Fermi)
python src/experiment_a_fp.py --out-dir results/experiment_a_fp

# Experiment B: collision suppression sweep (λ₀ = 0, 0.1, 0.5, 1.0)
python src/experiment_b_fp.py --out-dir results/experiment_b_fp

# Calibration — fast sanity check (~30 s)
python3 calibration/run_calibration.py \
  --config config/calibration_fast.yaml \
  --out-dir results/calibration_fast

# Calibration — medium run (~2–5 min, 200 particles × 50 k hits)
python3 calibration/run_calibration.py \
  --config config/calibration_medium.yaml \
  --out-dir results/calibration_medium

# Calibration — full run (~10–30 min, 500 particles × 100 k hits)
python3 calibration/run_calibration.py --out-dir results/calibration

# Calibration — skip diagnostic pass (faster for quick checks)
python3 calibration/run_calibration.py --no-diagnostics --out-dir results/calibration
```

Create `config/calibration_fast.yaml` for the fast variant:

```yaml
N_particles: 100
N_hits: 20000
n_u_bins: 20
acf_max_lag: 50
```

`config/calibration_medium.yaml` (the run used to generate the findings above):

```yaml
N_particles: 200
N_hits: 50000
n_u_bins: 40
acf_max_lag: 200
```

---

## Key parameters

| Parameter | Meaning |
| --------- | ------- |
| `kappa_w` | OU drift rate toward V* in the mixed wall kernel [1/time] |
| `wall_dtau` | Effective wall-interaction time per hit; η = κ_w · Δτ_w is fractional relaxation per bounce |
| `V_star` | Wall temperature scale (OU fixed point in speed) |
| `D_w` | Diffusion coefficient in speed space [speed²/time] |
| `m0`, `m1` | Mixing-time parameters: τ_mix(v) = m₀ + m₁ / (\|v\| + v_floor) |
| `lambda0` | Collision rate prefactor |

---

## What this does not do

- **The FP coefficients are extracted empirically by the calibration experiment**,
  not derived analytically.  A full analytic treatment would require computing the
  Lyapunov exponent and diffusion tensor of the Fermi-Ulam map, identifying the
  KAM boundary v_KAM(A, ω), and relating κ_w and D_w to the map's invariant
  measure in the chaotic sea.
- **No spatial transport.**  The collision rule is mean-field; there is no genuine
  spatial inhomogeneity beyond the binned local density estimate.
- **1D only.**  The physical Fermi-Ulam problem and the kinetic closure are both
  formulated in one spatial dimension.
