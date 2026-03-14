# Fermi-Ulam PDMP — proof of concept

A proof-of-concept particle simulation for a **Piecewise Deterministic Markov
Process (PDMP)** model of wall heating in a 1D Fermi-Ulam gas.

---

## Physical setup

N particles move ballistically in a slab [0, L].

- **Left wall (x = 0):** fixed, elastic reflection.
- **Right wall (x = L):** driven. Each particle carries a binary mixing state σ.
  - **σ = 0 (unmixed):** the actual deterministic Fermi-Ulam reflection,
    `v⁺ = 2 A ω cos(ω t) − v`.
  - **σ = 1 (mixed):** a stochastic wall-heating kernel that stands in for the
    chaotic phase of the Fermi-Ulam map (see below).
- **Bulk collisions:** Bird/DSMC-style mean-field Kac collisions with a local
  collision rate `ν_i = λ₀ n_loc(xᵢ)(|vᵢ − v̄_loc| + v_min)`.  Each collision
  resets σ = 0 and age = 0 for both particles, erasing wall-phase memory.

The PDMP structure is: deterministic free flight punctuated by stochastic jumps
at wall-hit and collision events.

---

## The mixing-state model

A particle transitions from σ = 0 to σ = 1 after a **mixing time**

```text
τ_mix(v) = m₀ + m₁ / (|v| + v_floor)
```

This represents the time needed for the deterministic Fermi-Ulam orbit to enter
the chaotic sea and lose memory of its initial phase.

Once mixed, right-wall hits apply a **reflected Ornstein-Uhlenbeck step** in
outgoing speed:

```text
s⁺ = s⁻ + κ_w (V* − s⁻) Δτ_w  +  √(2 D_w Δτ_w) ζ,   ζ ~ N(0,1)
s⁺ = |s⁺|,   v⁺ = −s⁺
```

This is the Euler-Maruyama discretisation of the Fokker-Planck generator

```text
L_wall f = −∂_s[ κ_w (V* − s) f ]  +  D_w ∂²_s f
```

and represents the effective boundary operator that the chaotic wall dynamics
induce on the speed distribution after many encounters.

**What this is not:** the OU kernel and τ_mix are phenomenological inputs, not
derived from the Fermi-Ulam map.  In the full problem, κ_w, V*, D_w, and
τ_mix(v, A, ω) must be computed from the map's Lyapunov exponents, diffusion
coefficients in the chaotic sea, and the location of KAM invariant curves.
This simulation takes those as free parameters and tests the structural
predictions of the PDMP framework against them.

---

## Core scientific question

The simulation tests one prediction:

> Wall heating is throttled by the probability that a particle survives long
> enough between bulk collisions to enter the mixed/chaotic wall regime.

The survival factor is

```text
χ = exp(−ν τ_mix)          [deterministic threshold mixing]
```

and the equilibrium energy is suppressed roughly as E_inf(λ₀) ~ χ · E_inf(0).

Experiment B confirms this suppression and identifies that the simple scalar
mean-field approximation χ ≈ μ/(μ+ν) overestimates χ; the threshold-clock,
flux-weighted formula `exp(−ν_i τ_mix(vᵢ))` averaged over the wall-hitting
distribution tracks the simulation more closely, though a remaining gap points
to the post-collision reset bias and the need for a full g⁰/g¹ two-population
closure.

---

## Project structure

```text
config/               YAML parameter files for each experiment
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
results/              Output npz files and figures (git-ignored)
```

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

- **τ_mix and the FP coefficients are not derived from the map.**  The full
  problem requires computing the Lyapunov exponent and diffusion tensor of the
  Fermi-Ulam standard map at each energy, identifying the KAM boundary
  v_KAM(A, ω), and extracting κ_w and D_w from the variance of velocity
  increments in the chaotic sea.  This simulation takes all of those as inputs.
- **No spatial transport.**  The collision rule is mean-field; there is no
  genuine spatial inhomogeneity beyond the binned local density estimate.
- **1D only.**  The physical Fermi-Ulam problem and the kinetic closure are
  both formulated here in one spatial dimension.
