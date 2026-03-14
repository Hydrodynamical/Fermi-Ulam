# Experiment A: Wall Heating Only — Theory and Results

**Model:** N = 2000 particles, slab [0, L = 1],
fixed elastic left wall, stochastic or vibrating right wall,
no bulk collisions, all particles permanently mixed (σ = 1).

---

## 1. The physical setup

Between wall encounters each particle moves ballistically,

```
ẋ = v,   v̇ = 0.
```

**Left wall (x = 0):** elastic reflection,

```
v⁺ = −v⁻.
```

**Right wall (x = L):** two models are compared.

### Deterministic Fermi wall

The wall oscillates with velocity U_w(t) = Aω cos(ωt).
The particle–wall collision rule is

```
v⁺ = 2 U_w(t) − v⁻.
```

With A = 0.05, ω = 20 the wall-velocity amplitude is Aω = 1.0.
This is the classical Fermi–Ulam model.

### Stochastic heating kernel

After a right-wall hit, the outgoing speed is drawn from

```
s_out = (1 − η)|v⁻| + η V* + √(2 D_w) ζ,   ζ ~ N(0,1),
s_out ← max(s_out, 0),
v⁺   = −s_out.
```

The three stochastic runs use the parameters below.

| Label | η | V* | D_w |
|-------|---|----|-----|
| weak | 0.02 | 1.0 | 0.01 |
| moderate | 0.10 | 1.0 | 0.05 |
| strong | 0.20 | 1.0 | 0.10 |

---

## 2. Theoretical analysis of the stochastic kernel

### 2.1 Fixed point of the kernel

The map s ↦ E[s_out] = (1−η)s + ηV* has a unique fixed point at s = V*.
So in the absence of the noise floor (and truncation), all particles thermalize
to speed V*, giving equilibrium energy

```
E_eq = V*²/2 + D_w / η.
```

For all three stochastic runs D_w/η = 0.5, so E_eq = 0.5 + 0.25 = 0.75
is predicted if the kernel thermalizes correctly.

### 2.2 Rate-weighted heating threshold

Faster particles reach the right wall more often: the wall-hit rate for a
particle of speed s is

```
r(s) ≈ s / (2L).
```

The rate-weighted mean energy change per unit time is

```
dE/dt ∝  ∫₀^∞ r(s) · ΔE(s) · f(s) ds
       ≈  −η E[|v|³]  +  η V* E[v²]  +  D_w E[|v|].
```

For the initial half-normal distribution (v ~ N(0,1)):

| Moment | Value |
|--------|-------|
| E[|v|]  | √(2/π) ≈ 0.798 |
| E[v²]   | 1.000 |
| E[|v|³] | 2√(2/π) ≈ 1.596 |

Setting dE/dt = 0 gives the **net-heating threshold**:

```
V* > E[|v|³] / E[v²] = 1.596.
```

All three stochastic runs use V* = 1.0 < 1.596, so they are predicted to
**cool** the gas from the N(0,1) initial distribution.
The rate-weighted dE/dt values are:

| Label | dE/dt (rate-weighted) |
|-------|----------------------|
| weak | −0.0039 |
| moderate | −0.0197 |
| strong | −0.0394 |

### 2.3 Truncation artifact: frozen particles

The `max(s_out, 0)` truncation is a second, independent cooling mechanism.
Whenever s_out < 0 (noise dominates), the particle is assigned speed 0
and velocity v⁺ = 0.  A particle with v = 0 **never reaches either wall**
and is permanently frozen at its current position.

This creates an absorbing state at v = 0.
The probability of a single right-wall hit producing speed 0 is

```
P(s_out < 0) = Φ(−[(1−η)s + ηV*] / √(2D_w)),
```

which is appreciable for slow particles (s << V*).
Once frozen, a particle contributes zero to E(t) and drags the
measured mean energy downward with every new freezing event.

The effect is **strongest for large η and D_w** (the strong kernel).

---

## 3. Experimental results

### 3.1 Energy vs time

| Run | E(0) | E(25) | E(50) | E(75) | E(100) | log–log slope α |
|-----|------|-------|-------|-------|--------|-----------------|
| weak | 0.495 | 0.454 | 0.415 | 0.399 | **0.388** | −0.10 |
| moderate | 0.495 | 0.387 | 0.288 | 0.244 | **0.188** | −0.71 |
| strong | 0.495 | 0.284 | 0.215 | 0.143 | **0.088** | −1.16 |
| deterministic | 0.495 | **1.339** | 1.085 | 1.169 | **1.033** | −0.07 |

The deterministic Fermi wall heats the gas by a factor of ~2 over t = 100.
All three stochastic runs cool the gas, with cooling rate growing with η,
consistent with the rate-weighted analysis and the truncation argument.

### 3.2 Final velocity distributions

Strikingly, by t = 100 a large fraction of particles have been frozen at v ≈ 0:

| Run | fraction |v| < 0.001 | median speed |
|-----|--------------------------|--------------|
| weak | 0.0% | 0.491 |
| moderate | 49.1% | 0.004 |
| strong | 67.9% | 0.000 |
| deterministic | 2.6% | 0.021 |

The strong kernel has frozen two thirds of the gas.
The remaining active particles have a heavy tail (max |v| ~ 3),
because only the fastest particles continue to reach the right wall.

For the deterministic Fermi wall the speed distribution is very different:
most particles are slow (median speed 0.02) but a large-amplitude tail
extends to |v| ~ 10.  This is the hallmark of Fermi acceleration: a
small fraction of particles are repeatedly accelerated, pulling E(t) up
while the bulk remains slow.

---

## 4. Interpretation and diagnosis

### What works

- The **deterministic Fermi wall** is functioning correctly and produces
  the expected stochastic heating behavior.
- The **collision detection, reflection, and position folding** are correct
  (no domain-escape events reported by diagnostics).

### What needs fixing for heating to appear in the stochastic runs

**Problem 1: V* = 1.0 is below the rate-weighted heating threshold (1.596).**

Fix: use V* ≥ 1.6 when starting from N(0,1).
The base config already sets `V_star: 2.0`, which gives a predicted
rate-weighted rate of +0.10 — clearly in the heating regime.

**Problem 2: the `max(s_out, 0)` truncation creates frozen particles.**

Fix: replace the truncation with a reflecting lower bound or use a
kernel that is guaranteed to produce s_out > 0, for example a
half-Maxwellian or a log-normal speed distribution.
A minimal fix is to replace:

```python
speed_out = np.maximum(speed_out, 0.0)
```

with

```python
speed_out = np.abs(speed_out)   # folded reflection at 0
```

This preserves the noise energy and eliminates the absorbing state.

---

## 5. Predictions for the corrected runs

With V* = 2.0 and the reflecting-boundary fix, the rate-weighted
heating rate starting from N(0,1) is:

```
dE/dt ∝  −η · 1.596  +  η · 2.0 · 1.0  +  D_w · 0.798
       =  0.404 η  +  0.798 D_w  >  0   for all η, D_w > 0.
```

The gas should heat monotonically.  The equilibrium energy becomes

```
E_eq = V*²/2 + D_w/η = 2.0 + 0.5 = 2.5   (base params η=0.15, D_w=0.05).
```

This is a factor of 5 above the initial temperature — a clear heating signal.

---

## 6. Connection to the PDMP framework

The rate-weighted threshold is directly related to the theoretical
prediction for heating efficiency in the PDMP picture.  When bulk
collisions are present (Experiments B–D), the survival factor

```
χ ≈ μ / (μ + ν)
```

(fraction of particles that mix before the next collision) multiplies
the effective wall energy input.  In Experiment A, χ = 1 (no collisions),
so wall heating is maximal.  The heating-threshold condition V* > 1.596
must be satisfied before adding collisions; otherwise collisions act on
a kernel that is already cooling, and the effect of χ < 1 cannot be
cleanly isolated.

---

*Generated 2026-03-13 from simulation output in this directory.*
