"""
toy_generator.py — Ground-truth hidden-state segment simulator.

Physical analogy
----------------
Each segment corresponds to a particle's free flight between collisions.  At the
start of the segment an initial energy u is drawn, and a hidden class z is assigned:

  C (chaotic)   — fast, roughly constant entry hazard
  S (sticky)    — slow, age-decaying entry hazard (sticky KAM layer)
  T (trapped)   — zero entry hazard (KAM island)

The mixing target ("entry") corresponds to the particle first reaching the
binwise-Markov proxy in the real system.

Usage
-----
python toy/toy_generator.py --out-dir results/toy [--seed N] [--n-segments N]
"""

import argparse
import pathlib

import numpy as np
from scipy.special import expit  # logistic sigmoid


# ---------------------------------------------------------------------------
# Ground-truth parameters
# ---------------------------------------------------------------------------

# Class probability crossover scales
U_C     = 5.0    # exponential decay scale for p_C
U_T     = 20.0   # sigmoid midpoint for p_T
DELTA_T = 3.0    # sigmoid width for p_T

# Class-conditional hazard parameters
# C: h_C(u, a) = min(0.8, 0.4 + 0.2 * exp(-u))   — roughly constant, large
# S: h_S(u, a) = ALPHA_S / (a + BETA_S)            — decaying with age
# T: h_T = 0
ALPHA_S = 1.5
BETA_S  = 5.0


def class_probs(u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (p_C, p_S, p_T) for energy array u.  All non-negative, sum to 1."""
    p_c = np.exp(-u / U_C)
    p_t = expit((u - U_T) / DELTA_T)
    p_s = np.clip(1.0 - p_c - p_t, 0.0, 1.0)
    # renormalize to absorb any floating-point drift
    total = p_c + p_s + p_t
    return p_c / total, p_s / total, p_t / total


def hazard_C(u: float, a: int) -> float:
    return min(0.8, 0.4 + 0.2 * np.exp(-u))


def hazard_S(u: float, a: int) -> float:
    return ALPHA_S / (a + BETA_S)


def hazard_T(u: float, a: int) -> float:
    return 0.0


_HAZARD_FN = [hazard_C, hazard_S, hazard_T]  # indexed by z


# ---------------------------------------------------------------------------
# Ground-truth marginal quantities (vectorised over u_grid × a_arr)
# ---------------------------------------------------------------------------

def ground_truth_survival(u_grid: np.ndarray, a_arr: np.ndarray) -> np.ndarray:
    """
    S_true(u, a) = sum_z p_z(u) * prod_{k<a} (1 - h_z(u, k))

    Returns
    -------
    S : (G, A+1) float64  — S[g, 0] == 1 always
    """
    G = len(u_grid)
    A = len(a_arr) - 1   # a_arr = [0, 1, ..., max_age]
    p_c, p_s, p_t = class_probs(u_grid)   # each (G,)

    # Build per-class survival products incrementally
    # surv_z[g] = prod_{k<a}(1 - h_z(u_g, k))
    S = np.zeros((G, A + 1), dtype=np.float64)
    S[:, 0] = 1.0  # S(u, 0) = 1 always

    surv_C = np.ones(G)
    surv_S = np.ones(G)
    # surv_T stays at 1 (h_T = 0)

    for a in range(A):
        # Update class survivals for each u
        h_c_vec = np.minimum(0.8, 0.4 + 0.2 * np.exp(-u_grid))
        h_s_vec = ALPHA_S / (a + BETA_S)   # scalar (no u-dependence in h_S for now)
        surv_C *= (1.0 - h_c_vec)
        surv_S *= (1.0 - h_s_vec)
        # surv_T unchanged (h_T = 0)

        S[:, a + 1] = p_c * surv_C + p_s * surv_S + p_t * 1.0

    return S


def ground_truth_hazard(u_grid: np.ndarray, a_arr: np.ndarray) -> np.ndarray:
    """
    Marginal h_true(u, a) = -d/da log S_true(u, a)  (discrete version)
                           = (S(u,a) - S(u,a+1)) / S(u,a)

    Returns
    -------
    h : (G, A) float64  — undefined (NaN) where S(u,a) == 0
    """
    S = ground_truth_survival(u_grid, a_arr)
    with np.errstate(invalid='ignore', divide='ignore'):
        h = (S[:, :-1] - S[:, 1:]) / np.where(S[:, :-1] > 0, S[:, :-1], np.nan)
    return h


# ---------------------------------------------------------------------------
# Segment simulation
# ---------------------------------------------------------------------------

def generate_segments(cfg: dict) -> dict:
    """
    Simulate N_segments independent segments and return observed data.

    Parameters
    ----------
    cfg : dict with keys
        n_segments   int    (default 10_000)
        u_min        float  (default 0.5)
        u_max        float  (default 40.0)
        reset_prob   float  geometric reset p (default 0.02)
        max_age      int    upper bound on segment length (default 500)
        seed         int    (default 42)

    Returns
    -------
    dict with keys
        u0       (N,)   initial energy per segment
        z        (N,)   int8  hidden class (0=C, 1=S, 2=T)
        T        (N,)   float64  entry age (np.inf if censored)
        L        (N,)   int32  segment length (reset age)
    """
    N      = int(cfg.get("n_segments",  10_000))
    u_min  = float(cfg.get("u_min",      0.5))
    u_max  = float(cfg.get("u_max",      40.0))
    p_rst  = float(cfg.get("reset_prob", 0.02))
    max_ag = int(cfg.get("max_age",      500))
    seed   = int(cfg.get("seed",         42))

    rng = np.random.default_rng(seed)

    # Draw initial energies log-uniform
    log_u = rng.uniform(np.log(u_min), np.log(u_max), size=N)
    u0    = np.exp(log_u)

    # Draw hidden classes
    p_c, p_s, p_t = class_probs(u0)
    probs = np.stack([p_c, p_s, p_t], axis=1)   # (N, 3)
    z_all = np.array([rng.choice(3, p=probs[i]) for i in range(N)], dtype=np.int8)

    T_all = np.full(N, np.inf, dtype=np.float64)
    L_all = np.zeros(N, dtype=np.int32)

    for i in range(N):
        u = u0[i]
        z = int(z_all[i])
        h_fn = _HAZARD_FN[z]
        for a in range(max_ag):
            # Check reset first
            if rng.random() < p_rst:
                L_all[i] = a
                break
            # Check entry
            h = h_fn(u, a)
            if rng.random() < h:
                T_all[i] = float(a)
                L_all[i] = a
                break
        else:
            # Hit max_age without reset or entry — censored
            L_all[i] = max_ag

    return {
        "u0": u0,
        "z":  z_all,
        "T":  T_all,
        "L":  L_all,
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_segments(data: dict, path) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    print(f"  Saved segments to {path}")


def load_segments(path) -> dict:
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_summary(segs: dict, cfg: dict) -> None:
    u0, z, T, L = segs["u0"], segs["z"], segs["T"], segs["L"]
    N = len(u0)
    n_c = int((z == 0).sum());  n_s = int((z == 1).sum());  n_t = int((z == 2).sum())
    n_cens = int(np.isinf(T).sum())
    finite_T = T[np.isfinite(T)]
    tau_rmst = L.mean()  # proxy

    print(f"\n  Segments generated : {N:,}")
    print(f"  Hidden class fractions:")
    print(f"    C (chaotic)  : {n_c/N:.1%}")
    print(f"    S (sticky)   : {n_s/N:.1%}")
    print(f"    T (trapped)  : {n_t/N:.1%}")
    print(f"  Censoring rate : {n_cens/N:.1%}")
    if len(finite_T) > 0:
        print(f"  Entry age (finite T)  : "
              f"median={np.median(finite_T):.1f}  "
              f"p95={np.percentile(finite_T, 95):.1f}")
    print(f"  Mean segment length : {tau_rmst:.1f} bounces")


def main():
    ap = argparse.ArgumentParser(description="Generate toy hidden-state segments")
    ap.add_argument("--out-dir",    default="results/toy")
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--n-segments", type=int,   default=10_000)
    ap.add_argument("--u-min",      type=float, default=0.5)
    ap.add_argument("--u-max",      type=float, default=40.0)
    ap.add_argument("--reset-prob", type=float, default=0.02)
    ap.add_argument("--max-age",    type=int,   default=500)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "n_segments":  args.n_segments,
        "u_min":       args.u_min,
        "u_max":       args.u_max,
        "reset_prob":  args.reset_prob,
        "max_age":     args.max_age,
        "seed":        args.seed,
    }

    print("[1/2] Generating segments ...")
    segs = generate_segments(cfg)
    _print_summary(segs, cfg)
    save_segments(segs, out_dir / "segments.npz")

    print("[2/2] Computing ground-truth hazard and survival ...")
    max_age = int(cfg["max_age"])
    a_arr   = np.arange(max_age + 1)
    u_grid  = np.exp(np.linspace(np.log(cfg["u_min"]), np.log(cfg["u_max"]), 50))
    S_true  = ground_truth_survival(u_grid, a_arr)
    h_true  = ground_truth_hazard(u_grid, a_arr)
    tau_rmst_true = S_true.sum(axis=1)  # RMST up to max_age

    print(f"  τ_rmst range : {tau_rmst_true.min():.2f} – {tau_rmst_true.max():.2f} bounces")

    np.savez_compressed(
        out_dir / "ground_truth.npz",
        u_grid=u_grid,
        a_arr=a_arr,
        S_true=S_true,
        h_true=h_true,
        tau_rmst_true=tau_rmst_true,
    )
    print(f"  Saved ground truth to {out_dir / 'ground_truth.npz'}")
    print("Done.")


if __name__ == "__main__":
    main()
