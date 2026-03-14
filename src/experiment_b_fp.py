"""Experiment B (FP/OU wall): collision suppression of wall heating.

Tests the core PDMP prediction:

    E_inf(lambda0) decreases as lambda0 increases,
    with suppression governed by chi = mu / (mu + nu).

Four runs — identical FP moderate wall, only lambda0 varies:
  lambda0 = 0.0   (no collisions, baseline)
  lambda0 = 0.1   (weak collisions)
  lambda0 = 0.5   (moderate collisions)
  lambda0 = 1.0   (strong collisions)

Wall: kappa_w=3.0, wall_dtau=0.05, V_star=2.0, D_w=0.05
Mixing: m0=0.1, m1=0.3  (tau_mix(v) = 0.1 + 0.3/(|v|+0.2))
t_end: 200  (long enough to reach equilibrium at all lambda0 values)

Chi is measured two ways:
  - p_mix(t): instantaneous fraction of mixed particles (sigma=1)
  - chi_direct: cumulative fraction of collision events where sigma=1 at impact

Three theory predictions compared:
  chi_scalar  : mu/(mu+nu_eff) at V*, bulk n_bar  [crude mean-field]
  chi_exp_w   : |v|-weighted mu_i/(mu_i+nu_i)    [exponential-clock, local nu]
  chi_thresh_w: |v|-weighted exp(-nu_i*tau_mix_i) [threshold-clock, matches code]

Usage (from repo root):
    python src/experiment_b_fp.py --out-dir results/experiment_b_fp
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import load_config
from simulator import run

CONFIG_DIR = Path(__file__).parent.parent / "config"

RUNS = [
    (0.0, "experiment_b_fp_l0.yaml",   "tab:blue"),
    (0.1, "experiment_b_fp_l01.yaml",  "tab:orange"),
    (0.5, "experiment_b_fp_l05.yaml",  "tab:red"),
    (1.0, "experiment_b_fp_l10.yaml",  "tab:purple"),
]


def equilibrium_slice(arr: np.ndarray, frac: float = 0.4) -> float:
    """Nanmean over the last frac of the time series — equilibrium estimate."""
    return float(np.nanmean(arr[int(len(arr) * (1 - frac)):]))


def chi_theory(lambda0: float, cfg: dict) -> float:
    """Scalar mean-field chi = mu/(mu+nu_eff) evaluated at V*, bulk n_bar."""
    V_star  = cfg["V_star"]
    v_floor = cfg["v_mix_floor"]
    v_min   = cfg["v_min"]
    n_bar   = 1.0 / cfg["L"]

    tau_mix = cfg["m0"] + cfg["m1"] / (V_star + v_floor)
    mu      = 1.0 / tau_mix
    nu_eff  = lambda0 * n_bar * (V_star + v_min)
    return mu / (mu + nu_eff) if (mu + nu_eff) > 0 else 1.0


def chi_theory_weighted(x: np.ndarray, v: np.ndarray,
                        lambda0: float, cfg: dict) -> dict:
    """
    Wall-flux-weighted chi using the actual equilibrium particle distribution.

    Per-particle local collision rate (reuses collisions.compute_local_density):
      nu_i = lambda0 * n_loc(x_i) * (|v_i - vbar_loc(x_i)| + v_min)

    Per-particle mixing time (reuses mixing.mixing_time):
      tau_i = m0 + m1 / (|v_i| + v_mix_floor)

    Two clock laws:
      chi_exp_i    = mu_i / (mu_i + nu_i)    [exponential clocks]
      chi_thresh_i = exp(-nu_i * tau_i)       [deterministic threshold — matches code]

    Wall-hitting flux weight:
      w_i = |v_i|   (proportional to right-wall encounter rate 1/(round-trip time))

    Returns dict with keys 'exp' and 'thresh'.
    """
    from collisions import compute_local_density
    from mixing import mixing_time

    class _P:
        pass
    p = _P(); p.x = x; p.v = v; p.N = len(v)

    n_loc, vbar_loc, _ = compute_local_density(p, cfg)
    nu  = lambda0 * n_loc * (np.abs(v - vbar_loc) + cfg["v_min"])
    tau = mixing_time(v, cfg)
    mu  = 1.0 / np.maximum(tau, 1e-12)

    chi_exp    = mu / (mu + nu)
    chi_thresh = np.exp(-nu * tau)

    w = np.abs(v)
    w_sum = w.sum()
    if w_sum == 0:
        return {"exp": float("nan"), "thresh": float("nan")}
    return {
        "exp":    float((chi_exp    * w).sum() / w_sum),
        "thresh": float((chi_thresh * w).sum() / w_sum),
    }


def run_all(out_dir: Path, verbose: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for lam, cfg_file, _ in RUNS:
        cfg_path = CONFIG_DIR / cfg_file
        cfg = load_config(cfg_path)
        print(f"\n=== lambda0={lam} ===")
        data = run(cfg, verbose=verbose)
        results[lam] = data
        np.savez(out_dir / f"lambda_{lam:.2f}.npz",
                 t=data["t"], E=data["E"], p_mix=data["p_mix"],
                 chi_direct=data["chi_direct"],
                 v_final=data["particles_final"]["v"],
                 x_final=data["particles_final"]["x"])
    return results


def plot_energy_curves(results: dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    for (lam, _, color) in RUNS:
        data = results[lam]
        t, E = data["t"], data["E"]
        E_eq = equilibrium_slice(E)
        ax.plot(t, E, color=color, lw=1.5, label=f"λ₀={lam}  (E_eq≈{E_eq:.3f})")

    ax.set_xlabel("Time t")
    ax.set_ylabel("E(t)")
    ax.set_title("Experiment B — energy vs time for varying collision rate λ₀")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = out_dir / "energy_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_mixing_curves(results: dict, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax_idx, key in enumerate(["p_mix", "chi_direct"]):
        ax = axes[ax_idx]
        for (lam, _, color) in RUNS:
            data = results[lam]
            t, y = data["t"], data[key]
            valid = np.isfinite(y)
            if valid.any():
                ax.plot(t[valid], y[valid], color=color, lw=1.5, label=f"λ₀={lam}")

        ax.set_xlabel("Time t")
        ax.set_ylim(0, 1.05)
        if ax_idx == 0:
            ax.set_ylabel("p_mix(t)  [fraction sigma=1]")
            ax.set_title("Instantaneous mixing fraction")
        else:
            ax.set_ylabel("chi_direct(t)  [cumul. P(mixed at collision)]")
            ax.set_title("Direct chi measurement from collision events")
        ax.legend(fontsize=9)

    fig.suptitle("Empirical survival factor χ — two measurements", fontsize=11)
    fig.tight_layout()
    path = out_dir / "chi_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_suppression_curve(results: dict, out_dir: Path):
    """The key plot: E_inf and chi vs lambda0, with three theory curves."""
    lambdas = np.array([lam for lam, _, _ in RUNS])
    E_inf   = np.array([equilibrium_slice(results[lam]["E"])                    for lam, _, _ in RUNS])
    chi_emp = np.array([equilibrium_slice(results[lam]["chi_direct"], frac=1.0) for lam, _, _ in RUNS])
    pmix_eq = np.array([equilibrium_slice(results[lam]["p_mix"])                for lam, _, _ in RUNS])

    cfg0 = load_config(CONFIG_DIR / RUNS[0][1])

    # Scalar mean-field theory (smooth curve)
    lam_smooth  = np.linspace(0, lambdas.max() * 1.1, 200)
    chi_scalar  = np.array([chi_theory(l, cfg0) for l in lam_smooth])
    E_inf_0     = E_inf[0]

    # Weighted theory: evaluated at equilibrium distribution of each run
    chi_exp_w    = []
    chi_thresh_w = []
    for lam, _, _ in RUNS:
        pf = results[lam]["particles_final"]
        w  = chi_theory_weighted(pf["x"], pf["v"], lam, cfg0)
        chi_exp_w.append(w["exp"])
        chi_thresh_w.append(w["thresh"])
    chi_exp_w    = np.array(chi_exp_w)
    chi_thresh_w = np.array(chi_thresh_w)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: E_inf vs lambda0
    ax = axes[0]
    ax.plot(lam_smooth, E_inf_0 * chi_scalar,  "k--", lw=1.2, label="E₀·χ_scalar")
    ax.plot(lambdas,    E_inf_0 * chi_thresh_w, "r:",  lw=1.8, label="E₀·χ_thresh_w")
    ax.scatter(lambdas, E_inf, zorder=5, s=70,
               c=[c for _, _, c in RUNS], label="E_inf (sim)")
    ax.set_xlabel("Collision rate λ₀")
    ax.set_ylabel("Equilibrium energy E_inf")
    ax.set_title("Collision suppression of wall heating")
    ax.legend(fontsize=9)

    # Right: chi vs lambda0
    ax = axes[1]
    ax.plot(lam_smooth, chi_scalar,   "k--", lw=1.2, label="χ_scalar  μ/(μ+ν) @ V*")
    ax.plot(lambdas,    chi_exp_w,    "b:",  lw=1.8, label="χ_exp_w   exp. clock, local ν")
    ax.plot(lambdas,    chi_thresh_w, "r:",  lw=1.8, label="χ_thresh_w  threshold clock (code)")
    ax.scatter(lambdas, chi_emp,   zorder=5, s=70, marker="o",
               c=[c for _, _, c in RUNS], label="χ_direct (sim)")
    ax.scatter(lambdas, pmix_eq,   zorder=5, s=70, marker="s",
               c=[c for _, _, c in RUNS], label="p_mix_eq (sim)", alpha=0.6)
    ax.set_xlabel("Collision rate λ₀")
    ax.set_ylabel("χ")
    ax.set_ylim(0, 1.1)
    ax.set_title("Survival factor χ — three theory predictions vs simulation")
    ax.legend(fontsize=8)

    fig.suptitle("Experiment B — collision suppression of FP/OU wall heating", fontsize=11)
    fig.tight_layout()
    path = out_dir / "suppression_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def print_summary(results: dict):
    cfg0 = load_config(CONFIG_DIR / RUNS[0][1])
    print("\n=== Experiment B summary ===")
    hdr = (f"{'λ₀':>5}  {'E_inf':>7}  {'E/E0':>6}  "
           f"{'chi_d':>7}  {'p_mix':>7}  "
           f"{'χ_scalar':>9}  {'χ_exp_w':>8}  {'χ_thresh_w':>11}")
    print(hdr)
    print("-" * len(hdr))
    E0 = equilibrium_slice(results[RUNS[0][0]]["E"])
    for lam, _, _ in RUNS:
        data = results[lam]
        pf   = data["particles_final"]
        E_eq = equilibrium_slice(data["E"])
        chi_d   = equilibrium_slice(data["chi_direct"], frac=1.0)
        pmix    = equilibrium_slice(data["p_mix"])
        chi_sc  = chi_theory(lam, cfg0)
        w       = chi_theory_weighted(pf["x"], pf["v"], lam, cfg0)
        print(f"{lam:5.2f}  {E_eq:7.4f}  {E_eq/E0:6.4f}  "
              f"{chi_d:7.4f}  {pmix:7.4f}  "
              f"{chi_sc:9.4f}  {w['exp']:8.4f}  {w['thresh']:11.4f}")


def main():
    parser = argparse.ArgumentParser(description="Experiment B: collision suppression")
    parser.add_argument("--out-dir", default="../results/experiment_b_fp")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    results = run_all(out_dir, verbose=not args.quiet)
    plot_energy_curves(results, out_dir)
    plot_mixing_curves(results, out_dir)
    plot_suppression_curve(results, out_dir)
    print_summary(results)
    print(f"\nAll outputs in {out_dir}/")


if __name__ == "__main__":
    main()
