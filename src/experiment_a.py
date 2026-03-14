"""Experiment A: wall heating only (no bulk collisions).

Runs four parameter sets and produces a single comparison figure:
  - weak kernel      (eta=0.02, D_w=0.01)
  - moderate kernel  (eta=0.10, D_w=0.05)
  - strong kernel    (eta=0.20, D_w=0.10)
  - deterministic Fermi wall (sigma=0 always, moving-wall reflection)

Usage (from repo root):
    python src/experiment_a.py --out-dir results/experiment_a
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
    ("weak",          "experiment_a_weak.yaml",          "tab:blue"),
    ("moderate",      "experiment_a_moderate.yaml",       "tab:orange"),
    ("strong",        "experiment_a_strong.yaml",         "tab:red"),
    ("deterministic", "experiment_a_deterministic.yaml",  "tab:green"),
]


def fit_power_law(t: np.ndarray, E: np.ndarray, frac: float = 0.5):
    """Fit E ~ t^alpha on the second half of the time series (log-log)."""
    start = int(len(t) * frac)
    t_fit = t[start:]
    E_fit = E[start:]
    mask  = (t_fit > 0) & (E_fit > 0)
    if mask.sum() < 5:
        return float("nan")
    coeffs = np.polyfit(np.log(t_fit[mask]), np.log(E_fit[mask]), 1)
    return coeffs[0]  # exponent alpha


def run_all(out_dir: Path, verbose: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for label, cfg_file, _ in RUNS:
        cfg_path = CONFIG_DIR / cfg_file
        cfg = load_config(cfg_path)
        print(f"\n=== {label} ===")
        data = run(cfg, verbose=verbose)
        results[label] = data
        np.savez(out_dir / f"{label}.npz",
                 t=data["t"], E=data["E"], p_mix=data["p_mix"],
                 v_final=data["particles_final"]["v"])

    return results


def plot_energy_comparison(results: dict, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax_lin = axes[0]
    ax_log = axes[1]

    for (label, _, color) in RUNS:
        data = results[label]
        t, E = data["t"], data["E"]
        alpha = fit_power_law(t, E)
        tag = f"{label}  (α≈{alpha:.2f})" if not np.isnan(alpha) else label

        ax_lin.plot(t, E, color=color, label=tag, lw=1.5)
        mask = (t > 0) & (E > 0)
        ax_log.plot(t[mask], E[mask], color=color, label=tag, lw=1.5)

    ax_lin.set_xlabel("Time t")
    ax_lin.set_ylabel("Mean kinetic energy E(t)")
    ax_lin.set_title("Energy growth (linear)")
    ax_lin.legend(fontsize=8)

    ax_log.set_xscale("log")
    ax_log.set_yscale("log")
    ax_log.set_xlabel("Time t")
    ax_log.set_ylabel("E(t)")
    ax_log.set_title("Energy growth (log–log, slope = heating exponent α)")
    ax_log.legend(fontsize=8)

    fig.tight_layout()
    path = out_dir / "energy_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_velocity_hists(results: dict, out_dir: Path):
    fig, axes = plt.subplots(1, len(RUNS), figsize=(14, 3), sharey=False)

    for ax, (label, _, color) in zip(axes, RUNS):
        v = results[label]["particles_final"]["v"]
        t_final = results[label]["t"][-1]
        ax.hist(v, bins=60, density=True, color=color, alpha=0.75,
                edgecolor="white", lw=0.3)
        ax.set_title(f"{label}\nt={t_final:.0f}", fontsize=9)
        ax.set_xlabel("v")
        if ax is axes[0]:
            ax.set_ylabel("density")

    fig.suptitle("Final velocity distributions — Experiment A", fontsize=11)
    fig.tight_layout()
    path = out_dir / "velocity_hists.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def print_summary(results: dict):
    print("\n=== Experiment A summary ===")
    print(f"{'label':<16}  {'E_init':>8}  {'E_final':>8}  {'ratio':>8}  {'alpha':>6}")
    print("-" * 55)
    for label, _, _ in RUNS:
        data  = results[label]
        E     = data["E"]
        t     = data["t"]
        E0    = E[0]
        Ef    = E[-1]
        ratio = Ef / E0 if E0 > 0 else float("nan")
        alpha = fit_power_law(t, E)
        print(f"{label:<16}  {E0:8.4f}  {Ef:8.4f}  {ratio:8.2f}  {alpha:6.3f}")


def main():
    parser = argparse.ArgumentParser(description="Experiment A: wall heating only")
    parser.add_argument("--out-dir", default="../results/experiment_a")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    results = run_all(out_dir, verbose=not args.quiet)
    plot_energy_comparison(results, out_dir)
    plot_velocity_hists(results, out_dir)
    print_summary(results)
    print(f"\nAll outputs in {out_dir}/")


if __name__ == "__main__":
    main()
