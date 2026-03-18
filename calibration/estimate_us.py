"""Estimate the global-stochasticity threshold u_s from a stratified sweep.

The sweep is analyzed using worst-seed diagnostics over phase seeds at each u0:

    entropy_min(u0)
    coverage_min(u0)
    p_trap(u0)
    lag1_acf_max(u0)

The threshold estimate is the top of the largest contiguous low-energy prefix
that satisfies the chosen criteria.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def compute_us_diagnostics(
    sweep: dict,
    entropy_threshold: float = 0.95,
    coverage_threshold: float = 0.95,
    trap_fraction_threshold: float = 0.01,
    floor_quantile: float = 0.01,
) -> dict:
    """Compute per-u0 diagnostics and a robust prefix-based u_s estimate."""
    u0_grid = sweep["u0_grid"]
    phase_entropy = sweep["phase_entropy"]
    visited_fraction = sweep["visited_fraction"]
    lag1_acf = sweep["lag1_acf_cospsi"]

    entropy_min = np.nanmin(phase_entropy, axis=1)
    entropy_mean = np.nanmean(phase_entropy, axis=1)
    entropy_floor = np.nanquantile(phase_entropy, floor_quantile, axis=1)
    coverage_min = np.nanmin(visited_fraction, axis=1)
    coverage_mean = np.nanmean(visited_fraction, axis=1)
    coverage_floor = np.nanquantile(visited_fraction, floor_quantile, axis=1)
    lag1_acf_max = np.nanmax(np.abs(lag1_acf), axis=1)

    bad_seed = (phase_entropy < entropy_threshold) | (visited_fraction < coverage_threshold)
    p_trap = np.nanmean(bad_seed.astype(np.float64), axis=1)

    good = (
        np.isfinite(entropy_floor)
        & np.isfinite(coverage_floor)
        & np.isfinite(p_trap)
        & (entropy_floor >= entropy_threshold)
        & (coverage_floor >= coverage_threshold)
        & (p_trap <= trap_fraction_threshold)
    )

    prefix_good = np.zeros_like(good, dtype=bool)
    still_good = True
    for i, g in enumerate(good):
        still_good = still_good and bool(g)
        prefix_good[i] = still_good

    if np.any(prefix_good):
        us_idx = int(np.where(prefix_good)[0][-1])
        u_s = float(u0_grid[us_idx])
    else:
        us_idx = -1
        u_s = float("nan")

    return {
        "u0_grid": u0_grid,
        "entropy_min": entropy_min,
        "entropy_mean": entropy_mean,
        "entropy_floor": entropy_floor,
        "coverage_min": coverage_min,
        "coverage_mean": coverage_mean,
        "coverage_floor": coverage_floor,
        "lag1_acf_max": lag1_acf_max,
        "p_trap": p_trap,
        "good": good,
        "prefix_good": prefix_good,
        "u_s": u_s,
        "u_s_idx": us_idx,
        "entropy_threshold": entropy_threshold,
        "coverage_threshold": coverage_threshold,
        "trap_fraction_threshold": trap_fraction_threshold,
        "floor_quantile": floor_quantile,
    }


def make_us_diagnostic_figure(out_path: Path, diag: dict) -> None:
    """Render the 2x2 threshold-diagnostic summary."""
    u0 = diag["u0_grid"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.32, wspace=0.28)

    ax = axes[0, 0]
    ax.semilogx(u0, diag["entropy_min"], "o", ms=3, color="0.75", alpha=0.8, label="min")
    ax.semilogx(u0, diag["entropy_floor"], "o-", ms=4, lw=1.3, color="tab:green", label=f"q={diag['floor_quantile']:.2f}")
    ax.axhline(diag["entropy_threshold"], color="k", ls="--", lw=1.0)
    if np.isfinite(diag["u_s"]):
        ax.axvline(diag["u_s"], color="tab:red", ls=":", lw=1.2)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("initial energy u0")
    ax.set_ylabel("phase entropy")
    ax.set_title("Phase entropy floor vs worst seed")
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[0, 1]
    ax.semilogx(u0, diag["p_trap"], "o-", ms=4, lw=1.3, color="tab:blue")
    ax.axhline(diag["trap_fraction_threshold"], color="k", ls="--", lw=1.0)
    if np.isfinite(diag["u_s"]):
        ax.axvline(diag["u_s"], color="tab:red", ls=":", lw=1.2)
    ax.set_xlabel("initial energy u0")
    ax.set_ylabel("trapped-seed fraction")
    ax.set_title("Trapped fraction")

    ax = axes[1, 0]
    ax.semilogx(u0, diag["coverage_min"], "o", ms=3, color="0.75", alpha=0.8, label="min")
    ax.semilogx(u0, diag["coverage_floor"], "o-", ms=4, lw=1.3, color="tab:orange", label=f"q={diag['floor_quantile']:.2f}")
    ax.axhline(diag["coverage_threshold"], color="k", ls="--", lw=1.0)
    if np.isfinite(diag["u_s"]):
        ax.axvline(diag["u_s"], color="tab:red", ls=":", lw=1.2)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("initial energy u0")
    ax.set_ylabel("phase coverage")
    ax.set_title("Phase coverage floor vs worst seed")
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1, 1]
    good = diag["good"].astype(float)
    prefix = diag["prefix_good"].astype(float)
    ax.semilogx(u0, good, "o-", ms=4, lw=1.1, color="tab:purple", label="good(u0)")
    ax.semilogx(u0, prefix, "s--", ms=4, lw=1.1, color="tab:red", label="prefix good")
    ax2 = ax.twinx()
    ax2.semilogx(u0, diag["lag1_acf_max"], "^-", ms=4, lw=1.1, color="0.5", alpha=0.8, label=r"max |acf_1(cos psi)|")
    if np.isfinite(diag["u_s"]):
        ax.axvline(diag["u_s"], color="tab:red", ls=":", lw=1.2)
        ax.text(diag["u_s"], 1.02, fr"$u_s \approx {diag['u_s']:.3g}$", color="tab:red", fontsize=9, ha="left", va="bottom")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("initial energy u0")
    ax.set_ylabel("threshold pass indicator")
    ax2.set_ylabel(r"max |acf_1(cos psi)|")
    ax.set_title("Estimated global-stochasticity threshold")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, fontsize=8, loc="upper right")

    fig.suptitle("u_s threshold diagnostics", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def make_poincare_sections(out_path: Path, sweep: dict, diag: dict) -> None:
    """Render representative Poincare sections below, near, and above u_s."""
    u_tail = sweep["u_tail_thin"]
    psi_tail = sweep["psi_tail_thin"]
    u0 = sweep["u0_grid"]
    us_idx = int(diag["u_s_idx"])

    if us_idx < 0:
        idx_triplet = [0, max(0, len(u0) // 2), len(u0) - 1]
    else:
        delta = max(1, len(u0) // 20)
        below = max(0, us_idx - delta)
        near = us_idx
        above = min(len(u0) - 1, us_idx + delta)
        idx_triplet = [below, near, above]
        if len(set(idx_triplet)) < 3:
            idx_triplet = [max(0, us_idx - 1), us_idx, min(len(u0) - 1, us_idx + 1)]
        if len(set(idx_triplet)) < 3:
            idx_triplet = [0, us_idx, len(u0) - 1]

    titles = ["Below threshold", "Near threshold", "Above threshold"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.subplots_adjust(wspace=0.30)

    for ax, idx, title in zip(axes, idx_triplet, titles):
        psi = psi_tail[idx].reshape(-1)
        u = u_tail[idx].reshape(-1)
        valid = np.isfinite(psi) & np.isfinite(u)
        ax.scatter(psi[valid], u[valid], s=0.3, alpha=0.25, rasterized=True)
        ax.set_xlim(0.0, 2.0 * np.pi)
        ax.set_yscale("log")
        ax.set_xlabel(r"phase $\psi$")
        ax.set_ylabel("energy u")
        ax.set_title(f"{title}\n$u_0={u0[idx]:.3g}$")

    fig.suptitle("Representative Poincare sections from stratified sweep", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Estimate u_s from a dedicated stratified sweep")
    ap.add_argument("--data", required=True, help="Path to us_sweep.npz")
    ap.add_argument("--out-dir", default=None, help="Output directory; defaults to the sweep directory")
    ap.add_argument("--entropy-threshold", type=float, default=0.95)
    ap.add_argument("--coverage-threshold", type=float, default=0.95)
    ap.add_argument("--trap-fraction-threshold", type=float, default=0.01)
    ap.add_argument("--floor-quantile", type=float, default=0.01,
                    help="Quantile used for robust entropy/coverage floor diagnostics")
    args = ap.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out_dir) if args.out_dir is not None else data_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep = np.load(data_path, allow_pickle=False)

    print(f"[u_s estimate] Loading sweep from {data_path} ...")
    diag = compute_us_diagnostics(
        sweep,
        entropy_threshold=float(args.entropy_threshold),
        coverage_threshold=float(args.coverage_threshold),
        trap_fraction_threshold=float(args.trap_fraction_threshold),
        floor_quantile=float(args.floor_quantile),
    )

    np.savez_compressed(
        out_dir / "us_diagnostics.npz",
        u0_grid=diag["u0_grid"],
        entropy_min=diag["entropy_min"],
        entropy_mean=diag["entropy_mean"],
        entropy_floor=diag["entropy_floor"],
        coverage_min=diag["coverage_min"],
        coverage_mean=diag["coverage_mean"],
        coverage_floor=diag["coverage_floor"],
        lag1_acf_max=diag["lag1_acf_max"],
        p_trap=diag["p_trap"],
        good=diag["good"].astype(np.int8),
        prefix_good=diag["prefix_good"].astype(np.int8),
        u_s=np.array([diag["u_s"]], dtype=np.float64),
        u_s_idx=np.array([diag["u_s_idx"]], dtype=np.int32),
        entropy_threshold=np.array([diag["entropy_threshold"]], dtype=np.float64),
        coverage_threshold=np.array([diag["coverage_threshold"]], dtype=np.float64),
        trap_fraction_threshold=np.array([diag["trap_fraction_threshold"]], dtype=np.float64),
        floor_quantile=np.array([diag["floor_quantile"]], dtype=np.float64),
    )
    make_us_diagnostic_figure(out_dir / "us_diagnostics.png", diag)
    make_poincare_sections(out_dir / "us_poincare_sections.png", sweep, diag)

    if np.isfinite(diag["u_s"]):
        print(f"[u_s estimate] Estimated u_s ≈ {diag['u_s']:.4g} (grid index {diag['u_s_idx']})")
    else:
        print("[u_s estimate] No contiguous low-energy good region found.")
    print(f"[u_s estimate] Saved {out_dir / 'us_diagnostics.npz'}")
    print(f"[u_s estimate] Saved {out_dir / 'us_diagnostics.png'}")
    print(f"[u_s estimate] Saved {out_dir / 'us_poincare_sections.png'}")


if __name__ == "__main__":
    main()