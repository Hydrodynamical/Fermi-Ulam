"""Render a compact dashboard for the real wall-bank and hazard dataset.

The dashboard is meant to answer, at a glance:
  1. What does the collision-free Fermi trajectory bank look like?
  2. Which energy bins are classified as proxy-valid / chaotic?
  3. How much of the row-wise hazard dataset comes from each regime?
  4. Is there visible age dependence in the empirical next-bounce entry hazard?

Usage
-----
python calibration/plot_real_wall_dashboard.py \
    --bank results/real_wall_bank_smoke/wall_bank.npz \
    --dataset results/real_wall_hazard_smoke/dataset.npz \
    --out results/real_wall_hazard_smoke/dashboard.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _compute_proxy_fraction_by_bin(u_bin_idx: np.ndarray, proxy_labels: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    flat_bins = u_bin_idx.reshape(-1)
    flat_proxy = proxy_labels.reshape(-1).astype(np.float64)
    counts = np.bincount(flat_bins, minlength=n_bins).astype(np.int64)
    proxy_sum = np.bincount(flat_bins, weights=flat_proxy, minlength=n_bins)
    frac = np.where(counts > 0, proxy_sum / counts, np.nan)
    return frac, counts


def _plot_poincare(ax, bank: dict, n_particles: int, max_points: int) -> None:
    u_traj = bank["u_traj"]
    psi_traj = bank["psi_traj"]
    proxy = bank["proxy_labels"].astype(bool)

    n_particles = min(n_particles, u_traj.shape[0])
    hits_per_particle = u_traj.shape[1]
    stride = max(1, int(np.ceil(n_particles * hits_per_particle / max_points)))

    psi = psi_traj[:n_particles, ::stride].reshape(-1)
    u = u_traj[:n_particles, ::stride].reshape(-1)
    m = proxy[:n_particles, ::stride].reshape(-1)

    ax.scatter(psi[~m], u[~m], s=0.35, alpha=0.18, color="tab:blue", rasterized=True, label="M=0")
    ax.scatter(psi[m], u[m], s=0.35, alpha=0.18, color="tab:red", rasterized=True, label="M=1")
    ax.set_yscale("log")
    ax.set_xlim(0.0, 2.0 * np.pi)
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_xlabel(r"phase $\psi$")
    ax.set_ylabel(r"energy $u$")
    ax.set_title("Poincare section with proxy labels")
    ax.legend(loc="upper right", fontsize=8, markerscale=8)


def _plot_bin_diagnostics(ax, bank: dict) -> None:
    u_centers = bank["u_centers"]
    entropy_norm = bank["entropy_norm"]
    tau_lag = bank["tau_lag"]
    chaos_mask = bank["chaos_mask"].astype(bool)

    ax.plot(u_centers, entropy_norm, "o-", color="tab:green", lw=1.2, ms=4, label=r"entropy $H_{\rm norm}$")
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("energy u")
    ax.set_ylabel(r"$H_{\rm norm}$")

    ax2 = ax.twinx()
    ax2.plot(u_centers, tau_lag, "s--", color="tab:orange", lw=1.2, ms=4, label=r"$\tau_{\rm lag}$")
    ax2.set_yscale("log")
    ax2.set_ylabel(r"$\tau_{\rm lag}$ [bounces]")

    chaotic_u = u_centers[chaos_mask]
    if len(chaotic_u):
        ax.scatter(chaotic_u, np.full(len(chaotic_u), 0.04), marker="|", s=160, color="black", label="chaos mask")

    lines = ax.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    if len(chaotic_u):
        handles, labs = ax.get_legend_handles_labels()
        lines += handles[-1:]
        labels += labs[-1:]
    ax.legend(lines, labels, loc="upper left", fontsize=8)
    ax.set_title("Bin diagnostics and chaos mask")


def _plot_proxy_fraction(ax, bank: dict) -> None:
    u_centers = bank["u_centers"]
    chaos_mask = bank["chaos_mask"].astype(bool)
    frac, counts = _compute_proxy_fraction_by_bin(
        bank["u_bin_idx"],
        bank["proxy_labels"],
        len(u_centers),
    )

    ax.plot(u_centers, frac, "o-", color="tab:red", lw=1.4, ms=4, label="proxy fraction")
    ax.scatter(u_centers[~chaos_mask], frac[~chaos_mask], color="tab:blue", s=18, zorder=3, label="start bins M=0")
    ax.scatter(u_centers[chaos_mask], frac[chaos_mask], color="black", s=18, zorder=3, label="chaotic bins")
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("energy u")
    ax.set_ylabel("fraction of bounces with M=1")

    ax2 = ax.twinx()
    ax2.bar(u_centers, counts, width=u_centers * 0.12, color="0.85", alpha=0.35, label="bin counts")
    ax2.set_yscale("log")
    ax2.set_ylabel("bounce count")

    lines = ax.get_lines()
    labels = [line.get_label() for line in lines]
    handles, labs = ax.get_legend_handles_labels()
    ax.legend(handles, labs, loc="lower right", fontsize=8)
    ax.set_title("Proxy occupancy by energy bin")


def _plot_proxy_raster(ax, bank: dict, n_particles: int, n_hits: int) -> None:
    proxy = bank["proxy_labels"].astype(bool)
    n_particles = min(n_particles, proxy.shape[0])
    n_hits = min(n_hits, proxy.shape[1])
    show = proxy[:n_particles, :n_hits].astype(float)

    im = ax.imshow(show, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=0.0, vmax=1.0)
    ax.set_xlabel("bounce index n")
    ax.set_ylabel("particle")
    ax.set_title("Proxy-state raster for early bounces")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["M=0", "M=1"])


def _plot_age_hazard(ax, dataset: dict, max_age_plot: int, min_count: int) -> None:
    u = dataset["X_full"][:, 0]
    age = dataset["age"]
    y = dataset["y"].astype(np.float64)

    valid = age <= max_age_plot
    u = u[valid]
    age = age[valid]
    y = y[valid]

    quantiles = np.quantile(u, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    labels = [
        f"low u [{quantiles[0]:.2f}, {quantiles[1]:.2f}]",
        f"mid u [{quantiles[1]:.2f}, {quantiles[2]:.2f}]",
        f"high u [{quantiles[2]:.2f}, {quantiles[3]:.2f}]",
    ]
    colors = ["tab:blue", "tab:orange", "tab:red"]

    for idx in range(3):
        lo = quantiles[idx]
        hi = quantiles[idx + 1]
        if idx < 2:
            group = (u >= lo) & (u < hi)
        else:
            group = (u >= lo) & (u <= hi)
        if not np.any(group):
            continue

        counts = np.bincount(age[group], minlength=max_age_plot + 1)
        hits = np.bincount(age[group], weights=y[group], minlength=max_age_plot + 1)
        hazard = np.where(counts >= min_count, hits / counts, np.nan)
        ax.plot(np.arange(max_age_plot + 1), hazard, lw=1.6, color=colors[idx], label=labels[idx])

    ax.set_xlabel("age a [bounces]")
    ax.set_ylabel(r"empirical next-bounce hazard $E[y\mid u,a]$")
    ax.set_ylim(bottom=0.0)
    ax.set_title("Age dependence by energy tercile")
    ax.legend(fontsize=7, loc="upper right")


def _plot_segment_lengths(ax, dataset: dict) -> None:
    seg_id = dataset["seg_id"]
    y = dataset["y"]

    seg_lengths = np.bincount(seg_id)
    seg_has_entry = np.bincount(seg_id, weights=y, minlength=len(seg_lengths)) > 0

    entry_lengths = seg_lengths[seg_has_entry]
    cens_lengths = seg_lengths[~seg_has_entry]
    max_len = int(np.percentile(seg_lengths, 99)) if len(seg_lengths) else 1
    bins = np.arange(1, max_len + 2)

    if len(cens_lengths):
        ax.hist(cens_lengths, bins=bins, density=True, alpha=0.45, color="0.6", label="censored/truncated")
    if len(entry_lengths):
        ax.hist(entry_lengths, bins=bins, density=True, alpha=0.55, color="tab:red", label="entry observed")

    ax.set_xlim(1, max_len)
    ax.set_xlabel("segment length [rows / bounces]")
    ax.set_ylabel("density")
    ax.set_title("Non-proxy segment lengths")
    ax.legend(fontsize=8, loc="upper right")

    text = (
        f"rows: {len(y):,}\n"
        f"segments: {len(seg_lengths):,}\n"
        f"positive rate: {y.mean():.3f}\n"
        f"entry segments: {seg_has_entry.mean():.1%}"
    )
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )


def make_dashboard(
    bank_path: Path,
    dataset_path: Path,
    out_path: Path,
    n_particles_section: int,
    n_particles_raster: int,
    n_hits_raster: int,
    max_points_section: int,
    max_age_plot: int,
    min_count: int,
) -> None:
    bank = np.load(bank_path, allow_pickle=False)
    dataset = np.load(dataset_path, allow_pickle=False)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.subplots_adjust(hspace=0.32, wspace=0.34)

    _plot_poincare(axes[0, 0], bank, n_particles=n_particles_section, max_points=max_points_section)
    _plot_bin_diagnostics(axes[0, 1], bank)
    _plot_proxy_fraction(axes[0, 2], bank)
    _plot_proxy_raster(axes[1, 0], bank, n_particles=n_particles_raster, n_hits=n_hits_raster)
    _plot_age_hazard(axes[1, 1], dataset, max_age_plot=max_age_plot, min_count=min_count)
    _plot_segment_lengths(axes[1, 2], dataset)

    fig.suptitle("Real Fermi-map wall bank and hazard dataset", fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot a dashboard for the real wall-bank and hazard dataset")
    ap.add_argument("--bank", required=True, help="Path to wall_bank.npz")
    ap.add_argument("--dataset", required=True, help="Path to dataset.npz")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--section-particles", type=int, default=12)
    ap.add_argument("--raster-particles", type=int, default=40)
    ap.add_argument("--raster-hits", type=int, default=600)
    ap.add_argument("--section-max-points", type=int, default=60000)
    ap.add_argument("--max-age-plot", type=int, default=200)
    ap.add_argument("--min-count", type=int, default=200)
    args = ap.parse_args()

    make_dashboard(
        bank_path=Path(args.bank),
        dataset_path=Path(args.dataset),
        out_path=Path(args.out),
        n_particles_section=args.section_particles,
        n_particles_raster=args.raster_particles,
        n_hits_raster=args.raster_hits,
        max_points_section=args.section_max_points,
        max_age_plot=args.max_age_plot,
        min_count=args.min_count,
    )
    print(f"Saved dashboard to {args.out}")


if __name__ == "__main__":
    main()