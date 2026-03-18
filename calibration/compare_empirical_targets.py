"""Compare empirical first-passage targets across multiple mask definitions.

This diagnostic is intentionally model-free. It uses the same first-passage
machinery for each target and compares:

    1. binwise masks vs energy,
    2. empirical hazard heatmaps,
    3. representative survival curves,
    4. tau_rmst(u).

Usage
-----
python calibration/compare_empirical_targets.py \
    --bank results/real_wall_bank_medium/wall_bank.npz \
    --entry-mask results/real_wall_bank_medium/diagnostics/core_mask.npz \
    --retention-mask results/real_wall_bank_medium/diagnostics/core_mask_retention_H1_r0p6_cv0p5.npz \
    --out-dir results/real_wall_bank_medium/diagnostics/target_compare
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from calibration.first_passage import compute_residual_waiting_times, estimate_survival_hazard
from calibration.survival_dataset import build_labels_from_mask


def _load_mask(mask_path: str) -> np.ndarray:
    data = np.load(mask_path, allow_pickle=False)
    if "core_mask" in data.files:
        return data["core_mask"].astype(bool)
    if "mask" in data.files:
        return data["mask"].astype(bool)
    raise ValueError(f"No core_mask or mask array found in {mask_path}")


def _compute_target(bank: dict, labels: np.ndarray, max_age: int, n_min: int) -> dict:
    u_centers = bank["u_centers"]
    fp = compute_residual_waiting_times(labels, bank["u_traj"], bank["u_edges"], len(u_centers), max_age=max_age)
    sv = estimate_survival_hazard(fp["rwt_by_bin"], fp["counts"], max_age=max_age, n_min=n_min)
    return {**fp, **sv}


def _select_bins(mask: np.ndarray, tau_rmst: np.ndarray, n_sel: int = 4) -> np.ndarray:
    idx = np.where(~mask & np.isfinite(tau_rmst))[0]
    if len(idx) == 0:
        return np.array([], dtype=np.int32)
    if len(idx) <= n_sel:
        return idx
    return idx[np.linspace(0, len(idx) - 1, n_sel, dtype=int)]


def _heatmap(ax, Z, u_grid, a_arr, title, vmin=None, vmax=None, cmap="viridis") -> None:
    im = ax.imshow(
        np.nan_to_num(Z, nan=np.nan),
        origin="lower",
        aspect="auto",
        extent=[a_arr[0], a_arr[-1], 0, len(u_grid) - 1],
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    tick_idx = np.linspace(0, len(u_grid) - 1, min(6, len(u_grid)), dtype=int)
    ax.set_yticks(tick_idx)
    ax.set_yticklabels([f"{u_grid[i]:.2f}" for i in tick_idx])
    ax.set_xlabel("age a [bounces]")
    ax.set_ylabel("energy u")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def make_figure(
    out_path: Path,
    u_centers: np.ndarray,
    targets: list[dict],
    max_age: int,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    fig.subplots_adjust(hspace=0.38, wspace=0.34)

    colors = ["tab:blue", "tab:orange", "tab:red"]
    labels = [target["name"] for target in targets]

    ax = axes[0, 0]
    for color, target in zip(colors, targets):
        ax.semilogx(u_centers, target["mask"].astype(int), drawstyle="steps-mid", lw=1.8, color=color, label=target["name"])
    ax.set_xlabel("energy u")
    ax.set_ylabel("mask indicator")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Mask comparison")
    ax.legend(fontsize=8, loc="upper right")

    h_max = 0.0
    for target in targets:
        if np.isfinite(target["h"]).any():
            h_max = max(h_max, float(np.nanpercentile(target["h"], 98)))
    h_max = h_max if h_max > 0 else 1.0

    for j, target in enumerate(targets):
        _heatmap(axes[0, j], target["h"], u_centers, target["a_arr"][:-1], f"{target['name']} hazard", 0, h_max)

    for j, target in enumerate(targets):
        ax = axes[1, j]
        sel = _select_bins(target["mask"], target["tau_rmst"], n_sel=4)
        palette = plt.cm.plasma(np.linspace(0.15, 0.85, max(len(sel), 1)))
        for color, idx in zip(palette, sel):
            ax.plot(target["a_arr"], target["S"][idx], lw=1.8, color=color, label=f"u={u_centers[idx]:.2f}")
        ax.set_xlabel("age a [bounces]")
        ax.set_ylabel("S(u, a)")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{target['name']} survival")
        if len(sel):
            ax.legend(fontsize=7, loc="upper right")

    ax = axes[2, 0]
    for color, target in zip(colors, targets):
        ax.semilogx(u_centers, target["tau_rmst"], "o-", ms=4, lw=1.5, color=color, label=target["name"])
    ax.set_xlabel("energy u")
    ax.set_ylabel("tau_rmst [bounces]")
    ax.set_title("RMST comparison")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[2, 1]
    for color, target in zip(colors, targets):
        plateau = target["S"][:, -1]
        ax.semilogx(u_centers, plateau, "o-", ms=4, lw=1.5, color=color, label=target["name"])
    ax.set_xlabel("energy u")
    ax.set_ylabel(f"S(u, {max_age})")
    ax.set_title("Truncated plateau comparison")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[2, 2]
    x = np.arange(len(targets))
    n_bins = [int(np.sum(target["mask"])) for target in targets]
    n_support = [int(np.sum(np.isfinite(target["tau_rmst"]))) for target in targets]
    width = 0.35
    ax.bar(x - width / 2, n_bins, width=width, color=colors, alpha=0.85, label="mask bins")
    ax.bar(x + width / 2, n_support, width=width, color="0.6", alpha=0.75, label="supported start bins")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("count")
    ax.set_title("Mask size and empirical support")
    ax.legend(fontsize=8)

    fig.suptitle("Empirical first-passage target comparison", fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare empirical first-passage targets across masks")
    ap.add_argument("--bank", required=True, help="Path to wall_bank.npz")
    ap.add_argument("--entry-mask", required=True, help="Path to entry-core mask npz")
    ap.add_argument("--retention-mask", required=True, help="Path to retention-core mask npz")
    ap.add_argument("--out-dir", required=True, help="Directory for comparison outputs")
    ap.add_argument("--max-age", type=int, default=200)
    ap.add_argument("--n-min", type=int, default=30)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Loading wall bank from {args.bank} ...")
    bank = np.load(args.bank, allow_pickle=False)
    u_centers = bank["u_centers"]
    proxy_labels = bank["proxy_labels"].astype(bool)
    entry_mask = _load_mask(args.entry_mask)
    retention_mask = _load_mask(args.retention_mask)
    entry_labels = build_labels_from_mask(bank["u_bin_idx"], entry_mask)
    retention_labels = build_labels_from_mask(bank["u_bin_idx"], retention_mask)

    print(f"[2/3] Computing empirical first-passage curves for all three targets ...")
    targets = [
        {"name": "proxy", "mask": bank["chaos_mask"].astype(bool), **_compute_target(bank, proxy_labels, args.max_age, args.n_min)},
        {"name": "entry-core", "mask": entry_mask, **_compute_target(bank, entry_labels, args.max_age, args.n_min)},
        {"name": "retention-core", "mask": retention_mask, **_compute_target(bank, retention_labels, args.max_age, args.n_min)},
    ]

    print("\n  --- Target summary ---")
    for target in targets:
        supported = int(np.sum(np.isfinite(target["tau_rmst"])))
        print(
            f"  {target['name']:<15} mask_bins={int(np.sum(target['mask'])):<3d}  "
            f"supported_bins={supported:<3d}  tau_rmst_range="
            f"[{np.nanmin(target['tau_rmst']):.2f}, {np.nanmax(target['tau_rmst']):.2f}]"
        )

    np.savez_compressed(
        out_dir / "empirical_target_comparison.npz",
        u_centers=u_centers,
        max_age=np.array([args.max_age], dtype=np.int32),
        proxy_mask=targets[0]["mask"].astype(np.int8),
        proxy_S=targets[0]["S"],
        proxy_h=targets[0]["h"],
        proxy_tau_rmst=targets[0]["tau_rmst"],
        entry_core_mask=targets[1]["mask"].astype(np.int8),
        entry_core_S=targets[1]["S"],
        entry_core_h=targets[1]["h"],
        entry_core_tau_rmst=targets[1]["tau_rmst"],
        retention_core_mask=targets[2]["mask"].astype(np.int8),
        retention_core_S=targets[2]["S"],
        retention_core_h=targets[2]["h"],
        retention_core_tau_rmst=targets[2]["tau_rmst"],
    )

    print(f"[3/3] Rendering comparison figure to {out_dir / 'empirical_target_comparison.png'} ...")
    make_figure(out_dir / "empirical_target_comparison.png", u_centers, targets, args.max_age)
    print("Done.")


if __name__ == "__main__":
    main()