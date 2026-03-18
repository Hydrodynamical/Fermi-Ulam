"""Evaluate learned hazard models against empirical real-data first-passage curves.

This script uses the exported real wall bank as the empirical source of truth.
It computes the residual waiting-time law into the proxy set directly from the
bank, reconstructs survival from the learned hazard models, and compares:

    S(u, a)
    h(u, a)
    tau_rmst(u)

Usage
-----
python calibration/evaluate_real_survival.py \
    --bank results/real_wall_bank_smoke/wall_bank.npz \
    --models results/real_wall_hazard_smoke/models \
    --out-dir results/real_wall_hazard_smoke/eval
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from calibration.first_passage import compute_residual_waiting_times, estimate_survival_hazard
from calibration.survival_dataset import build_labels_from_mask


def reconstruct_survival(model, u_grid: np.ndarray, a_arr: np.ndarray, full_features: bool) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct survival S(u, a) and hazard h(u, a) from a learned model."""
    n_u = len(u_grid)
    n_age = len(a_arr) - 1
    survival = np.ones((n_u, n_age + 1), dtype=np.float64)
    hazard = np.zeros((n_u, n_age), dtype=np.float64)

    for age in range(n_age):
        if full_features:
            X = np.column_stack([u_grid, np.full(n_u, np.log1p(age))])
        else:
            X = u_grid.reshape(-1, 1)
        h_age = model.predict_proba(X)[:, 1]
        hazard[:, age] = h_age
        survival[:, age + 1] = survival[:, age] * (1.0 - h_age)

    return survival, hazard


def rmst(survival: np.ndarray) -> np.ndarray:
    return np.sum(survival, axis=1)


def compute_metrics(
    empirical: dict,
    model_full: dict,
    model_base: dict,
    counts: np.ndarray,
) -> dict:
    """Compute summary metrics on bins with empirical support."""
    valid_tau = np.isfinite(empirical["tau_rmst"])
    valid_plateau = np.isfinite(empirical["S"][:, -1])
    valid_hazard = np.isfinite(empirical["h"])
    supported = counts > 0

    tau_mask = valid_tau & supported
    plateau_mask = valid_plateau & supported
    hazard_mask = valid_hazard & supported[:, None]
    surv_mask = np.isfinite(empirical["S"]) & supported[:, None]

    metrics = {
        "n_supported_bins": int(np.sum(tau_mask)),
        "rmst_mae_full": float(np.mean(np.abs(empirical["tau_rmst"][tau_mask] - model_full["tau_rmst"][tau_mask]))),
        "rmst_mae_base": float(np.mean(np.abs(empirical["tau_rmst"][tau_mask] - model_base["tau_rmst"][tau_mask]))),
        "plateau_mae_full": float(np.mean(np.abs(empirical["S"][plateau_mask, -1] - model_full["S"][plateau_mask, -1]))),
        "plateau_mae_base": float(np.mean(np.abs(empirical["S"][plateau_mask, -1] - model_base["S"][plateau_mask, -1]))),
        "hazard_mae_full": float(np.mean(np.abs(empirical["h"][hazard_mask] - model_full["h"][hazard_mask]))),
        "hazard_mae_base": float(np.mean(np.abs(empirical["h"][hazard_mask] - model_base["h"][hazard_mask]))),
        "survival_mae_full": float(np.mean(np.abs(empirical["S"][surv_mask] - model_full["S"][surv_mask]))),
        "survival_mae_base": float(np.mean(np.abs(empirical["S"][surv_mask] - model_base["S"][surv_mask]))),
    }
    return metrics


def _heatmap(ax, Z, u_grid, a_arr, title, vmin=None, vmax=None, cmap="viridis"):
    im = ax.imshow(
        Z,
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
    plt.colorbar(im, ax=ax)


def make_figure(
    u_grid: np.ndarray,
    a_arr: np.ndarray,
    empirical: dict,
    model_full: dict,
    model_base: dict,
    counts: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.subplots_adjust(hspace=0.40, wspace=0.38)

    h_true = np.nan_to_num(empirical["h"], nan=0.0)
    h_full = model_full["h"]
    h_base = model_base["h"]
    h_max = float(np.nanpercentile(empirical["h"], 98)) if np.isfinite(empirical["h"]).any() else 1.0

    _heatmap(axes[0, 0], h_true, u_grid, a_arr[:-1], "empirical h(u, a)", 0, h_max)
    _heatmap(axes[0, 1], h_full, u_grid, a_arr[:-1], "learned full h(u, a)", 0, h_max)
    _heatmap(axes[0, 2], np.abs(empirical["h"] - h_full), u_grid, a_arr[:-1], "|empirical - full|", 0, None, cmap="Reds")

    valid_bins = np.where(np.isfinite(empirical["tau_rmst"]) & (counts > 0))[0]
    if len(valid_bins) == 0:
        valid_bins = np.arange(len(u_grid))
    sel = valid_bins[np.linspace(0, len(valid_bins) - 1, min(5, len(valid_bins)), dtype=int)]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(sel)))

    for ax, model_s, label in [
        (axes[1, 0], model_full["S"], "full [u, log(1+a)]"),
        (axes[1, 1], model_base["S"], "base [u]"),
    ]:
        for color, idx in zip(colors, sel):
            ax.plot(a_arr, empirical["S"][idx], color=color, lw=2)
            ax.plot(a_arr, model_s[idx], color=color, lw=1.5, ls="--")
        ax.set_xlabel("age a [bounces]")
        ax.set_ylabel("S(u, a)")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Empirical vs {label}")
        handles = [plt.Line2D([0], [0], color=color, lw=2) for color in colors]
        labels = [f"u={u_grid[idx]:.2f}" for idx in sel]
        ax.legend(handles, labels, fontsize=7, loc="upper right")

    ax = axes[1, 2]
    ax.plot(u_grid, empirical["tau_rmst"], "k-", lw=2, label="empirical")
    ax.plot(u_grid, model_full["tau_rmst"], "b--", lw=1.5, label="full")
    ax.plot(u_grid, model_base["tau_rmst"], "r:", lw=1.5, label="base")
    ax.set_xscale("log")
    ax.set_xlabel("energy u")
    ax.set_ylabel("tau_rmst [bounces]")
    ax.set_title("tau_rmst(u)")
    ax.legend(fontsize=8)

    fig.suptitle("Real-data hazard evaluation", fontsize=12)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate learned hazard models on real wall-bank data")
    ap.add_argument("--bank", required=True, help="Path to wall_bank.npz")
    ap.add_argument("--models", required=True, help="Directory containing model_full.pkl and model_base.pkl")
    ap.add_argument("--out-dir", default="results/real_wall_hazard/eval")
    ap.add_argument("--max-age", type=int, default=200)
    ap.add_argument("--n-min", type=int, default=30, help="Minimum start count per bin for empirical survival")
    ap.add_argument(
        "--label-source",
        choices=["proxy", "core-mask"],
        default="proxy",
        help="Which mask-derived event definition to evaluate against",
    )
    ap.add_argument(
        "--mask-path",
        default=None,
        help="Path to diagnostics/core_mask.npz when --label-source=core-mask",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading wall bank from {args.bank} ...")
    bank = np.load(args.bank, allow_pickle=False)
    u_traj = bank["u_traj"]
    if args.label_source == "proxy":
        labels = bank["proxy_labels"].astype(bool)
    else:
        if args.mask_path is None:
            raise ValueError("--mask-path is required when --label-source=core-mask")
        mask_npz = np.load(args.mask_path, allow_pickle=False)
        if "core_mask" in mask_npz.files:
            mask = mask_npz["core_mask"].astype(bool)
        elif "mask" in mask_npz.files:
            mask = mask_npz["mask"].astype(bool)
        else:
            raise ValueError(f"No core_mask or mask array found in {args.mask_path}")
        labels = build_labels_from_mask(bank["u_bin_idx"], mask)
    u_edges = bank["u_edges"]
    u_centers = bank["u_centers"]
    n_bins = len(u_centers)
    print(f"  Label source     : {args.label_source}")

    print(f"[2/4] Computing empirical first-passage curves (max_age={args.max_age}) ...")
    fp = compute_residual_waiting_times(labels, u_traj, u_edges, n_bins, max_age=args.max_age)
    empirical = estimate_survival_hazard(fp["rwt_by_bin"], fp["counts"], args.max_age, n_min=args.n_min)

    print(f"[3/4] Loading models from {args.models} and reconstructing survival ...")
    models_dir = Path(args.models)
    model_full = joblib.load(models_dir / "model_full.pkl")
    model_base = joblib.load(models_dir / "model_base.pkl")
    a_arr = empirical["a_arr"]

    S_full, h_full = reconstruct_survival(model_full, u_centers, a_arr, full_features=True)
    S_base, h_base = reconstruct_survival(model_base, u_centers, a_arr, full_features=False)
    model_full_data = {"S": S_full, "h": h_full, "tau_rmst": rmst(S_full)}
    model_base_data = {"S": S_base, "h": h_base, "tau_rmst": rmst(S_base)}

    metrics = compute_metrics(empirical, model_full_data, model_base_data, fp["counts"])
    print("\n  --- Real-data survival metrics ---")
    print(f"  Supported bins     : {metrics['n_supported_bins']}")
    print(f"  RMST MAE full      : {metrics['rmst_mae_full']:.3f}")
    print(f"  RMST MAE base      : {metrics['rmst_mae_base']:.3f}")
    print(f"  Plateau MAE full   : {metrics['plateau_mae_full']:.4f}")
    print(f"  Plateau MAE base   : {metrics['plateau_mae_base']:.4f}")
    print(f"  Hazard MAE full    : {metrics['hazard_mae_full']:.4f}")
    print(f"  Hazard MAE base    : {metrics['hazard_mae_base']:.4f}")
    print(f"  Survival MAE full  : {metrics['survival_mae_full']:.4f}")
    print(f"  Survival MAE base  : {metrics['survival_mae_base']:.4f}")

    np.savez_compressed(
        out_dir / "real_survival_metrics.npz",
        label_source=np.array([args.label_source]),
        u_grid=u_centers,
        a_arr=a_arr,
        counts=fp["counts"],
        censored=fp["censored"],
        S_empirical=empirical["S"],
        h_empirical=empirical["h"],
        tau_rmst_empirical=empirical["tau_rmst"],
        S_full=S_full,
        h_full=h_full,
        tau_rmst_full=model_full_data["tau_rmst"],
        S_base=S_base,
        h_base=h_base,
        tau_rmst_base=model_base_data["tau_rmst"],
        n_supported_bins=metrics["n_supported_bins"],
        rmst_mae_full=metrics["rmst_mae_full"],
        rmst_mae_base=metrics["rmst_mae_base"],
        plateau_mae_full=metrics["plateau_mae_full"],
        plateau_mae_base=metrics["plateau_mae_base"],
        hazard_mae_full=metrics["hazard_mae_full"],
        hazard_mae_base=metrics["hazard_mae_base"],
        survival_mae_full=metrics["survival_mae_full"],
        survival_mae_base=metrics["survival_mae_base"],
    )

    print(f"[4/4] Rendering figure to {out_dir / 'real_survival_evaluation.png'} ...")
    make_figure(u_centers, a_arr, empirical, model_full_data, model_base_data, fp["counts"], out_dir / "real_survival_evaluation.png")
    print("Done.")


if __name__ == "__main__":
    main()