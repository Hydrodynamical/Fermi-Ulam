"""
evaluate_survival.py — Reconstruct survival curves from learned hazard models
and compare to ground truth.

Metrics (printed and saved):
    rmst_err_full  MAE of τ_rmst between truth and full model  over u_grid
    rmst_err_base  MAE of τ_rmst between truth and base model
    plateau_err    MAE of S_inf(u) = S(u, max_age) between truth and full model

Figure: toy_evaluation.png  (2 rows × 3 cols)
    (0,0)  h_true(u, a) heatmap
    (0,1)  h_hat_full(u, a) heatmap
    (0,2)  |h_true - h_hat_full| residual
    (1,0)  S curves: truth vs full  (5 selected u values)
    (1,1)  S curves: truth vs base  (5 selected u values)
    (1,2)  τ_rmst(u): truth / full / base (3 lines)

Usage
-----
python toy/evaluate_survival.py \
    --segments results/toy/segments.npz \
    --models results/toy \
    --out-dir results/toy
"""

import argparse
import pathlib

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Survival reconstruction
# ---------------------------------------------------------------------------

def reconstruct_survival(model, u_grid: np.ndarray, a_arr: np.ndarray,
                          full_features: bool) -> np.ndarray:
    """
    S_hat(u, a) = prod_{k=0}^{a-1} (1 - h_hat(u, k))

    Parameters
    ----------
    model       : fitted Pipeline (StandardScaler + MLPClassifier)
    u_grid      : (G,)  evaluation energies
    a_arr       : (A+1,)  [0, 1, ..., max_age]
    full_features : True → features [u, log(1+a)];  False → [u]

    Returns
    -------
    S_hat : (G, A+1)  S_hat[:, 0] == 1
    h_hat : (G, A)
    """
    G  = len(u_grid)
    A  = len(a_arr) - 1
    S  = np.ones((G, A + 1), dtype=np.float64)
    H  = np.zeros((G, A),    dtype=np.float64)

    for a in range(A):
        if full_features:
            X = np.column_stack([u_grid, np.full(G, np.log1p(a))])
        else:
            X = u_grid.reshape(-1, 1)
        h_hat = model.predict_proba(X)[:, 1]   # (G,)
        H[:, a]     = h_hat
        S[:, a + 1] = S[:, a] * (1.0 - h_hat)

    return S, H


def rmst(S: np.ndarray) -> np.ndarray:
    """Restricted Mean Survival Time = sum_a S[g, a].  Shape (G,)."""
    return S.sum(axis=1)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _heatmap(ax, Z, u_grid, a_arr, title, vmin=None, vmax=None, cmap="viridis"):
    """Plot Z (G × A) as a heatmap with u on y-axis (log) and a on x-axis."""
    A = Z.shape[1]
    im = ax.imshow(
        Z, origin="lower", aspect="auto",
        extent=[a_arr[0], a_arr[-1], 0, len(u_grid) - 1],
        vmin=vmin, vmax=vmax, cmap=cmap,
    )
    # Label y-axis with actual u values
    tick_idx = np.linspace(0, len(u_grid) - 1, 6, dtype=int)
    ax.set_yticks(tick_idx)
    ax.set_yticklabels([f"{u_grid[i]:.1f}" for i in tick_idx])
    ax.set_xlabel("age a (bounces)")
    ax.set_ylabel("energy u")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)


def make_figure(
    u_grid, a_arr,
    h_true, h_hat_full, h_hat_base,
    S_true, S_hat_full, S_hat_base,
    tau_rmst_true, tau_rmst_full, tau_rmst_base,
    out_path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.subplots_adjust(hspace=0.40, wspace=0.40)

    # ---- Row 0: hazard heatmaps ----
    h_max = float(np.nanpercentile(h_true, 98))
    _heatmap(axes[0, 0], h_true,              u_grid, a_arr[:-1], "h_true(u, a)",     0, h_max)
    _heatmap(axes[0, 1], h_hat_full,          u_grid, a_arr[:-1], "h_hat full(u, a)", 0, h_max)
    resid = np.abs(h_true - h_hat_full)
    _heatmap(axes[0, 2], resid, u_grid, a_arr[:-1], "|h_true − h_hat|", 0, None, cmap="Reds")

    # ---- Row 1, col 0-1: survival curves at 5 selected u values ----
    u_sel_idx = np.linspace(0, len(u_grid) - 1, 5, dtype=int)
    colors    = plt.cm.plasma(np.linspace(0.1, 0.9, 5))

    for ax, S_hat, label in [
        (axes[1, 0], S_hat_full, "full [u, log(1+a)]"),
        (axes[1, 1], S_hat_base, "base [u]"),
    ]:
        for j, gi in enumerate(u_sel_idx):
            c  = colors[j]
            uv = u_grid[gi]
            ax.plot(a_arr, S_true[gi],    color=c, lw=2,   label=f"u={uv:.1f} truth")
            ax.plot(a_arr, S_hat[gi],     color=c, lw=1.5, ls="--", label=f"u={uv:.1f} model")
        ax.set_xlabel("age a (bounces)")
        ax.set_ylabel("S(u, a)")
        ax.set_title(f"Survival: truth vs {label}")
        ax.set_ylim(0, 1.05)
        # Compact legend
        handles = ax.get_lines()[::2]
        labels_ = [f"u={u_grid[gi]:.1f}" for gi in u_sel_idx]
        ax.legend(handles, labels_, fontsize=7, loc="upper right")

    # ---- Row 1, col 2: τ_rmst(u) ----
    ax = axes[1, 2]
    ax.plot(u_grid, tau_rmst_true, "k-",  lw=2, label="truth")
    ax.plot(u_grid, tau_rmst_full, "b--", lw=1.5, label="full model")
    ax.plot(u_grid, tau_rmst_base, "r:",  lw=1.5, label="base model")
    ax.set_xscale("log")
    ax.set_xlabel("energy u")
    ax.set_ylabel("τ_rmst (bounces)")
    ax.set_title("τ_rmst(u): truth vs models")
    ax.legend(fontsize=8)

    fig.suptitle("Toy hazard benchmark — Version 1 (static u)", fontsize=11)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate toy hazard models")
    ap.add_argument("--segments",  required=True, help="Path to segments.npz")
    ap.add_argument("--models",    required=True, help="Directory containing model_*.pkl")
    ap.add_argument("--out-dir",   default="results/toy")
    ap.add_argument("--max-age",   type=int, default=200,
                    help="Truncate survival reconstruction at this age")
    args = ap.parse_args()

    out_dir    = pathlib.Path(args.out_dir)
    models_dir = pathlib.Path(args.models)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load ground truth
    gt_path = models_dir / "ground_truth.npz"
    print(f"[1/4] Loading ground truth from {gt_path} ...")
    gt = np.load(gt_path, allow_pickle=False)
    u_grid         = gt["u_grid"]
    a_arr_full     = gt["a_arr"]
    S_true_full    = gt["S_true"]
    h_true_full    = gt["h_true"]

    # Restrict to max_age
    max_age = min(args.max_age, len(a_arr_full) - 1)
    a_arr   = a_arr_full[:max_age + 1]
    S_true  = S_true_full[:, :max_age + 1]
    h_true  = h_true_full[:, :max_age]

    # Load models
    print(f"[2/4] Loading models from {models_dir} ...")
    model_full = joblib.load(models_dir / "model_full.pkl")
    model_base = joblib.load(models_dir / "model_base.pkl")

    # Reconstruct survival
    print("[3/4] Reconstructing survival curves ...")
    S_hat_full, h_hat_full = reconstruct_survival(model_full, u_grid, a_arr, full_features=True)
    S_hat_base, h_hat_base = reconstruct_survival(model_base, u_grid, a_arr, full_features=False)

    # Compute RMST
    tau_true = rmst(S_true)
    tau_full = rmst(S_hat_full)
    tau_base = rmst(S_hat_base)

    rmst_err_full  = float(np.mean(np.abs(tau_true - tau_full)))
    rmst_err_base  = float(np.mean(np.abs(tau_true - tau_base)))
    plateau_err    = float(np.mean(np.abs(S_true[:, -1] - S_hat_full[:, -1])))

    print(f"\n  --- Survival metrics (over {len(u_grid)} u values) ---")
    print(f"  RMST MAE  full  : {rmst_err_full:.3f} bounces")
    print(f"  RMST MAE  base  : {rmst_err_base:.3f} bounces")
    print(f"  Plateau MAE full: {plateau_err:.4f}")

    # Save metrics
    np.savez(
        out_dir / "survival_metrics.npz",
        u_grid=u_grid,
        tau_rmst_true=tau_true,
        tau_rmst_full=tau_full,
        tau_rmst_base=tau_base,
        S_true=S_true,
        S_hat_full=S_hat_full,
        S_hat_base=S_hat_base,
        h_true=h_true,
        h_hat_full=h_hat_full,
        h_hat_base=h_hat_base,
        rmst_err_full=rmst_err_full,
        rmst_err_base=rmst_err_base,
        plateau_err=plateau_err,
    )

    # Figure
    print("[4/4] Generating figure ...")
    make_figure(
        u_grid, a_arr,
        h_true, h_hat_full, h_hat_base,
        S_true, S_hat_full, S_hat_base,
        tau_true, tau_full, tau_base,
        out_dir / "toy_evaluation.png",
    )
    print("Done.")


if __name__ == "__main__":
    main()
