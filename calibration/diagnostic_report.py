"""8-panel diagnostic dashboard for the calibration experiment.

Loads all diagnostic .npz files from a results directory and produces
a single diagnostic_report.png summarising the four key questions:

    Panel 1: Phase entropy H(u)             — Is the map chaotic?
    Panel 2: Integrated ACF time τ_int(u)   — How fast does phase mix?
    Panel 3: Small-jump ratio R(u, η=0.1)   — Is diffusion in u valid?
    Panel 4: Skewness + kurtosis of Δu      — Is Δu Gaussian-ish?
    Panel 5: Skewness + kurtosis of Δlog u  — Is log(u) better?
    Panel 6: Phase-conditioned drift (u, ψ) — Hidden Markov failure?
    Panel 7: Var(Δu) vs lag for 3 bins      — Does variance scale linearly?
    Panel 8: Log-density residual r(u)      — Where does FP fail?

Usage
-----
    from diagnostic_report import make_diagnostic_report
    make_diagnostic_report(diag_dir, u_centers, u_edges, out_path)

Or stand-alone:
    python3 calibration/diagnostic_report.py results/calibration/diagnostics
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_diagnostic_report(
    diag_dir,
    u_centers: np.ndarray,
    u_edges: np.ndarray,
    out_path=None,
) -> None:
    """Produce the 8-panel dashboard from saved .npz files.

    Parameters
    ----------
    diag_dir  : path to directory containing the diagnostic .npz files
    u_centers : (n_bins,) energy grid centres
    u_edges   : (n_bins+1,) energy bin edges
    out_path  : output PNG path; defaults to diag_dir/diagnostic_report.png
    """
    diag_dir = Path(diag_dir)
    if out_path is None:
        out_path = diag_dir / "diagnostic_report.png"

    # Load available .npz files (gracefully skip missing ones)
    mix   = _load_npz(diag_dir / "mixing_diagnostics.npz")
    incr  = _load_npz(diag_dir / "increment_diagnostics.npz")
    mark  = _load_npz(diag_dir / "markov_tests.npz")
    chaos = _load_npz(diag_dir / "chaos_mask.npz")
    fpd   = _load_npz(diag_dir / "fp_diagnostics.npz")

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("Calibration diagnostic report", fontsize=12, y=1.01)

    _panel_entropy(axes[0, 0], u_centers, mix, chaos)
    _panel_tau_int(axes[0, 1], u_centers, mix)
    _panel_small_jump(axes[0, 2], u_centers, incr)
    _panel_skew_kurt_u(axes[0, 3], u_centers, incr)
    _panel_skew_kurt_logu(axes[1, 0], u_centers, incr)
    _panel_drift_heatmap(axes[1, 1], u_centers, mark)
    _panel_var_vs_lag(axes[1, 2], u_centers, incr)
    _panel_log_residual(axes[1, 3], u_centers, fpd)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[diagnostic_report] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Individual panel functions
# ---------------------------------------------------------------------------

def _panel_entropy(ax, u_centers, mix, chaos):
    ax.set_title("Phase entropy H(u)")
    if mix is None or "entropy_norm" not in mix:
        _no_data(ax); return
    h = mix["entropy_norm"]
    ax.plot(u_centers, h, "o-", ms=3, lw=1.2, color="tab:purple")
    if chaos is not None and "entropy_threshold" in chaos:
        ax.axhline(float(chaos["entropy_threshold"]), color="r", ls="--", lw=0.8,
                   label=f"threshold={float(chaos['entropy_threshold']):.2f}")
        ax.legend(fontsize=7)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("u")
    ax.set_ylabel("H_norm(u)")


def _panel_tau_int(ax, u_centers, mix):
    ax.set_title("Integrated ACF time τ_int(u)")
    if mix is None or "tau_int" not in mix:
        _no_data(ax); return
    ti = mix["tau_int"]
    ax.plot(u_centers, ti, "o-", ms=3, lw=1.2, color="tab:green")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("u")
    ax.set_ylabel("τ_int (bounces)")


def _panel_small_jump(ax, u_centers, incr):
    ax.set_title("Small-jump ratio R(u, η)")
    if incr is None or "ratios" not in incr:
        _no_data(ax); return
    ratios = incr["ratios"]       # (n_bins, n_eta)
    etas   = incr["eta_values"]
    for j, eta in enumerate(etas):
        ax.plot(u_centers, ratios[:, j], lw=1.2, label=f"η={eta}")
    ax.axhline(0.5, color="k", ls="--", lw=0.5)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("u")
    ax.set_ylabel("P(|Δu| ≤ η·u)")
    ax.legend(fontsize=7)


def _panel_skew_kurt_u(ax, u_centers, incr):
    ax.set_title("Δu moments")
    if incr is None or "skew_u" not in incr:
        _no_data(ax); return
    ax2 = ax.twinx()
    ax.plot(u_centers, incr["skew_u"],  "o-", ms=2, lw=1.0, color="tab:blue",  label="skew")
    ax2.plot(u_centers, incr["kurt_u"], "s--", ms=2, lw=1.0, color="tab:orange", label="xkurt")
    ax.axhline(0, color="k", lw=0.4, ls=":")
    ax2.axhline(0, color="gray", lw=0.4, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("u")
    ax.set_ylabel("skewness", color="tab:blue")
    ax2.set_ylabel("excess kurtosis", color="tab:orange")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)


def _panel_skew_kurt_logu(ax, u_centers, incr):
    ax.set_title("Δlog(u) moments")
    if incr is None or "skew_logu" not in incr:
        _no_data(ax); return
    ax2 = ax.twinx()
    ax.plot(u_centers, incr["skew_logu"],  "o-", ms=2, lw=1.0, color="tab:blue",  label="skew")
    ax2.plot(u_centers, incr["kurt_logu"], "s--", ms=2, lw=1.0, color="tab:orange", label="xkurt")
    ax.axhline(0, color="k", lw=0.4, ls=":")
    ax2.axhline(0, color="gray", lw=0.4, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("u")
    ax.set_ylabel("skewness", color="tab:blue")
    ax2.set_ylabel("excess kurtosis", color="tab:orange")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)


def _panel_drift_heatmap(ax, u_centers, mark):
    ax.set_title("Phase-conditioned drift E[Δu | u, ψ]")
    if mark is None or "drift_map" not in mark:
        _no_data(ax); return
    dm = mark["drift_map"]  # (n_u, n_psi)
    psi_edges = mark["psi_edges"]
    psi_centers = 0.5 * (psi_edges[:-1] + psi_edges[1:])
    im = ax.pcolormesh(
        psi_centers, u_centers, dm,
        cmap="RdBu_r", shading="auto",
        vmin=-np.nanstd(dm) * 2, vmax=np.nanstd(dm) * 2,
    )
    ax.set_yscale("log")
    ax.set_xlabel("ψ (rad)")
    ax.set_ylabel("u")
    ax.set_xticks([0, np.pi, 2 * np.pi])
    ax.set_xticklabels(["0", "π", "2π"])
    plt.colorbar(im, ax=ax, label="E[Δu]", fraction=0.046, pad=0.04)


def _panel_var_vs_lag(ax, u_centers, incr):
    ax.set_title("Var(Δu) vs lag  (3 bins)")
    if incr is None or "var_u" not in incr:
        _no_data(ax); return
    var_u  = incr["var_u"]   # (n_bins, n_lags)
    lags   = incr["lag_values"]
    n_bins = var_u.shape[0]
    # Pick 3 representative bins: low, mid, high energy
    bins = [n_bins // 8, n_bins // 2, 7 * n_bins // 8]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for b, col in zip(bins, colors):
        row = var_u[b]
        valid = np.isfinite(row)
        if not valid.any():
            continue
        ax.loglog(lags[valid], row[valid], "o-", ms=3, lw=1.2, color=col,
                  label=f"u={u_centers[b]:.2f}")
    # Reference line: slope 1 (linear scaling)
    lref = np.array([lags[0], lags[-1]], dtype=float)
    # Scale to middle of data range
    all_vals = var_u[np.isfinite(var_u)]
    if len(all_vals) > 0:
        ref_scale = np.nanmedian(var_u[:, 0][np.isfinite(var_u[:, 0])])
        ax.loglog(lref, ref_scale * lref / lref[0], "k--", lw=0.8, label="slope 1")
    ax.set_xlabel("lag m (bounces)")
    ax.set_ylabel("Var(Δu)")
    ax.legend(fontsize=7)


def _panel_log_residual(ax, u_centers, fpd):
    ax.set_title("FP log-density residual r(u)")
    if fpd is None or "residual" not in fpd:
        _no_data(ax); return
    r = fpd["residual"]
    ax.plot(u_centers, r, "o-", ms=3, lw=1.2, color="tab:red")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.fill_between(u_centers, r, 0,
                    where=np.isfinite(r), alpha=0.15, color="tab:red")
    ax.set_xscale("log")
    ax.set_xlabel("u")
    ax.set_ylabel("log f_emp − log f_fp")


def _no_data(ax):
    ax.text(0.5, 0.5, "no data", ha="center", va="center",
            transform=ax.transAxes, color="gray")


def _load_npz(path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        d = np.load(path)
        return {k: d[k] for k in d.files}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) < 2:
        print("Usage: python diagnostic_report.py <diag_dir> [out_path]")
        _sys.exit(1)
    _diag_dir = Path(_sys.argv[1])
    _out = _sys.argv[2] if len(_sys.argv) > 2 else None

    # Try to load u_centers from parent calibration_results.npz
    _cal_npz = _diag_dir.parent / "calibration_results.npz"
    if not _cal_npz.exists():
        print(f"calibration_results.npz not found in {_diag_dir.parent}")
        _sys.exit(1)
    _cal = np.load(_cal_npz)
    make_diagnostic_report(_diag_dir, _cal["u_centers"], _cal["u_edges"], out_path=_out)
