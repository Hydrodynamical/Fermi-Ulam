"""Calibration-specific plots.

All functions accept out_path=None (interactive show) or a file path (save PNG).
All figures use a log scale on the energy axis where appropriate.

Functions
---------
plot_poincare_section   -- (psi, u) scatter: KAM structure vs chaotic sea
plot_phase_acf          -- C(k) vs k for selected energy bins
plot_tau_mix            -- tau_lag(u) and m(u) vs u (log-log)
plot_km_coefficients    -- b(u) and a(u) side-by-side vs log(u)
plot_fp_comparison      -- f_FP(u) vs f_emp(u) overlay (log-log)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Poincaré section
# ---------------------------------------------------------------------------

def plot_poincare_section(
    u_traj: np.ndarray,
    psi_traj: np.ndarray,
    n_particles: int = 5,
    out_path=None,
):
    """Scatter plot of (psi_n, u_n) at all right-wall hits for n_particles.

    This is the standard Poincaré section of the Fermi-Ulam map.  Regular
    (KAM) tori appear as closed curves; the chaotic sea fills the remaining area.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    n = min(n_particles, u_traj.shape[0])
    for i in range(n):
        ax.scatter(
            psi_traj[i], u_traj[i],
            s=0.3, alpha=0.4, rasterized=True,
            label=f"particle {i}" if n <= 5 else None,
        )
    ax.set_xlabel(r"Phase $\psi$ (rad)")
    ax.set_ylabel(r"Energy $u = \frac{1}{2}s^2$")
    ax.set_yscale("log")
    ax.set_xlim(0.0, 2.0 * np.pi)
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_title("Poincaré section $(\\psi, u)$ at right-wall hits")
    if n <= 5:
        ax.legend(fontsize=7, markerscale=5)
    fig.tight_layout()
    _save_or_show(fig, out_path)


# ---------------------------------------------------------------------------
# Phase ACF
# ---------------------------------------------------------------------------

def plot_phase_acf(
    C: np.ndarray,
    u_centers: np.ndarray,
    bins_to_plot=None,
    out_path=None,
):
    """Plot C(k) vs k for selected energy bins.

    Parameters
    ----------
    C            : (n_bins, max_lag+1)  ACF from compute_phase_acf
    u_centers    : (n_bins,)
    bins_to_plot : list of bin indices; default: 5 evenly spaced
    """
    n_bins = C.shape[0]
    if bins_to_plot is None:
        bins_to_plot = np.linspace(0, n_bins - 1, 5, dtype=int)

    lags = np.arange(C.shape[1])
    fig, ax = plt.subplots(figsize=(8, 4))
    for b in bins_to_plot:
        ax.plot(lags, C[b], lw=1.2, label=f"$u={u_centers[b]:.2f}$")
    ax.axhline(0.0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("Lag $k$ (wall bounces)")
    ax.set_ylabel(r"$C(k) = \langle\cos(\psi_{n+k} - \psi_n)\rangle$")
    ax.set_title("Phase ACF by energy bin")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save_or_show(fig, out_path)


# ---------------------------------------------------------------------------
# Mixing time
# ---------------------------------------------------------------------------

def plot_tau_mix(
    u_centers: np.ndarray,
    tau_lag: np.ndarray,
    m_arr: np.ndarray = None,
    out_path=None,
):
    """Plot tau_lag(u) and optionally m(u) on log-log axes."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(u_centers, tau_lag, "o-", ms=4, lw=1.2,
            label=r"$\tau_{\rm lag}(u)$ [bounces]")
    if m_arr is not None:
        ax.plot(u_centers, m_arr, "s--", ms=4, lw=1.2,
                label=r"$m(u) = \lceil 3\,\tau_{\rm lag}\rceil$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Energy $u$")
    ax.set_ylabel("Lag (wall bounces)")
    ax.set_title("Phase mixing time from ACF")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save_or_show(fig, out_path)


# ---------------------------------------------------------------------------
# KM coefficients
# ---------------------------------------------------------------------------

def plot_km_coefficients(
    u_centers: np.ndarray,
    b: np.ndarray,
    a: np.ndarray,
    out_path=None,
):
    """Plot b(u) and a(u) side-by-side on log(u) axis."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(u_centers, b, "o-", ms=4, lw=1.2, color="tab:blue")
    ax.axhline(0.0, color="k", lw=0.5, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("Energy $u$")
    ax.set_ylabel(r"$\hat{b}(u)$  [per bounce]")
    ax.set_title("KM drift coefficient $b(u)$")

    ax = axes[1]
    # a may be negative in very low-sample bins; plot absolute value with markers.
    a_pos = np.where(a > 0, a, np.nan)
    ax.plot(u_centers, a_pos, "o-", ms=4, lw=1.2, color="tab:orange")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Energy $u$")
    ax.set_ylabel(r"$\hat{a}(u)$  [per bounce]")
    ax.set_title("KM diffusion coefficient $a(u)$")

    fig.tight_layout()
    _save_or_show(fig, out_path)


# ---------------------------------------------------------------------------
# FP comparison
# ---------------------------------------------------------------------------

def plot_fp_comparison(
    u_centers: np.ndarray,
    f_fp: np.ndarray,
    f_emp: np.ndarray,
    metrics: dict,
    out_path=None,
):
    """Overlay FP steady-state prediction against empirical energy histogram."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(u_centers, f_emp, "o", ms=3, alpha=0.7,
            color="tab:blue", label="Empirical $f(u)$")
    ax.plot(u_centers, f_fp, "-", lw=2.0,
            color="tab:red", label="FP steady-state $f_{\\rm FP}(u)$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Energy $u$")
    ax.set_ylabel("Probability density $f(u)$")
    l1 = metrics.get("l1_error", float("nan"))
    kl = metrics.get("kl_divergence", float("nan"))
    ax.set_title(f"FP validation:  $L^1 = {l1:.3f}$,  $D_{{\\rm KL}} = {kl:.4f}$")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save_or_show(fig, out_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_or_show(fig, out_path):
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()
