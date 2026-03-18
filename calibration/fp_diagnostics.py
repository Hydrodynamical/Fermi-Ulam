"""Extended Fokker-Planck diagnostics for the calibration experiment.

Answers: Where exactly does the FP model fail, and is it structural?

Functions
---------
compute_ks_distance          -- Kolmogorov-Smirnov statistic between CDFs
compute_log_density_residual -- r(u) = log f_emp(u) - log f_fp(u)
forward_validate_fp          -- evolve FP PDE forward and check convergence
bootstrap_km_confidence      -- bootstrap CI on b(u) and a(u)  [opt-in]

Interpretation
--------------
KS distance: less sensitive to binning than L1; reveals where CDFs diverge.
Log residual: shows whether FP under/over-estimates at low vs high energy.
Forward validation: even if L1 is large initially, if the FP attractor is the
  same as the empirical steady state then the *operator* is correct.
Bootstrap CI: if KM coefficients have wide CI, mismatch may be statistical
  rather than structural.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from kramers_moyal import estimate_km_coefficients
from fokker_planck import steady_state_fp, compare_fp_to_empirical


# ---------------------------------------------------------------------------
# KS distance
# ---------------------------------------------------------------------------

def compute_ks_distance(
    u_centers: np.ndarray,
    f_fp: np.ndarray,
    f_emp: np.ndarray,
    u_edges: np.ndarray,
) -> dict:
    """Kolmogorov-Smirnov distance between empirical and FP CDFs.

    Both CDFs are built by cumulative trapezoidal integration of the density.

    Returns
    -------
    dict with keys:
      ks_stat  : float  max |CDF_emp - CDF_fp|
      ks_u     : float  energy at which the maximum occurs
      cdf_fp   : (n_bins,)
      cdf_emp  : (n_bins,)
    """
    du = np.diff(u_edges)  # (n_bins,)

    # Handle NaN by zero-filling for CDF purposes
    f_fp_clean  = np.where(np.isfinite(f_fp), f_fp, 0.0)
    f_emp_clean = np.where(np.isfinite(f_emp), f_emp, 0.0)

    cdf_fp  = np.cumsum(f_fp_clean * du)
    cdf_emp = np.cumsum(f_emp_clean * du)

    # Normalise so each CDF ends at 1
    if cdf_fp[-1] > 0:
        cdf_fp  /= cdf_fp[-1]
    if cdf_emp[-1] > 0:
        cdf_emp /= cdf_emp[-1]

    diff = np.abs(cdf_emp - cdf_fp)
    ks_stat = float(diff.max())
    ks_idx = int(diff.argmax())
    ks_u = float(u_centers[ks_idx])

    return {
        "ks_stat": ks_stat,
        "ks_u":    ks_u,
        "cdf_fp":  cdf_fp,
        "cdf_emp": cdf_emp,
    }


# ---------------------------------------------------------------------------
# Log-density residual
# ---------------------------------------------------------------------------

def compute_log_density_residual(
    u_centers: np.ndarray,
    f_fp: np.ndarray,
    f_emp: np.ndarray,
) -> dict:
    """Log-density residual r(u) = log f_emp(u) - log f_fp(u).

    Positive r → FP underestimates density at u.
    Negative r → FP overestimates density at u.
    A flat r (r ≈ const) means the shape is right but the normalization differs.

    Returns
    -------
    dict with keys:
      residual : (n_bins,)  NaN where either density is non-positive
    """
    valid = (f_fp > 0) & (f_emp > 0) & np.isfinite(f_fp) & np.isfinite(f_emp)
    residual = np.full(len(u_centers), np.nan)
    residual[valid] = np.log(f_emp[valid]) - np.log(f_fp[valid])
    return {"residual": residual}


# ---------------------------------------------------------------------------
# Forward FP validation
# ---------------------------------------------------------------------------

def forward_validate_fp(
    u_centers: np.ndarray,
    u_edges: np.ndarray,
    b: np.ndarray,
    a: np.ndarray,
    f_emp: np.ndarray,
    n_steps: int = 10,
) -> dict:
    """Evolve the FP PDE forward from f_emp and measure convergence to steady state.

    PDE: df/dt = -d/du[b(u) f] + (1/2) d²/du²[a(u) f]

    Discretisation: central differences in u, forward Euler in t.
    Time step dt is chosen conservatively from the CFL condition:
        dt < 0.5 * du_min² / max(a_safe)

    The steady-state f_fp (from b and a) is the analytic attractor of this PDE.
    Starting from f_emp, the L1 distance to f_fp should decrease toward zero if
    the operator is correct.

    If L1 INCREASES or does not change, the PDE attractor differs from f_fp →
    the operator may have a sign error or the boundary conditions are wrong.

    Returns
    -------
    dict with keys:
      f_evolved       : (n_steps, n_bins)  density at each step
      step_l1_to_fp   : (n_steps,)         L1(f_evolved[k], f_fp) vs step index
      step_l1_to_emp  : (n_steps,)         L1(f_evolved[k], f_emp) vs step index
    """
    n_bins = len(u_centers)
    du = np.diff(u_edges)  # (n_bins,)

    # Fill NaN in b, a by linear interpolation on log(u)
    log_u = np.log(u_centers)
    b_filled = _fill_nan_interp(log_u, b)
    a_filled = _fill_nan_interp(log_u, a)
    a_safe = np.maximum(np.abs(a_filled), 1e-12 * (np.abs(a_filled).max() + 1e-300))

    # CFL: dt ≤ 0.5 * min(du)² / max(a) and dt ≤ 0.5 * min(du) / max(|b|)
    du_min = du.min()
    dt_diff = 0.5 * du_min ** 2 / a_safe.max()
    dt_adv  = 0.5 * du_min / (np.abs(b_filled).max() + 1e-300)
    dt = float(min(dt_diff, dt_adv))

    # Analytic steady state (target)
    from fokker_planck import steady_state_fp
    f_ss = steady_state_fp(u_centers, b, a)
    f_ss_clean = np.where(np.isfinite(f_ss), f_ss, 0.0)

    # Initialise from empirical density
    f = np.where(np.isfinite(f_emp), f_emp, 0.0)
    # Renormalize
    norm = np.sum(f * du)
    if norm > 0:
        f /= norm

    f_evolved = np.empty((n_steps, n_bins))
    l1_to_fp  = np.empty(n_steps)
    l1_to_emp = np.empty(n_steps)

    for step in range(n_steps):
        f = _fp_step(f, b_filled, a_safe, du, dt)
        f_evolved[step] = f
        l1_to_fp[step]  = float(np.sum(np.abs(f - f_ss_clean) * du))
        l1_to_emp[step] = float(np.sum(np.abs(f - np.where(np.isfinite(f_emp), f_emp, 0.0)) * du))

    return {
        "f_evolved":      f_evolved,
        "step_l1_to_fp":  l1_to_fp,
        "step_l1_to_emp": l1_to_emp,
        "dt":             dt,
        "n_steps":        n_steps,
    }


def _fp_step(
    f: np.ndarray,
    b: np.ndarray,
    a: np.ndarray,
    du: np.ndarray,
    dt: float,
) -> np.ndarray:
    """One forward-Euler step of the 1D FP equation on a non-uniform grid.

    Uses central differences for the diffusion flux and upwind for drift.
    Reflecting boundaries: flux = 0 at both ends.
    """
    n = len(f)

    # Diffusion flux at interior cell boundaries: J_diff[j] between cell j and j+1
    # J_diff = -d/du[a*f] / 2, approximated as -(a[j+1]*f[j+1] - a[j]*f[j]) / du_j
    af = a * f
    # du[j] is the width of cell j; boundaries are at u_edges[j]
    # Grid spacing between centres:
    du_c = 0.5 * (du[:-1] + du[1:])  # (n-1,) distance between adjacent centres

    J_diff = -0.5 * (af[1:] - af[:-1]) / du_c   # (n-1,) at interior boundaries

    # Drift flux at interior cell boundaries (upwind): J_drift = b*f
    b_boundary = 0.5 * (b[:-1] + b[1:])  # (n-1,) interpolated b at boundaries
    J_drift = np.where(b_boundary >= 0, b_boundary * f[:-1], b_boundary * f[1:])

    J = J_drift + J_diff  # (n-1,) total flux

    # Reflecting BCs: J_left = J_right = 0 (no flux through endpoints)
    # df/dt = -(J[j] - J[j-1]) / du[j]  with J[0] = J[n-1] = 0 (boundary fluxes)
    J_full = np.concatenate([[0.0], J, [0.0]])  # (n+1,)
    dfdt = -(J_full[1:] - J_full[:-1]) / du    # (n,)

    f_new = f + dt * dfdt

    # Clip negatives (numerical noise) and renormalize
    f_new = np.maximum(f_new, 0.0)
    norm = np.sum(f_new * du)
    if norm > 0:
        f_new /= norm
    return f_new


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_km_confidence(
    u_traj: np.ndarray,
    u_edges: np.ndarray,
    m_arr: np.ndarray,
    n_bootstrap: int = 50,
    n_min: int = 50,
    seed: int = 0,
) -> dict:
    """Bootstrap confidence intervals on b(u) and a(u) by resampling particles.

    Resamples N_particles trajectories with replacement n_bootstrap times,
    re-running estimate_km_coefficients each time.

    Returns
    -------
    dict with keys:
      b_ci : (n_bins, 3)  [p5, p50, p95] across bootstrap replicates
      a_ci : (n_bins, 3)
    """
    N_p = u_traj.shape[0]
    n_bins = len(u_edges) - 1
    rng = np.random.default_rng(seed)

    b_boot = np.full((n_bootstrap, n_bins), np.nan)
    a_boot = np.full((n_bootstrap, n_bins), np.nan)

    for k in range(n_bootstrap):
        idx = rng.integers(0, N_p, size=N_p)
        u_resample = u_traj[idx]
        km = estimate_km_coefficients(u_resample, u_edges, m_arr, n_min=n_min)
        b_boot[k] = km["b"]
        a_boot[k] = km["a"]

    b_ci = np.nanpercentile(b_boot, [5, 50, 95], axis=0).T  # (n_bins, 3)
    a_ci = np.nanpercentile(a_boot, [5, 50, 95], axis=0).T

    return {
        "b_ci": b_ci,
        "a_ci": a_ci,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fill_nan_interp(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    valid = np.isfinite(y)
    if valid.sum() < 2:
        return np.where(valid, y, 0.0)
    return np.interp(x, x[valid], y[valid])
