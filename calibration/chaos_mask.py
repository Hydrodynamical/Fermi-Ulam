"""KAM / chaotic-sea mask for the calibration experiment.

Answers: Are the bad-fitting bins contaminated by KAM islands?
Does the FP fit improve when restricted to genuinely chaotic bins?

Functions
---------
build_chaotic_mask     -- identify bins in the chaotic sea by entropy + τ_lag
estimate_km_masked     -- Kramers-Moyal coefficients on masked bins only
compute_fp_masked      -- FP steady-state and comparison on masked coefficients

Workflow
--------
1. build_chaotic_mask(entropy_norm, tau_lag) → bool mask (n_bins,)
2. estimate_km_masked(u_traj, u_edges, m_arr, mask) → b_masked, a_masked
3. compute_fp_masked(u_centers, b_masked, a_masked, f_emp) → f_fp_masked, metrics

If the masked FP fit is substantially better than the global fit, the failure
is localised to KAM-contaminated bins and the chaotic-sea operator is well
estimated.
"""
import sys
from pathlib import Path

import numpy as np

# Allow importing sibling modules when run as a script
sys.path.insert(0, str(Path(__file__).parent))
from fokker_planck import steady_state_fp, compare_fp_to_empirical


def build_chaotic_mask(
    entropy_norm: np.ndarray,
    tau_lag: np.ndarray,
    entropy_threshold: float = 0.8,
    tau_threshold: float = None,
) -> np.ndarray:
    """Boolean mask selecting bins that are in the chaotic sea.

    A bin is declared chaotic if:
      entropy_norm[b] >= entropy_threshold   (phase is nearly uniform)
      AND
      tau_lag[b] <= tau_threshold            (phase decorrelates quickly)

    Parameters
    ----------
    entropy_norm      : (n_bins,)  normalized phase entropy in [0,1]
    tau_lag           : (n_bins,)  ACF-based mixing lag (wall bounces)
    entropy_threshold : bins below this entropy are treated as non-chaotic
    tau_threshold     : bins above this lag are treated as non-chaotic;
                        defaults to the median of finite tau_lag values

    Returns
    -------
    mask : (n_bins,) bool  True = chaotic, False = non-chaotic / undetermined
    """
    n_bins = len(entropy_norm)

    if tau_threshold is None:
        finite_tau = tau_lag[np.isfinite(tau_lag)]
        if len(finite_tau) == 0:
            tau_threshold = np.inf
        else:
            tau_threshold = float(np.median(finite_tau))

    entropy_ok = np.where(np.isfinite(entropy_norm), entropy_norm >= entropy_threshold, False)
    tau_ok     = np.where(np.isfinite(tau_lag), tau_lag <= tau_threshold, False)

    return entropy_ok & tau_ok


def estimate_km_masked(
    u_traj: np.ndarray,
    u_edges: np.ndarray,
    m_arr: np.ndarray,
    mask: np.ndarray,
    n_min: int = 50,
) -> dict:
    """Kramers-Moyal drift b̂(u) and diffusion â(u) using only chaotic bins.

    Identical loop to kramers_moyal.estimate_km_coefficients but skips
    accumulation for bins where mask[b] = False.
    Chaotic bins get fresh estimates; non-chaotic bins return NaN.

    Returns
    -------
    dict with keys:
      b_masked      : (n_bins,)
      a_masked      : (n_bins,)
      counts_masked : (n_bins,) int64
    """
    N_p, N_h1 = u_traj.shape
    N_h = N_h1 - 1
    n_bins = len(u_edges) - 1

    sum1 = np.zeros(n_bins)
    sum2 = np.zeros(n_bins)
    cnt  = np.zeros(n_bins, dtype=np.int64)

    for i in range(N_p):
        u_i = u_traj[i]
        for b in range(n_bins):
            if not mask[b]:
                continue
            m = int(m_arr[b])
            if m > N_h:
                continue
            u_base = u_i[: N_h + 1 - m]
            in_bin = (u_base >= u_edges[b]) & (u_base < u_edges[b + 1])
            if not in_bin.any():
                continue
            base_idx = np.where(in_bin)[0]
            du = u_i[base_idx + m] - u_base[in_bin]
            sum1[b] += du.sum()
            sum2[b] += (du ** 2).sum()
            cnt[b]  += len(base_idx)

    b_masked = np.full(n_bins, np.nan)
    a_masked = np.full(n_bins, np.nan)
    for b in range(n_bins):
        if not mask[b] or cnt[b] < n_min:
            continue
        m = int(m_arr[b])
        n = cnt[b]
        mean_du = sum1[b] / n
        b_masked[b] = mean_du / m
        var_du = sum2[b] / n - mean_du ** 2
        a_masked[b] = var_du / m

    return {
        "b_masked":      b_masked,
        "a_masked":      a_masked,
        "counts_masked": cnt,
    }


def compute_fp_masked(
    u_centers: np.ndarray,
    b_masked: np.ndarray,
    a_masked: np.ndarray,
    f_emp: np.ndarray,
) -> dict:
    """FP steady-state and comparison metrics using masked KM coefficients.

    Returns
    -------
    dict with keys:
      f_fp_masked    : (n_bins,)  FP density from masked b/a
      metrics_masked : dict with l1_error, l2_error, kl_divergence
    """
    f_fp_masked = steady_state_fp(u_centers, b_masked, a_masked)
    metrics_masked = compare_fp_to_empirical(u_centers, f_fp_masked, f_emp)
    return {
        "f_fp_masked":    f_fp_masked,
        "metrics_masked": metrics_masked,
    }
