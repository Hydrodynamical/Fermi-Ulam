"""Fokker-Planck steady-state solver and empirical comparison.

The 1-D Fokker-Planck equation for the energy marginal f(u, t):

    df/dt = -d/du [ b(u) f ]  +  (1/2) d^2/du^2 [ a(u) f ]

Steady-state with reflecting boundary conditions means zero probability flux:

    J(u) = b(u) f(u)  -  (1/2) d/du [ a(u) f(u) ]  =  0

This is a first-order linear ODE in the product h(u) = a(u) f(u):

    dh/du  =  (2 b(u) / a(u))  h(u)

Solution by integrating factor:

    h(u) = h(0) * exp( 2 * integral_0^u  b(u') / a(u')  du' )

    f_FP(u) = C / a(u) * exp( 2 * integral_0^u  b(u') / a(u')  du' )

where C is the normalisation constant (integral f du = 1).

This formula is exact for any smooth b and a with a(u) > 0.  It requires no
PDE solver — just one numerical integration of the ratio b/a.

Note on per-bounce vs physical-time coefficients
-------------------------------------------------
The ratio b(u)/a(u) is the same whether b and a are in per-bounce units or
physical-time units, because both are divided by the same mean round-trip
time T_bar(u).  We can therefore use the per-bounce outputs of
kramers_moyal.estimate_km_coefficients directly in this formula.
"""
import numpy as np


def steady_state_fp(
    u_centers: np.ndarray,
    b: np.ndarray,
    a: np.ndarray,
) -> np.ndarray:
    """Compute the FP steady-state density on the u_centers grid.

    Parameters
    ----------
    u_centers : (n_bins,)  energy grid, sorted ascending (log-spaced centres)
    b         : (n_bins,)  drift;      NaN entries are linearly interpolated
    a         : (n_bins,)  diffusion;  NaN entries are linearly interpolated

    Returns
    -------
    f_fp : (n_bins,)  density, normalised so integral f_fp du = 1
           NaN if normalisation fails (e.g. all-NaN input).
    """
    log_u = np.log(u_centers)

    b_filled = _fill_nan(log_u, b)
    a_filled = _fill_nan(log_u, a)

    # Guard against non-positive diffusion (numerical artefact near boundaries).
    a_pos = np.abs(a_filled)
    a_safe = np.maximum(a_pos, 1e-12 * np.nanmax(a_pos + 1e-300))

    ratio = b_filled / a_safe  # b/a  (n_bins,)

    # Cumulative trapezoidal integral of b/a from u_centers[0] to u_centers[k].
    # Pure NumPy: trapezoid areas, then cumulative sum.
    du = np.diff(u_centers)                          # (n_bins-1,)
    trap = 0.5 * (ratio[:-1] + ratio[1:]) * du      # (n_bins-1,)
    integral = np.concatenate([[0.0], np.cumsum(trap)])  # (n_bins,)

    f_unnorm = np.exp(2.0 * integral) / a_safe
    f_unnorm = np.maximum(f_unnorm, 0.0)

    norm = np.trapezoid(f_unnorm, u_centers)
    if not np.isfinite(norm) or norm <= 0.0:
        return np.full_like(u_centers, np.nan)

    return f_unnorm / norm


def empirical_energy_density(
    u_traj: np.ndarray,
    u_edges: np.ndarray,
) -> np.ndarray:
    """Histogram of energies from all (particle, hit) pairs, normalised to
    integrate to 1 over the energy variable u.

    Parameters
    ----------
    u_traj  : (N_particles, N_hits+1)
    u_edges : (n_bins+1,)

    Returns
    -------
    f_emp : (n_bins,)  density (counts / (total_count * du_bin))
    """
    u_flat = u_traj.ravel()
    counts, _ = np.histogram(u_flat, bins=u_edges)
    du = np.diff(u_edges)
    norm = (counts.astype(float) * du).sum()
    if norm <= 0.0:
        return np.zeros(len(u_edges) - 1)
    return counts / norm


def compare_fp_to_empirical(
    u_centers: np.ndarray,
    f_fp: np.ndarray,
    f_emp: np.ndarray,
) -> dict:
    """Compute L1, L2, and KL divergence between f_FP and f_emp.

    Only bins where both densities are finite and f_emp > 0 contribute.

    Returns
    -------
    dict with keys 'l1_error', 'l2_error', 'kl_divergence'
    """
    valid = np.isfinite(f_fp) & np.isfinite(f_emp) & (f_emp > 0) & (f_fp > 0)
    if valid.sum() == 0:
        return {
            "l1_error": float("nan"),
            "l2_error": float("nan"),
            "kl_divergence": float("nan"),
        }

    f1 = f_fp[valid]
    f2 = f_emp[valid]
    # Bin widths at valid centres (approximate as differences; use midpoints).
    edges_approx = np.concatenate([
        [u_centers[valid][0]],
        0.5 * (u_centers[valid][:-1] + u_centers[valid][1:]),
        [u_centers[valid][-1]],
    ])
    du = np.diff(edges_approx)

    l1 = float(np.sum(np.abs(f1 - f2) * du))
    l2 = float(np.sqrt(np.sum((f1 - f2) ** 2 * du)))
    kl = float(np.sum(f2 * np.log(f2 / f1) * du))  # KL( f_emp || f_fp )

    return {"l1_error": l1, "l2_error": l2, "kl_divergence": kl}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fill_nan(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fill NaN values in y(x) by linear interpolation on x."""
    valid = np.isfinite(y)
    if valid.sum() < 2:
        filled = np.where(valid, y, 0.0)
        return filled
    return np.interp(x, x[valid], y[valid])
