"""Estimate the phase mixing time tau_lag(u) from trajectory data.

The key observable is the phase autocorrelation function (ACF):

    C(k; u_bin) = E[ cos(psi_{n+k} - psi_n)  |  u_n in u_bin ]

where the expectation is over all particles i and all base hits n such that
u[i, n] falls in u_bin.  C(k) starts at 1 (k=0) and decays toward 0 as the
map loses memory of the initial phase.  The mixing lag tau_lag(u) is the
smallest k where |C(k)| < threshold.

Why cos(Delta psi) rather than Delta psi?
  psi is stored as (angle) mod 2*pi, so differences can wrap.  Cosine is
  2*pi-periodic, so cos(psi_lag - psi_base) is always correct regardless
  of wrapping — no unwrapping step needed.

Vectorisation strategy
----------------------
Outer loop: N_particles (~500 iterations, Python)
Inner loop: lag k = 0..max_lag (~200 iterations, Python)
  Body: vectorised NumPy over all base hits (~N_hits elements)
        np.add.at(accumulator, bin_indices, cos_values)  [scatter-add]

Total Python iterations: N_particles * (max_lag+1) ~ 100 000.
Each iteration does O(N_hits) floating-point work in NumPy.
Typical wall time: 2–10 minutes for the default config.

Note on "same bin" assumption
------------------------------
We condition on u_n being in a bin but do NOT require u_{n+k} to remain
there.  This estimates the unconditional ACF starting from u_n, which is
the right object for measuring how quickly the map loses phase memory at
that energy.  Particles that drift to another energy bin simply contribute
to the ACF at their base-hit bin.
"""
import numpy as np


def compute_phase_acf(
    u_traj: np.ndarray,
    psi_traj: np.ndarray,
    u_edges: np.ndarray,
    max_lag: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute C(k; u_bin) for k = 0 .. max_lag.

    Parameters
    ----------
    u_traj   : (N_particles, N_hits+1)
    psi_traj : (N_particles, N_hits+1)
    u_edges  : (n_bins+1,)  log-spaced bin edges
    max_lag  : maximum lag k to compute

    Returns
    -------
    C      : (n_bins, max_lag+1)  normalised ACF; C[:, 0] == 1 by construction
             NaN for bins with zero base hits.
    counts : (n_bins,)  number of (particle, base-hit) pairs per bin
    """
    N_p, N_h1 = u_traj.shape
    N_h = N_h1 - 1
    n_bins = len(u_edges) - 1

    # Accumulators indexed [bin, lag].
    C_sum = np.zeros((n_bins, max_lag + 1), dtype=np.float64)
    C_cnt = np.zeros((n_bins, max_lag + 1), dtype=np.int64)

    N_base = N_h + 1 - max_lag  # number of valid base-hit positions per particle
    if N_base <= 0:
        raise ValueError(
            f"max_lag={max_lag} >= N_hits={N_h}. "
            "Reduce acf_max_lag or increase N_hits."
        )

    for i in range(N_p):
        u_i = u_traj[i]       # (N_h+1,)
        psi_i = psi_traj[i]   # (N_h+1,)

        # Bin index for each of the first N_base hits (valid base positions).
        base_bins = np.digitize(u_i[:N_base], u_edges) - 1
        base_bins = np.clip(base_bins, 0, n_bins - 1)  # (N_base,)

        psi_base = psi_i[:N_base]  # (N_base,)

        for k in range(max_lag + 1):
            psi_lag = psi_i[k : k + N_base]              # (N_base,)
            contrib = np.cos(psi_lag - psi_base)          # (N_base,)
            np.add.at(C_sum[:, k], base_bins, contrib)
            np.add.at(C_cnt[:, k], base_bins, 1)

    # Normalise each (bin, lag) cell.
    valid = C_cnt > 0
    C = np.where(valid, C_sum / np.maximum(C_cnt, 1), np.nan)

    counts = C_cnt[:, 0]  # number of base-hit pairs per bin (lag-0 count)
    return C, counts


def estimate_tau_mix_from_acf(
    C: np.ndarray,
    counts: np.ndarray,
    threshold: float = 0.05,
    n_min: int = 200,
) -> np.ndarray:
    """Find tau_lag(u) = smallest lag k where |C[u_bin, k]| < threshold.

    Parameters
    ----------
    C        : (n_bins, max_lag+1)  ACF from compute_phase_acf
    counts   : (n_bins,)            base-hit counts
    threshold: decorrelation criterion
    n_min    : bins with fewer than n_min base hits get NaN

    Returns
    -------
    tau_lag : (n_bins,)  mixing lag in wall-bounce units; NaN if undetermined
    """
    n_bins = C.shape[0]
    tau_lag = np.full(n_bins, np.nan)

    for b in range(n_bins):
        if counts[b] < n_min:
            continue
        acf = C[b]
        below = np.where(np.abs(acf) < threshold)[0]
        if len(below) == 0:
            # Did not decorrelate within max_lag; assign the maximum observed lag.
            tau_lag[b] = float(C.shape[1] - 1)
        else:
            tau_lag[b] = float(below[0])

    return tau_lag


def choose_coarse_lag(
    tau_lag: np.ndarray,
    safety: float = 3.0,
    m_min: int = 5,
) -> np.ndarray:
    """Compute the coarse-graining lag m(u) from tau_lag(u).

    m(u) = max(m_min, ceil(safety * tau_lag(u)))

    NaN bins (insufficient data or undetermined) fall back to m_min.

    Returns
    -------
    m_arr : (n_bins,)  int64 array of coarse lags
    """
    finite = np.isfinite(tau_lag)
    m = np.where(
        finite,
        np.maximum(m_min, np.ceil(safety * tau_lag)).astype(int),
        m_min,
    )
    return m.astype(np.int64)
