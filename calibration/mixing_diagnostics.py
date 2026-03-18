"""Mixing diagnostic metrics for the calibration experiment.

Answers: Is the phase actually mixing? Where does it fail?

Functions
---------
compute_integrated_acf_time  -- τ_int(u) from existing ACF matrix
compute_phase_uniformity     -- TV distance and circular modes at coarse lag m(u)
compute_phase_entropy        -- normalized Shannon entropy of marginal phase histogram

All functions consume the same trajectory arrays (u_traj, psi_traj) already
produced by generate_trajectories(), requiring no additional simulation.
"""
import numpy as np

TWO_PI = 2.0 * np.pi


def compute_integrated_acf_time(C: np.ndarray) -> np.ndarray:
    """Integrated autocorrelation time τ_int(u) from the phase ACF matrix.

    Standard IACT formula using the positive part only to avoid noise
    amplification from the tail:

        τ_int[b] = 1 + 2 * sum_{k=1}^{max_lag} max(C[b, k], 0)

    Parameters
    ----------
    C : (n_bins, max_lag+1) float64, from compute_phase_acf (C[:, 0] == 1)

    Returns
    -------
    tau_int : (n_bins,)  NaN where C[b] is all-NaN
    """
    n_bins = C.shape[0]
    tau_int = np.full(n_bins, np.nan)
    for b in range(n_bins):
        row = C[b]
        if not np.isfinite(row[0]):
            continue
        positive_tail = np.maximum(row[1:], 0.0)
        tau_int[b] = 1.0 + 2.0 * positive_tail.sum()
    return tau_int


def compute_phase_uniformity(
    u_traj: np.ndarray,
    psi_traj: np.ndarray,
    u_edges: np.ndarray,
    m_arr: np.ndarray,
    n_psi_bins: int = 36,
) -> dict:
    """Phase distribution at coarse lag m(u): TV distance from uniform + circular modes.

    For each energy bin b at lag m = m_arr[b], collects all phases
    psi_traj[i, n+m] for base hits n where u_traj[i, n] is in bin b.

    Metrics
    -------
    TV distance: 0.5 * sum_j |p_j - 1/n_psi_bins|   in [0, 1]
    Circular modes: |E[exp(i*l*psi)]| for l = 1, 2, 3, 4  (0 if uniform)

    Parameters
    ----------
    u_traj   : (N_particles, N_hits+1)
    psi_traj : (N_particles, N_hits+1)
    u_edges  : (n_bins+1,)
    m_arr    : (n_bins,)  int64  coarse lag per bin
    n_psi_bins : number of phase histogram bins

    Returns
    -------
    dict with keys:
      tv_distance    : (n_bins,)     NaN for bins with zero samples
      circular_modes : (n_bins, 4)   |E[exp(i*l*psi)]| for l=1,2,3,4
      counts         : (n_bins,)     int64 sample count per bin
    """
    N_p, N_h1 = u_traj.shape
    N_h = N_h1 - 1
    n_bins = len(u_edges) - 1
    m_max = int(m_arr.max())

    psi_bin_edges = np.linspace(0.0, TWO_PI, n_psi_bins + 1)
    uniform_p = 1.0 / n_psi_bins

    # Accumulators for circular modes: real and imaginary parts, 4 harmonics
    mode_re = np.zeros((n_bins, 4))
    mode_im = np.zeros((n_bins, 4))
    cnt = np.zeros(n_bins, dtype=np.int64)

    # Phase histograms (used only at the end for TV; stored as running sum)
    phase_hist = np.zeros((n_bins, n_psi_bins), dtype=np.float64)

    for i in range(N_p):
        u_i = u_traj[i]
        psi_i = psi_traj[i]

        for b in range(n_bins):
            m = int(m_arr[b])
            if m > N_h:
                continue
            u_base = u_i[: N_h + 1 - m]
            in_bin = (u_base >= u_edges[b]) & (u_base < u_edges[b + 1])
            if not in_bin.any():
                continue
            base_idx = np.where(in_bin)[0]
            psi_lag = psi_i[base_idx + m]  # phases at lag m

            # Phase histogram contribution
            h, _ = np.histogram(psi_lag, bins=psi_bin_edges)
            phase_hist[b] += h

            # Circular modes
            for j, ell in enumerate([1, 2, 3, 4]):
                mode_re[b, j] += np.cos(ell * psi_lag).sum()
                mode_im[b, j] += np.sin(ell * psi_lag).sum()

            cnt[b] += len(base_idx)

    # TV distance and normalised circular modes
    tv = np.full(n_bins, np.nan)
    circ = np.full((n_bins, 4), np.nan)
    for b in range(n_bins):
        if cnt[b] == 0:
            continue
        n = cnt[b]
        p = phase_hist[b] / n  # empirical probabilities
        tv[b] = 0.5 * np.abs(p - uniform_p).sum()
        for j in range(4):
            circ[b, j] = np.sqrt(mode_re[b, j] ** 2 + mode_im[b, j] ** 2) / n

    return {
        "tv_distance": tv,
        "circular_modes": circ,
        "counts": cnt,
    }


def compute_phase_entropy(
    u_traj: np.ndarray,
    psi_traj: np.ndarray,
    u_edges: np.ndarray,
    n_psi_bins: int = 36,
    n_min: int = 50,
) -> dict:
    """Normalized Shannon entropy of the marginal phase distribution per energy bin.

    Uses all hits (not conditioned on lag) to build the marginal ψ distribution
    within each energy bin.

        H(u) = -sum_j p_j * log(p_j)
        H_norm(u) = H(u) / log(n_psi_bins)    [in [0, 1]]

    H_norm ≈ 1 → phase is approximately uniform (chaotic sea)
    H_norm << 1 → phase is concentrated (KAM tori / sticky orbits)

    Returns
    -------
    dict with keys:
      entropy      : (n_bins,)  raw Shannon entropy in nats
      entropy_norm : (n_bins,)  normalized to [0,1]
      counts       : (n_bins,)  total hit count per bin
    """
    N_p, N_h1 = u_traj.shape
    n_bins = len(u_edges) - 1
    psi_bin_edges = np.linspace(0.0, TWO_PI, n_psi_bins + 1)
    max_H = np.log(n_psi_bins)

    phase_hist = np.zeros((n_bins, n_psi_bins), dtype=np.float64)
    cnt = np.zeros(n_bins, dtype=np.int64)

    # Vectorise over hits for each particle
    for i in range(N_p):
        u_i = u_traj[i]           # (N_h+1,)
        psi_i = psi_traj[i]       # (N_h+1,)
        bin_idx = np.clip(np.digitize(u_i, u_edges) - 1, 0, n_bins - 1)

        for b in range(n_bins):
            mask = bin_idx == b
            if not mask.any():
                continue
            h, _ = np.histogram(psi_i[mask], bins=psi_bin_edges)
            phase_hist[b] += h
            cnt[b] += mask.sum()

    entropy = np.full(n_bins, np.nan)
    entropy_norm = np.full(n_bins, np.nan)
    for b in range(n_bins):
        if cnt[b] < n_min:
            continue
        p = phase_hist[b] / cnt[b]
        # Avoid log(0): only sum over p > 0
        pos = p > 0
        H = -np.sum(p[pos] * np.log(p[pos]))
        entropy[b] = H
        entropy_norm[b] = H / max_H

    return {
        "entropy": entropy,
        "entropy_norm": entropy_norm,
        "counts": cnt,
    }
