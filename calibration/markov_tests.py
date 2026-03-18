"""Markovianity diagnostics for the calibration experiment.

Answers: Is the coarse-grained process Markov in u alone?
Does the drift depend on the current phase ψ?

Functions
---------
compute_phase_conditioned_moments  -- E[Δu|u,ψ] and Var[Δu|u,ψ] on a 2D grid
test_semigroup_consistency         -- checks E[Δu_{2m}] ≈ 2·E[Δu_m]
test_lag1_autocorrelation          -- Corr(Δu_n, Δu_{n-m}) within each bin

Interpretation
--------------
phase_conditioned drift map:
  - Flat in ψ → Markov reduction to u alone is valid
  - Varies strongly in ψ → hidden memory; process is not Markov in u

semigroup:
  - drift_ratio ≈ 1, var_ratio ≈ 1 → diffusion-Markov consistent
  - drift_ratio != 1 → non-linear drift or non-Markov

lag1_autocorr:
  - Near zero → increments are uncorrelated (consistent with Markov)
  - Significantly nonzero → memory at the coarse scale
"""
import numpy as np

TWO_PI = 2.0 * np.pi


def compute_phase_conditioned_moments(
    u_traj: np.ndarray,
    psi_traj: np.ndarray,
    u_edges: np.ndarray,
    m_arr: np.ndarray,
    n_psi_bins: int = 12,
    n_min: int = 20,
) -> dict:
    """E[Δu | u_n in u_bin, ψ_n in ψ_bin] on a (n_u_bins × n_psi_bins) grid.

    This is the most informative single diagnostic. If the map has been
    reduced correctly to a Markov process in u, the drift should be nearly
    independent of ψ. Persistent ψ-dependence means the closure is missing
    a slow variable.

    Lag used per u_bin: m = m_arr[u_bin] (same coarse lag as KM estimation).

    Parameters
    ----------
    u_traj     : (N_particles, N_hits+1)
    psi_traj   : (N_particles, N_hits+1)
    u_edges    : (n_u_bins+1,)
    m_arr      : (n_u_bins,) int64
    n_psi_bins : number of ψ phase bins in [0, 2π)
    n_min      : minimum samples per 2D cell to report

    Returns
    -------
    dict with keys:
      drift_map  : (n_u_bins, n_psi_bins)  E[Δu | u_bin, ψ_bin]; NaN if < n_min
      var_map    : (n_u_bins, n_psi_bins)  Var[Δu | u_bin, ψ_bin]
      counts_map : (n_u_bins, n_psi_bins)  int64 sample counts
      psi_edges  : (n_psi_bins+1,)
    """
    N_p, N_h1 = u_traj.shape
    N_h = N_h1 - 1
    n_u = len(u_edges) - 1

    psi_edges = np.linspace(0.0, TWO_PI, n_psi_bins + 1)

    sum1 = np.zeros((n_u, n_psi_bins))   # sum of Δu
    sum2 = np.zeros((n_u, n_psi_bins))   # sum of (Δu)²
    cnt  = np.zeros((n_u, n_psi_bins), dtype=np.int64)

    for i in range(N_p):
        u_i   = u_traj[i]
        psi_i = psi_traj[i]

        for b in range(n_u):
            m = int(m_arr[b])
            if m > N_h:
                continue
            u_base = u_i[: N_h + 1 - m]
            psi_base = psi_i[: N_h + 1 - m]
            in_u_bin = (u_base >= u_edges[b]) & (u_base < u_edges[b + 1])
            if not in_u_bin.any():
                continue
            base_idx = np.where(in_u_bin)[0]

            du = u_i[base_idx + m] - u_base[in_u_bin]
            psi_now = psi_base[in_u_bin]

            # Phase bin index
            psi_idx = np.clip(
                np.digitize(psi_now, psi_edges) - 1, 0, n_psi_bins - 1
            )
            np.add.at(sum1[b], psi_idx, du)
            np.add.at(sum2[b], psi_idx, du ** 2)
            np.add.at(cnt[b], psi_idx, 1)

    drift_map = np.full((n_u, n_psi_bins), np.nan)
    var_map   = np.full((n_u, n_psi_bins), np.nan)
    for b in range(n_u):
        for p in range(n_psi_bins):
            n = cnt[b, p]
            if n < n_min:
                continue
            mean_du = sum1[b, p] / n
            drift_map[b, p] = mean_du
            var_map[b, p] = sum2[b, p] / n - mean_du ** 2

    return {
        "drift_map":  drift_map,
        "var_map":    var_map,
        "counts_map": cnt,
        "psi_edges":  psi_edges,
    }


def test_semigroup_consistency(
    u_traj: np.ndarray,
    u_edges: np.ndarray,
    m_arr: np.ndarray,
    n_min: int = 50,
) -> dict:
    """Check Chapman-Kolmogorov / semigroup consistency at double the coarse lag.

    For a diffusion with generator L_wall:
        E[u_{n+2m} - u_n | u_n in bin] ≈ 2 · E[u_{n+m} - u_n | u_n in bin]
        Var[u_{n+2m} - u_n | u_n in bin] ≈ 2 · Var[u_{n+m} - u_n | u_n in bin]

    Ratios close to 1 support the Markov-diffusion picture;
    ratios far from 1 indicate memory or nonlinear drift.

    Returns
    -------
    dict with keys:
      drift_ratio : (n_bins,)  E[Δu_2m] / (2 * E[Δu_m])
      var_ratio   : (n_bins,)  Var[Δu_2m] / (2 * Var[Δu_m])
      counts_m    : (n_bins,)  sample count at lag m
      counts_2m   : (n_bins,)  sample count at lag 2m
    """
    N_p, N_h1 = u_traj.shape
    N_h = N_h1 - 1
    n_bins = len(u_edges) - 1

    # Collect at lag m and 2m simultaneously
    sum1_m   = np.zeros(n_bins)
    sum2_m   = np.zeros(n_bins)
    sum1_2m  = np.zeros(n_bins)
    sum2_2m  = np.zeros(n_bins)
    cnt_m    = np.zeros(n_bins, dtype=np.int64)
    cnt_2m   = np.zeros(n_bins, dtype=np.int64)

    for i in range(N_p):
        u_i = u_traj[i]
        for b in range(n_bins):
            m = int(m_arr[b])
            m2 = 2 * m

            # Lag m
            if m <= N_h:
                u_base_m = u_i[: N_h + 1 - m]
                in_bin = (u_base_m >= u_edges[b]) & (u_base_m < u_edges[b + 1])
                if in_bin.any():
                    base_idx = np.where(in_bin)[0]
                    du_m = u_i[base_idx + m] - u_base_m[in_bin]
                    sum1_m[b] += du_m.sum()
                    sum2_m[b] += (du_m ** 2).sum()
                    cnt_m[b]  += len(base_idx)

            # Lag 2m
            if m2 <= N_h:
                u_base_2m = u_i[: N_h + 1 - m2]
                in_bin2 = (u_base_2m >= u_edges[b]) & (u_base_2m < u_edges[b + 1])
                if in_bin2.any():
                    base_idx2 = np.where(in_bin2)[0]
                    du_2m = u_i[base_idx2 + m2] - u_base_2m[in_bin2]
                    sum1_2m[b] += du_2m.sum()
                    sum2_2m[b] += (du_2m ** 2).sum()
                    cnt_2m[b]  += len(base_idx2)

    drift_ratio = np.full(n_bins, np.nan)
    var_ratio   = np.full(n_bins, np.nan)

    for b in range(n_bins):
        if cnt_m[b] < n_min or cnt_2m[b] < n_min:
            continue
        nm  = cnt_m[b]
        n2m = cnt_2m[b]
        mean_m   = sum1_m[b] / nm
        var_m    = sum2_m[b] / nm - mean_m ** 2
        mean_2m  = sum1_2m[b] / n2m
        var_2m   = sum2_2m[b] / n2m - mean_2m ** 2

        # Avoid division by near-zero
        if abs(mean_m) > 1e-15:
            drift_ratio[b] = mean_2m / (2.0 * mean_m)
        if var_m > 1e-30:
            var_ratio[b] = var_2m / (2.0 * var_m)

    return {
        "drift_ratio": drift_ratio,
        "var_ratio":   var_ratio,
        "counts_m":    cnt_m,
        "counts_2m":   cnt_2m,
    }


def test_lag1_autocorrelation(
    u_traj: np.ndarray,
    u_edges: np.ndarray,
    m_arr: np.ndarray,
    n_min: int = 50,
) -> dict:
    """Pearson correlation between consecutive coarse increments Δu_n and Δu_{n+m}.

    For a Markov process Corr(Δu_n, Δu_{n+m}) should be near zero.
    Persistent nonzero correlation indicates memory at the coarse scale.

    Returns
    -------
    dict with keys:
      lag1_autocorr : (n_bins,)  Pearson correlation; NaN if insufficient data
      counts        : (n_bins,)
    """
    N_p, N_h1 = u_traj.shape
    N_h = N_h1 - 1
    n_bins = len(u_edges) - 1

    # For each bin, collect pairs (Δu_n, Δu_{n+m}) where u_n in bin
    # Pair condition: u[n], u[n+m], u[n+2m] all exist; u[n] in bin
    sum_x  = np.zeros(n_bins)
    sum_y  = np.zeros(n_bins)
    sum_xx = np.zeros(n_bins)
    sum_yy = np.zeros(n_bins)
    sum_xy = np.zeros(n_bins)
    cnt    = np.zeros(n_bins, dtype=np.int64)

    for i in range(N_p):
        u_i = u_traj[i]
        for b in range(n_bins):
            m = int(m_arr[b])
            m2 = 2 * m
            if m2 > N_h:
                continue
            # Base hits: u[n] in bin, with room for n+2m
            u_base = u_i[: N_h + 1 - m2]
            in_bin = (u_base >= u_edges[b]) & (u_base < u_edges[b + 1])
            if not in_bin.any():
                continue
            base_idx = np.where(in_bin)[0]
            du1 = u_i[base_idx + m]  - u_base[in_bin]    # Δu_n
            du2 = u_i[base_idx + m2] - u_i[base_idx + m]  # Δu_{n+m}

            sum_x[b]  += du1.sum()
            sum_y[b]  += du2.sum()
            sum_xx[b] += (du1 ** 2).sum()
            sum_yy[b] += (du2 ** 2).sum()
            sum_xy[b] += (du1 * du2).sum()
            cnt[b]    += len(base_idx)

    lag1_autocorr = np.full(n_bins, np.nan)
    for b in range(n_bins):
        n = cnt[b]
        if n < n_min:
            continue
        mx = sum_x[b] / n
        my = sum_y[b] / n
        vx = sum_xx[b] / n - mx ** 2
        vy = sum_yy[b] / n - my ** 2
        cov = sum_xy[b] / n - mx * my
        denom = np.sqrt(vx * vy)
        if denom > 1e-30:
            lag1_autocorr[b] = cov / denom

    return {
        "lag1_autocorr": lag1_autocorr,
        "counts": cnt,
    }
