"""Increment-law diagnostics for the calibration experiment.

Answers: Is the Fermi-Ulam map diffusive in u? Or is log(u) a better variable?
Is the diffusion closure justified at each energy?

Functions
---------
compute_increment_moments   -- mean, var, skewness, kurtosis, quantiles for Δu and Δlog(u)
compute_small_jump_ratios   -- P(|Δu| ≤ η·u_n | u_n in bin)
compute_variance_vs_lag     -- Var(Δu) and Var(Δlog u) as a function of lag

Decision rules
--------------
Diffusion in u is valid if:
  - skewness(Δu) ≈ 0, excess kurtosis(Δu) moderate (< 5)
  - small_jump_ratio at η=0.1 is large (> 0.5)
  - var_u scales roughly linearly in lag

If instead:
  - skewness(Δlog u) is much better-behaved than skewness(Δu)
  - var_logu scales better linearly in lag
→ log-energy is the right closure variable.
"""
import numpy as np


def compute_increment_moments(
    u_traj: np.ndarray,
    u_edges: np.ndarray,
    m_arr: np.ndarray,
    n_min: int = 50,
) -> dict:
    """Compute mean, variance, skewness, excess kurtosis, and quantiles of
    Δu and Δlog(u) per energy bin at the coarse lag m(u).

    Parameters
    ----------
    u_traj  : (N_particles, N_hits+1)
    u_edges : (n_bins+1,)
    m_arr   : (n_bins,)  int64  coarse lag per bin
    n_min   : minimum samples per bin

    Returns
    -------
    dict with keys for both u and log(u) increments:
      mean_u, var_u, skew_u, kurt_u         : (n_bins,)
      mean_logu, var_logu, skew_logu, kurt_logu : (n_bins,)
      quantiles_u   : (n_bins, 7)  quantiles at [1,5,25,50,75,95,99]%
      quantiles_logu: (n_bins, 7)
      counts        : (n_bins,)
    """
    N_p, N_h1 = u_traj.shape
    N_h = N_h1 - 1
    n_bins = len(u_edges) - 1
    q_levels = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]

    # Collect all increments per bin (stored as lists for variable length)
    du_lists = [[] for _ in range(n_bins)]
    dlogu_lists = [[] for _ in range(n_bins)]

    for i in range(N_p):
        u_i = u_traj[i]
        for b in range(n_bins):
            m = int(m_arr[b])
            if m > N_h:
                continue
            u_base = u_i[: N_h + 1 - m]
            in_bin = (u_base >= u_edges[b]) & (u_base < u_edges[b + 1])
            if not in_bin.any():
                continue
            base_idx = np.where(in_bin)[0]
            u_now = u_base[in_bin]
            u_next = u_i[base_idx + m]

            du_lists[b].append(u_next - u_now)
            # Guard log(0): u_next should be > 0 by map construction, but clamp
            dlogu_lists[b].append(
                np.log(np.maximum(u_next, 1e-300)) - np.log(np.maximum(u_now, 1e-300))
            )

    # Compute statistics
    mean_u = np.full(n_bins, np.nan)
    var_u = np.full(n_bins, np.nan)
    skew_u = np.full(n_bins, np.nan)
    kurt_u = np.full(n_bins, np.nan)
    quantiles_u = np.full((n_bins, 7), np.nan)

    mean_logu = np.full(n_bins, np.nan)
    var_logu = np.full(n_bins, np.nan)
    skew_logu = np.full(n_bins, np.nan)
    kurt_logu = np.full(n_bins, np.nan)
    quantiles_logu = np.full((n_bins, 7), np.nan)

    counts = np.zeros(n_bins, dtype=np.int64)

    for b in range(n_bins):
        if len(du_lists[b]) == 0:
            continue
        du = np.concatenate(du_lists[b])
        dlogu = np.concatenate(dlogu_lists[b])
        n = len(du)
        counts[b] = n
        if n < n_min:
            continue

        mean_u[b], var_u[b], skew_u[b], kurt_u[b] = _moments4(du)
        mean_logu[b], var_logu[b], skew_logu[b], kurt_logu[b] = _moments4(dlogu)
        quantiles_u[b] = np.quantile(du, q_levels)
        quantiles_logu[b] = np.quantile(dlogu, q_levels)

    return {
        "mean_u": mean_u,
        "var_u": var_u,
        "skew_u": skew_u,
        "kurt_u": kurt_u,
        "mean_logu": mean_logu,
        "var_logu": var_logu,
        "skew_logu": skew_logu,
        "kurt_logu": kurt_logu,
        "quantiles_u": quantiles_u,
        "quantiles_logu": quantiles_logu,
        "counts": counts,
    }


def compute_small_jump_ratios(
    u_traj: np.ndarray,
    u_edges: np.ndarray,
    m_arr: np.ndarray,
    eta_values: tuple = (0.05, 0.1, 0.2, 0.5),
    n_min: int = 50,
) -> dict:
    """Probability that |Δu| ≤ η·u_n at the coarse lag m(u), per energy bin.

    R[b, j] = P(|u_{n+m} - u_n| ≤ eta_j * u_n  |  u_n in bin b)

    For the diffusion closure to hold, Δu must be small relative to u_n;
    this probability should be large (> 0.5) at diffusion-valid energies.

    Returns
    -------
    dict with keys:
      ratios     : (n_bins, n_eta)  probability per (bin, eta)
      eta_values : (n_eta,)
      counts     : (n_bins,)
    """
    N_p, N_h1 = u_traj.shape
    N_h = N_h1 - 1
    n_bins = len(u_edges) - 1
    n_eta = len(eta_values)
    etas = np.asarray(eta_values)

    small_cnt = np.zeros((n_bins, n_eta), dtype=np.int64)
    total_cnt = np.zeros(n_bins, dtype=np.int64)

    for i in range(N_p):
        u_i = u_traj[i]
        for b in range(n_bins):
            m = int(m_arr[b])
            if m > N_h:
                continue
            u_base = u_i[: N_h + 1 - m]
            in_bin = (u_base >= u_edges[b]) & (u_base < u_edges[b + 1])
            if not in_bin.any():
                continue
            base_idx = np.where(in_bin)[0]
            u_now = u_base[in_bin]
            u_next = u_i[base_idx + m]
            du_abs = np.abs(u_next - u_now)

            for j, eta in enumerate(etas):
                small_cnt[b, j] += (du_abs <= eta * u_now).sum()
            total_cnt[b] += len(base_idx)

    ratios = np.full((n_bins, n_eta), np.nan)
    for b in range(n_bins):
        if total_cnt[b] >= n_min:
            ratios[b] = small_cnt[b] / total_cnt[b]

    return {
        "ratios": ratios,
        "eta_values": etas,
        "counts": total_cnt,
    }


def compute_variance_vs_lag(
    u_traj: np.ndarray,
    u_edges: np.ndarray,
    lag_values: tuple = (1, 2, 4, 8, 16, 32, 64),
    n_min: int = 50,
) -> dict:
    """Var(Δu | u_n in bin) and Var(Δlog u | u_n in bin) as a function of lag.

    For a pure diffusion, Var(Δu) ≈ a(u) · m  (linear scaling).
    Plotting Var vs lag on log-log axes should show slope ≈ 1.
    Slope < 1 → subdiffusive; slope > 1 → superdiffusive / jump-like.

    Uses lag_values directly (not the per-bin m_arr) so all bins are evaluated
    at the same lags, allowing easy cross-bin comparison.

    Returns
    -------
    dict with keys:
      var_u     : (n_bins, n_lags)
      var_logu  : (n_bins, n_lags)
      lag_values: (n_lags,)
      counts    : (n_bins, n_lags)  number of (particle, base-hit) pairs per (bin, lag)
    """
    N_p, N_h1 = u_traj.shape
    N_h = N_h1 - 1
    n_bins = len(u_edges) - 1
    lags = np.asarray(lag_values, dtype=int)
    n_lags = len(lags)
    m_max = int(lags.max())

    sum2_u = np.zeros((n_bins, n_lags))
    sum1_u = np.zeros((n_bins, n_lags))
    sum2_logu = np.zeros((n_bins, n_lags))
    sum1_logu = np.zeros((n_bins, n_lags))
    cnt = np.zeros((n_bins, n_lags), dtype=np.int64)

    for i in range(N_p):
        u_i = u_traj[i]
        for li, m in enumerate(lags):
            if m > N_h:
                continue
            u_base = u_i[: N_h + 1 - m]
            bin_idx = np.clip(np.digitize(u_base, u_edges) - 1, 0, n_bins - 1)
            u_next = u_i[m:]

            du = u_next - u_base
            dlogu = np.log(np.maximum(u_next, 1e-300)) - np.log(np.maximum(u_base, 1e-300))

            np.add.at(sum1_u[:, li], bin_idx, du)
            np.add.at(sum2_u[:, li], bin_idx, du ** 2)
            np.add.at(sum1_logu[:, li], bin_idx, dlogu)
            np.add.at(sum2_logu[:, li], bin_idx, dlogu ** 2)
            np.add.at(cnt[:, li], bin_idx, 1)

    var_u = np.full((n_bins, n_lags), np.nan)
    var_logu = np.full((n_bins, n_lags), np.nan)
    for b in range(n_bins):
        for li in range(n_lags):
            n = cnt[b, li]
            if n < n_min:
                continue
            mean_u = sum1_u[b, li] / n
            var_u[b, li] = sum2_u[b, li] / n - mean_u ** 2
            mean_logu = sum1_logu[b, li] / n
            var_logu[b, li] = sum2_logu[b, li] / n - mean_logu ** 2

    return {
        "var_u": var_u,
        "var_logu": var_logu,
        "lag_values": lags,
        "counts": cnt,
    }


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _moments4(x: np.ndarray):
    """Return (mean, variance, skewness, excess kurtosis) of array x."""
    n = len(x)
    if n < 4:
        return np.nan, np.nan, np.nan, np.nan
    mu = x.mean()
    sigma2 = x.var()
    if sigma2 <= 0:
        return mu, 0.0, 0.0, 0.0
    sigma = np.sqrt(sigma2)
    z = (x - mu) / sigma
    skew = (z ** 3).mean()
    kurt = (z ** 4).mean() - 3.0  # excess kurtosis
    return mu, sigma2, skew, kurt
