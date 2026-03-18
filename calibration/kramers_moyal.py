"""Kramers-Moyal (KM) coefficient estimation for the wall operator.

Given the energy sequence u_0, u_1, ..., u_N at right-wall hits, and a
coarse lag m = m(u_bin) chosen to exceed tau_lag(u), the KM coefficients are:

    b(u) = E[ (u_{n+m} - u_n) / m  |  u_n in bin ]      [drift]
    a(u) = Var[ u_{n+m} - u_n ] / m  |  u_n in bin ]    [diffusion]

where Var denotes the centred second moment.  Division by m converts the
sum over m bounce steps into a per-bounce rate.

Convention note
---------------
These are per-bounce-count rates.  The FP equation in physical time requires

    b_time(u) = b_bounce(u) / T_bar(u)
    a_time(u) = a_bounce(u) / T_bar(u)

where T_bar(u) = 2*L / sqrt(2*u) is the mean round-trip time.  However, for
computing the steady-state density (which depends only on b/a), the T_bar
factor cancels exactly, so per-bounce coefficients are sufficient for the
Fokker-Planck validation step.  Run fokker_planck.py with per-bounce b and a
directly; if you later need physical-time coefficients, call
convert_to_physical_time() below.

Lag sensitivity
---------------
For a well-mixed Markov process in u, b(u) and a(u) should be independent
of m (once m > tau_lag).  The caller can verify this by re-running
estimate_km_coefficients with different m_safety_factor values and comparing.

Loop structure
--------------
Outer over N_particles (~500), inner over n_bins (~40).
Each (i, b) body does vectorised NumPy over base hits in that bin (~N_hits/n_bins).
Total Python iterations: N_particles * n_bins = ~20 000.
"""
import numpy as np


def estimate_km_coefficients(
    u_traj: np.ndarray,
    u_edges: np.ndarray,
    m_arr: np.ndarray,
    n_min: int = 200,
) -> dict:
    """Estimate per-bounce KM drift b(u) and diffusion a(u).

    Parameters
    ----------
    u_traj  : (N_particles, N_hits+1)
    u_edges : (n_bins+1,)
    m_arr   : (n_bins,)  int  coarse lag per bin (from choose_coarse_lag)
    n_min   : minimum (particle, base-hit) pairs required to report a bin

    Returns
    -------
    dict with keys:
      'b'      : (n_bins,)  per-bounce drift  (NaN for under-sampled bins)
      'a'      : (n_bins,)  per-bounce diffusion  (centred second moment / m)
      'counts' : (n_bins,)  int64  number of (particle, base-hit) pairs used
    """
    n_bins = len(u_edges) - 1
    N_p, N_h1 = u_traj.shape
    N_h = N_h1 - 1

    m_max = int(m_arr.max())

    sum1 = np.zeros(n_bins, dtype=np.float64)   # sum of delta_u
    sum2 = np.zeros(n_bins, dtype=np.float64)   # sum of delta_u^2
    cnt = np.zeros(n_bins, dtype=np.int64)

    for i in range(N_p):
        u_i = u_traj[i]  # (N_h+1,)

        for b in range(n_bins):
            m = int(m_arr[b])
            if m > N_h:
                continue  # not enough hits for this lag

            # Base hits: u_i[n] in bin b, with room for lag m.
            u_base = u_i[: N_h + 1 - m]   # (N_h+1-m,)
            in_bin = (u_base >= u_edges[b]) & (u_base < u_edges[b + 1])
            if not in_bin.any():
                continue

            base_idx = np.where(in_bin)[0]
            du = u_i[base_idx + m] - u_i[base_idx]

            sum1[b] += du.sum()
            sum2[b] += (du ** 2).sum()
            cnt[b] += len(base_idx)

    # Drift: mean(delta_u) / m
    b_est = np.full(n_bins, np.nan)
    a_est = np.full(n_bins, np.nan)

    for b in range(n_bins):
        if cnt[b] < n_min:
            continue
        m = int(m_arr[b])
        n = cnt[b]
        mean_du = sum1[b] / n
        b_est[b] = mean_du / m
        # Centred second moment: E[(du - E[du])^2]
        var_du = sum2[b] / n - mean_du ** 2
        a_est[b] = var_du / m

    return {
        "b": b_est,
        "a": a_est,
        "counts": cnt,
    }


def convert_to_physical_time(
    u_centers: np.ndarray,
    b_bounce: np.ndarray,
    a_bounce: np.ndarray,
    L: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert per-bounce KM coefficients to physical-time rates.

    T_bar(u) = 2*L / sqrt(2*u)  (mean round-trip time at energy u)

    b_time = b_bounce / T_bar(u)
    a_time = a_bounce / T_bar(u)

    For the FP steady-state formula, b/a is time-scale invariant and this
    conversion is unnecessary.  Call this only if you need dimensional rates.

    Returns
    -------
    b_time, a_time : (n_bins,) arrays
    """
    s_bar = np.sqrt(2.0 * u_centers)
    T_bar = 2.0 * L / np.maximum(s_bar, 1e-12)
    b_time = b_bounce / T_bar
    a_time = a_bounce / T_bar
    return b_time, a_time
