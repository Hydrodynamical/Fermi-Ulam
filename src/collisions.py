"""Bulk collision logic (Bird/DSMC-style mean-field).

Collision rate for particle i:
  nu_i = lambda0 * n_loc(x_i) * (|v_i - v_bar_loc(x_i)| + v_min)

At each timestep, particle i collides with probability nu_i * dt.
Partner j is chosen from the same spatial bin with probability
proportional to |v_i - v_j| + eps_v.

Collision rule (1D center-of-mass randomization, energy-conserving):
  V_cm = (v_i + v_j) / 2
  g    = v_i - v_j
  g'   = s * |g|,  s in {-1, +1} with prob 1/2
  v_i' = V_cm + g'/2
  v_j' = V_cm - g'/2
"""
import numpy as np


def _bin_particles(x: np.ndarray, L: float, n_bins: int):
    """Return bin index for each particle."""
    edges = np.linspace(0.0, L, n_bins + 1)
    bins  = np.digitize(x, edges) - 1
    bins  = np.clip(bins, 0, n_bins - 1)
    return bins, edges


def compute_local_density(particles, cfg: dict):
    """Estimate n_loc and v_bar_loc for each particle using spatial bins."""
    n_bins = cfg["n_bins"]
    L      = cfg["L"]
    bins, edges = _bin_particles(particles.x, L, n_bins)
    dx = L / n_bins

    n_loc    = np.zeros(particles.N)
    vbar_loc = np.zeros(particles.N)

    for b in range(n_bins):
        mask = (bins == b)
        count = mask.sum()
        if count > 0:
            n_loc[mask]    = count / (dx * particles.N)
            vbar_loc[mask] = particles.v[mask].mean()

    return n_loc, vbar_loc, bins


def collision_step(particles, cfg: dict, dt: float, rng: np.random.Generator,
                   stats: dict):
    """
    Attempt bulk collisions for all particles.
    Returns number of collisions performed.
    """
    lambda0  = cfg["lambda0"]
    v_min    = cfg["v_min"]
    eps_v    = cfg["eps_v"]

    n_loc, vbar_loc, bins = compute_local_density(particles, cfg)

    rel_speed = np.abs(particles.v - vbar_loc) + v_min
    nu        = lambda0 * n_loc * rel_speed
    prob      = 1.0 - np.exp(-nu * dt)   # exact Poisson probability
    will_collide = rng.random(particles.N) < prob

    n_collisions   = 0
    n_coll_mixed   = 0   # collisions where particle i was already mixed (sigma=1)
    n_coll_unmixed = 0   # collisions where particle i was not yet mixed (sigma=0)
    candidate_indices = np.where(will_collide)[0]

    for i in candidate_indices:
        b = bins[i]
        # Potential partners: same bin, not self
        partners = np.where((bins == b) & (np.arange(particles.N) != i))[0]
        if len(partners) == 0:
            continue

        # Choose partner weighted by relative speed
        rel = np.abs(particles.v[i] - particles.v[partners]) + eps_v
        prob_partner = rel / rel.sum()
        j = rng.choice(partners, p=prob_partner)

        # Track mixing state at collision time (chi measurement)
        if particles.sigma[i]:
            n_coll_mixed += 1
        else:
            n_coll_unmixed += 1

        # 1D CM-randomization collision
        V_cm = 0.5 * (particles.v[i] + particles.v[j])
        g    = particles.v[i] - particles.v[j]
        s    = rng.choice([-1.0, 1.0])
        g_prime = s * np.abs(g)

        particles.v[i] = V_cm + 0.5 * g_prime
        particles.v[j] = V_cm - 0.5 * g_prime

        # Reset wall-phase memory
        particles.reset_mixing([i, j])
        n_collisions += 1

    stats["n_bulk_collisions"]    = n_collisions
    stats["n_coll_mixed_total"]   = stats.get("n_coll_mixed_total",   0) + n_coll_mixed
    stats["n_coll_unmixed_total"] = stats.get("n_coll_unmixed_total", 0) + n_coll_unmixed
    return n_collisions
