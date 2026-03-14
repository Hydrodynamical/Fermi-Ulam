"""Wall interaction rules.

Left wall (x=0):  elastic reflection  v -> -v
Right wall (x=L):
  - sigma=0 (unmixed): deterministic moving-wall reflection
  - sigma=1 (mixed):   Euler-Maruyama step for an OU process in outgoing speed

Mixed-wall kernel (FP/OU discretization)
-----------------------------------------
At each right-wall hit the outgoing speed s = |v| evolves by one
Euler-Maruyama step of the Ornstein-Uhlenbeck SDE

    ds = kappa_w (V* - s) dt_w  +  sqrt(2 D_w) dW,

giving

    s+ = s- + kappa_w (V* - s-) dt_w  +  sqrt(2 D_w dt_w) zeta,
    s+ = |s+|                           (reflection at 0)
    v+ = -s+                            (inward after right-wall hit)

This is the one-hit Euler-Maruyama discretization of the FP generator

    L_wall f = -d/ds [ kappa_w (V* - s) f ]  +  D_w d²f/ds²

Parameters
----------
kappa_w  : drift rate toward wall temperature scale V*  [1/time]
V_star   : target speed scale (wall temperature sqrt)   [speed]
D_w      : diffusion coefficient in speed space         [speed²/time]
wall_dtau: effective wall-interaction time per hit      [time]

The old affine-reset parameter eta maps to eta = kappa_w * wall_dtau.
"""
import numpy as np


def _wall_velocity(t: float, A: float, omega: float) -> float:
    return A * omega * np.cos(omega * t)


def apply_left_wall(particles, hits: np.ndarray):
    """Elastic reflection at x=0 for particles with hits[i]=True."""
    particles.x[hits] = -particles.x[hits]          # fold back into domain
    particles.v[hits] = -particles.v[hits]


def apply_right_wall(particles, hits: np.ndarray, t: float, cfg: dict,
                     rng: np.random.Generator):
    """
    Right wall at x=L.
    For unmixed (sigma=0): deterministic vibrating-wall reflection.
    For mixed   (sigma=1): stochastic wall-heating kernel.
    """
    L = cfg["L"]
    # Fold position back (overshoot)
    overshoot = particles.x[hits] - L
    particles.x[hits] = L - overshoot  # reflect position

    unmixed = hits & ~particles.sigma
    mixed   = hits &  particles.sigma

    # --- Deterministic moving-wall reflection ---
    if unmixed.any():
        U_w = _wall_velocity(t, cfg["A"], cfg["omega"])
        v_in = particles.v[unmixed]
        particles.v[unmixed] = 2.0 * U_w - v_in  # v+ = 2*U_w - v-

    # --- Stochastic wall-heating kernel (OU/Fokker-Planck, Euler-Maruyama) ---
    if mixed.any():
        kappa = cfg["kappa_w"]
        dtau  = cfg["wall_dtau"]
        V_s   = cfg["V_star"]
        D_w   = cfg["D_w"]
        s_in  = np.abs(particles.v[mixed])
        n_hit = mixed.sum()

        speed_out = s_in + kappa * (V_s - s_in) * dtau \
                    + np.sqrt(2.0 * D_w * dtau) * rng.standard_normal(n_hit)
        speed_out = np.abs(speed_out)   # reflect at 0: no absorbing state, preserves FP symmetry
        particles.v[mixed] = -speed_out  # inward (leftward) after right-wall hit
