"""Exact discrete Fermi-Ulam map at right-wall hit events.

State at the n-th right-wall hit:
  u_n   = 0.5 * s_n**2   kinetic energy  (s_n = |v_n| = incoming speed)
  psi_n = (omega * t_n) mod 2*pi   wall phase at the hit

Map equations
-------------
  U_w       = A * omega * cos(psi_n)          wall velocity at hit n
  s_{n+1}   = |2 * U_w - s_n|                 speed after reflection + left-wall bounce
  u_{n+1}   = 0.5 * s_{n+1}**2
  T_n       = 2 * L / s_{n+1}                 round-trip time to next right-wall hit
  psi_{n+1} = (psi_n + omega * T_n) mod 2*pi

Derivation: at the right wall, specular reflection off a moving wall gives
  v_out = 2 * U_w - v_in   (matches walls.py line 66: v+ = 2*U_w - v-)
After the right-wall hit the particle moves left, bounces elastically off
the fixed left wall (speed unchanged), and returns to the right wall.
The entire round trip takes T_n = 2*L / s_{n+1}.

Edge-case guard
---------------
If s_{n+1} = 0 the particle is stopped (T_n -> inf, psi_{n+1} undefined).
This is measure-zero in the chaotic sea but can arise in floating point near
KAM-curve boundaries.  We clamp s_{n+1} >= s_floor (default 1e-6) to keep
T_n finite.  The effect on statistics is negligible.

No collisions, no mixing-state transitions, no fixed time-step.
One call to step() = one exact wall-bounce event.
"""
import numpy as np

TWO_PI = 2.0 * np.pi


def step(
    u: np.ndarray,
    psi: np.ndarray,
    A: float,
    omega: float,
    L: float,
    s_floor: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance N particles by one right-wall-hit event (vectorised).

    Parameters
    ----------
    u       : (N,) incoming energy
    psi     : (N,) incoming wall phase in [0, 2*pi)
    A, omega, L : Fermi-Ulam map parameters (scalars)
    s_floor : minimum outgoing speed clamp

    Returns
    -------
    u_out   : (N,) outgoing energy
    psi_out : (N,) outgoing phase in [0, 2*pi)
    T       : (N,) round-trip time to next right-wall hit
    """
    s_in = np.sqrt(2.0 * u)
    U_w = A * omega * np.cos(psi)
    s_out = np.abs(2.0 * U_w - s_in)
    s_out = np.maximum(s_out, s_floor)
    u_out = 0.5 * s_out ** 2
    T = 2.0 * L / s_out
    psi_out = (psi + omega * T) % TWO_PI
    return u_out, psi_out, T


def run_ensemble(
    u0_arr: np.ndarray,
    psi0_arr: np.ndarray,
    N_hits: int,
    A: float,
    omega: float,
    L: float,
    s_floor: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run N_particles independent trajectories for N_hits steps.

    Initial conditions u0_arr and psi0_arr are arrays of shape (N_particles,).
    The inner loop runs N_hits Python iterations; each iteration applies the
    vectorised step() across all particles simultaneously.

    Memory: 3 arrays of shape (N_particles, N_hits+1) x float64.
    For N_particles=500, N_hits=100_000: ~1.2 GB.  Caller should check first.

    Returns
    -------
    u_traj   : (N_particles, N_hits+1)  energy at each hit (col 0 = initial)
    psi_traj : (N_particles, N_hits+1)  phase at each hit
    t_traj   : (N_particles, N_hits+1)  cumulative physical time
    """
    N = len(u0_arr)
    u_traj = np.empty((N, N_hits + 1), dtype=np.float64)
    psi_traj = np.empty((N, N_hits + 1), dtype=np.float64)
    t_traj = np.empty((N, N_hits + 1), dtype=np.float64)

    u_traj[:, 0] = u0_arr
    psi_traj[:, 0] = psi0_arr
    t_traj[:, 0] = 0.0

    u = u0_arr.copy()
    psi = psi0_arr.copy()
    t = np.zeros(N, dtype=np.float64)

    for n in range(N_hits):
        u, psi, T = step(u, psi, A, omega, L, s_floor)
        t += T
        u_traj[:, n + 1] = u
        psi_traj[:, n + 1] = psi
        t_traj[:, n + 1] = t

    return u_traj, psi_traj, t_traj
