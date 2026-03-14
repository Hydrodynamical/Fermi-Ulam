"""Particle state arrays for the 1D PDMP wall heating simulation.

State per particle:
  x     : position in [0, L]
  v     : velocity (signed, positive = rightward)
  sigma : mixing state (False = unmixed/deterministic, True = mixed/chaotic)
  age   : time since last bulk collision (used for threshold mixing)
"""
import numpy as np


class Particles:
    def __init__(self, N: int, L: float, v_init_scale: float, rng: np.random.Generator):
        self.N = N
        self.L = L

        # Positions: uniform in (0, L)
        self.x = rng.uniform(0.01 * L, 0.99 * L, size=N)

        # Velocities: zero-mean Gaussian, then shift slightly so not all zero
        self.v = rng.normal(0.0, v_init_scale, size=N)

        # Mixing states: all unmixed initially
        self.sigma = np.zeros(N, dtype=bool)

        # Age since last bulk collision
        self.age = np.zeros(N, dtype=float)

    def advect(self, dt: float):
        """Move particles ballistically; age ticks forward."""
        self.x += self.v * dt
        self.age += dt

    def reset_mixing(self, idx):
        """Called after a bulk collision resets wall-phase memory."""
        self.sigma[idx] = False
        self.age[idx] = 0.0
