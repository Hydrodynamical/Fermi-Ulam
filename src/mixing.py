"""Deterministic threshold mixing (Option A from the plan).

A particle becomes mixed (sigma=1) when its age since the last bulk
collision exceeds tau_mix(v):

    tau_mix(v) = m0 + m1 / (|v| + v_mix_floor)

This is the simplest PDMP realization of the wall-phase-mixing mechanism.
"""
import numpy as np


def mixing_time(v: np.ndarray, cfg: dict) -> np.ndarray:
    m0    = cfg["m0"]
    m1    = cfg["m1"]
    floor = cfg["v_mix_floor"]
    return m0 + m1 / (np.abs(v) + floor)


def apply_mixing(particles, cfg: dict):
    """Set sigma=True for particles whose age exceeds their mixing time."""
    tau = mixing_time(particles.v, cfg)
    newly_mixed = (~particles.sigma) & (particles.age >= tau)
    particles.sigma[newly_mixed] = True
    return newly_mixed.sum()
