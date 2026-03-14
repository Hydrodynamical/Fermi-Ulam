"""Runtime sanity checks for the simulation."""
import numpy as np


def check_domain(particles, cfg: dict) -> list[str]:
    """Return list of warning strings if particles escape the domain."""
    L = cfg["L"]
    warnings = []
    out_left  = (particles.x < 0).sum()
    out_right = (particles.x > L).sum()
    if out_left:
        warnings.append(f"[diag] {out_left} particles left of x=0")
    if out_right:
        warnings.append(f"[diag] {out_right} particles right of x=L")
    return warnings


def check_energy(particles, prev_E: float, tol: float = 50.0) -> list[str]:
    """Warn if energy grows suspiciously fast in one step."""
    E = 0.5 * np.mean(particles.v ** 2)
    warnings = []
    if prev_E > 0 and E > tol * prev_E:
        warnings.append(f"[diag] Energy jumped from {prev_E:.3f} to {E:.3f}")
    return warnings
