"""Ensemble generation and config loading for the calibration experiment.

Config loading chain
--------------------
1. src/config.py:load_config()  -> base.yaml  (gets A, omega, L, seed, ...)
2. config/calibration_base.yaml              (adds N_particles, N_hits, u_min, ...)
3. optional override YAML                    (experiment-specific overrides)

The chain ensures physical parameters (A, omega, L) in the calibration are
always consistent with the main PDMP simulator.
"""
import sys
from pathlib import Path

import numpy as np
import yaml

# Allow importing src/config.py from any working directory.
# Use append (not insert) so calibration/ stays first in sys.path when
# this module is imported by run_calibration.py.
_REPO = Path(__file__).parent.parent
_src = str(_REPO / "src")
if _src not in sys.path:
    sys.path.append(_src)
from config import load_config  # noqa: E402

_CAL_BASE = _REPO / "config" / "calibration_base.yaml"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_cal_config(override_path=None) -> dict:
    """Load and merge the three-layer config.

    Returns a flat dict with all keys from base.yaml,
    calibration_base.yaml, and optionally override_path.
    Later layers win on conflict.
    """
    cfg = load_config()  # base.yaml

    with open(_CAL_BASE) as f:
        cal_defaults = yaml.safe_load(f)
    cfg.update(cal_defaults)

    if override_path is not None:
        with open(override_path) as f:
            override = yaml.safe_load(f)
        cfg.update(override)

    return cfg


# ---------------------------------------------------------------------------
# Energy binning
# ---------------------------------------------------------------------------

def make_energy_bins(u_min: float, u_max: float, n_bins: int):
    """Log-spaced energy bin edges and centres.

    Returns
    -------
    edges   : (n_bins+1,) float64
    centers : (n_bins,)   float64
    """
    edges = np.logspace(np.log10(u_min), np.log10(u_max), n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])  # geometric mean (log-midpoint)
    return edges, centers


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

def sample_initial_conditions(
    N: int,
    u_min: float,
    u_max: float,
    rng: np.random.Generator,
):
    """Sample (u0, psi0) for N independent trajectories.

    u0   : log-uniform in [u_min, u_max] — equal density per decade
    psi0 : uniform in [0, 2*pi)
    """
    log_u = rng.uniform(np.log(u_min), np.log(u_max), size=N)
    u0 = np.exp(log_u)
    psi0 = rng.uniform(0.0, 2.0 * np.pi, size=N)
    return u0, psi0


# ---------------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------------

def generate_trajectories(cfg: dict) -> dict:
    """Run the full ensemble and return raw trajectory arrays.

    Memory estimate is printed; a warning is issued if > 2 GB.

    Returns
    -------
    dict with keys:
      'u_traj'   : (N_particles, N_hits+1) float64
      'psi_traj' : (N_particles, N_hits+1) float64
      't_traj'   : (N_particles, N_hits+1) float64
      'cfg'      : the config dict used
    """
    # Import here to avoid circular dependency at module level.
    from map import run_ensemble  # noqa: E402

    rng = np.random.default_rng(cfg["seed"])
    N_p = cfg["N_particles"]
    N_h = cfg["N_hits"]

    mem_gb = N_p * (N_h + 1) * 3 * 8 / 1e9
    print(f"[trajectories] Memory estimate: {mem_gb:.2f} GB")
    if mem_gb > 2.0:
        print(
            f"[trajectories] WARNING: {mem_gb:.1f} GB is large. "
            "Consider reducing N_particles or N_hits."
        )

    u0, psi0 = sample_initial_conditions(N_p, cfg["u_min"], cfg["u_max"], rng)

    print(
        f"[trajectories] Running {N_p} particles × {N_h} hits "
        f"(A={cfg['A']}, omega={cfg['omega']}, L={cfg['L']}) ..."
    )
    u_traj, psi_traj, t_traj = run_ensemble(
        u0, psi0, N_h,
        cfg["A"], cfg["omega"], cfg["L"],
    )
    print("[trajectories] Done.")

    return {
        "u_traj": u_traj,
        "psi_traj": psi_traj,
        "t_traj": t_traj,
        "cfg": cfg,
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_trajectories(data: dict, path):
    """Save u_traj, psi_traj, t_traj to a compressed .npz file."""
    np.savez_compressed(
        path,
        u_traj=data["u_traj"],
        psi_traj=data["psi_traj"],
        t_traj=data["t_traj"],
    )


def load_trajectories(path) -> dict:
    """Load trajectory arrays from a .npz file."""
    d = np.load(path)
    return {k: d[k] for k in d.files}
