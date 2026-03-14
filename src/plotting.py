"""Plotting utilities for the wall-heating simulation."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_energy(data: dict, out_path=None):
    fig, ax = plt.subplots()
    ax.plot(data["t"], data["E"])
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean kinetic energy E(t)")
    ax.set_title("Wall heating: energy growth")
    ax.set_yscale("log")
    _save_or_show(fig, out_path)


def plot_mixing_fraction(data: dict, out_path=None):
    fig, ax = plt.subplots()
    ax.plot(data["t"], data["p_mix"])
    ax.set_xlabel("Time")
    ax.set_ylabel("Fraction mixed p_mix(t)")
    ax.set_title("Mixed-particle fraction over time")
    ax.set_ylim(0, 1)
    _save_or_show(fig, out_path)


def plot_velocity_hist(v: np.ndarray, t: float, out_path=None):
    fig, ax = plt.subplots()
    ax.hist(v, bins=60, density=True, color="steelblue", edgecolor="white", lw=0.3)
    ax.set_xlabel("Velocity v")
    ax.set_ylabel("Probability density")
    ax.set_title(f"Velocity distribution at t={t:.2f}")
    _save_or_show(fig, out_path)


def plot_phase_space(particles, t: float, L: float, out_path=None):
    fig, ax = plt.subplots()
    colors = np.where(particles.sigma, "crimson", "steelblue")
    ax.scatter(particles.x, particles.v, c=colors, s=1, alpha=0.4)
    ax.set_xlabel("Position x")
    ax.set_ylabel("Velocity v")
    ax.set_title(f"Phase space at t={t:.2f}  (red=mixed, blue=unmixed)")
    ax.set_xlim(0, L)
    _save_or_show(fig, out_path)


def _save_or_show(fig, out_path):
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()
