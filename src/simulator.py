"""Main time-stepping simulator for the 1D PDMP wall-heating model.

Architecture: fixed-dt hybrid (Architecture 2 from the plan).

Each step:
  1. Advect all particles.
  2. Process left-wall hits (elastic).
  3. Process right-wall hits (deterministic or stochastic depending on sigma).
  4. Update mixing states (deterministic threshold).
  5. Sample bulk collisions.
  6. Record observables.
"""
import numpy as np
from particles import Particles
from walls import apply_left_wall, apply_right_wall
from collisions import collision_step
from mixing import apply_mixing
from observables import ObservableRecorder
from diagnostics import check_domain, check_energy


def run(cfg: dict, verbose: bool = True) -> dict:
    rng = np.random.default_rng(cfg["seed"])

    dt          = cfg["dt"]
    t_end       = cfg["t_end"]
    out_every   = cfg["output_every"]
    L           = cfg["L"]

    particles = Particles(cfg["N"], L, cfg["v_init_scale"], rng)
    recorder  = ObservableRecorder()

    stats = {
        "n_bulk_collisions_total": 0,
        "n_right_wall_total": 0,
        "n_bulk_collisions": 0,
    }

    t      = 0.0
    step   = 0
    prev_E = 0.5 * np.mean(particles.v ** 2)

    n_steps = int(t_end / dt)

    for step in range(n_steps):
        t = step * dt

        # 1. Advect
        particles.advect(dt)

        # 2. Left wall hits (x <= 0)
        left_hits = particles.x <= 0.0
        if left_hits.any():
            apply_left_wall(particles, left_hits)

        # 3. Right wall hits (x >= L)
        right_hits = particles.x >= L
        n_right = right_hits.sum()
        if n_right > 0:
            apply_right_wall(particles, right_hits, t, cfg, rng)
            stats["n_right_wall_total"] += n_right

        # 4. Update mixing states
        apply_mixing(particles, cfg)

        # 5. Bulk collisions
        n_coll = collision_step(particles, cfg, dt, rng, stats)
        stats["n_bulk_collisions_total"] += n_coll

        # 6. Record observables
        if step % out_every == 0:
            recorder.record(t, particles, stats)

            if verbose and step % (out_every * 10) == 0:
                E     = 0.5 * np.mean(particles.v ** 2)
                pmix  = particles.sigma.mean()
                print(f"  t={t:7.3f}  E={E:.4f}  p_mix={pmix:.3f}"
                      f"  collisions={stats['n_bulk_collisions_total']}")

            # Diagnostics
            warns = check_domain(particles, cfg)
            warns += check_energy(particles, prev_E)
            for w in warns:
                print(w)
            prev_E = 0.5 * np.mean(particles.v ** 2)

    data = recorder.to_arrays()
    data["particles_final"] = {
        "x": particles.x.copy(),
        "v": particles.v.copy(),
        "sigma": particles.sigma.copy(),
        "age": particles.age.copy(),
    }
    data["cfg"] = cfg
    return data
