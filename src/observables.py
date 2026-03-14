"""Record macroscopic observables at each output step."""
import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class Snapshot:
    t: float
    E: float              # mean kinetic energy per particle
    p_mix: float          # fraction of mixed particles (instantaneous chi proxy)
    v_mean: float         # mean velocity
    v_std: float          # velocity std dev
    n_bulk_collisions: int
    n_wall_hits_right: int
    chi_direct: float     # cumulative fraction of collision events where sigma=1 at collision time


@dataclass
class ObservableRecorder:
    snapshots: List[Snapshot] = field(default_factory=list)
    # Counters reset each output interval
    _n_bulk: int = 0
    _n_right_wall: int = 0

    def record(self, t: float, particles, stats: dict):
        v = particles.v
        E = 0.5 * np.mean(v ** 2)
        p_mix = particles.sigma.mean()

        n_cm = stats.get("n_coll_mixed_total", 0)
        n_cu = stats.get("n_coll_unmixed_total", 0)
        n_ct = n_cm + n_cu
        chi_direct = n_cm / n_ct if n_ct > 0 else float("nan")

        snap = Snapshot(
            t=t,
            E=E,
            p_mix=float(p_mix),
            v_mean=float(v.mean()),
            v_std=float(v.std()),
            n_bulk_collisions=stats.get("n_bulk_collisions_total", 0),
            n_wall_hits_right=stats.get("n_right_wall_total", 0),
            chi_direct=chi_direct,
        )
        self.snapshots.append(snap)

    def to_arrays(self):
        t          = np.array([s.t          for s in self.snapshots])
        E          = np.array([s.E          for s in self.snapshots])
        p_mix      = np.array([s.p_mix      for s in self.snapshots])
        v_mean     = np.array([s.v_mean     for s in self.snapshots])
        v_std      = np.array([s.v_std      for s in self.snapshots])
        chi_direct = np.array([s.chi_direct for s in self.snapshots])
        return dict(t=t, E=E, p_mix=p_mix, v_mean=v_mean, v_std=v_std,
                    chi_direct=chi_direct)
