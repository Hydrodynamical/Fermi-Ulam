"""Utilities for building row-wise hazard datasets from labeled Fermi trajectories.

The target event is first entry into the binwise Markov proxy at the next bounce.
For each non-proxy bounce n with label M_n = 0 we emit one supervised row:

    X_full = [u_n, log(1 + a_n)]
    X_base = [u_n]
    y_n    = 1 iff M_{n+1} = 1

where a_n is the age since the start of the current non-proxy stretch.
Rows are grouped by non-proxy excursion through seg_id so downstream model
evaluation can split by segment rather than by individual rows.
"""

from __future__ import annotations

import numpy as np


def compute_stretch_ages(labels: np.ndarray, active_value: bool) -> np.ndarray:
    """Return age within consecutive stretches where labels == active_value."""
    if labels.ndim != 2:
        raise ValueError("labels must be a 2D array")

    ages = np.full(labels.shape, -1, dtype=np.int32)

    for i in range(labels.shape[0]):
        age = 0
        in_stretch = False
        for n in range(labels.shape[1]):
            if bool(labels[i, n]) != bool(active_value):
                age = 0
                in_stretch = False
                continue

            if not in_stretch:
                in_stretch = True
                age = 0

            ages[i, n] = age
            age += 1

    return ages


def compute_nonproxy_ages(labels: np.ndarray) -> np.ndarray:
    """Return age-since-excursion-start for each non-proxy bounce."""
    return compute_stretch_ages(labels, active_value=False)


def build_labels_from_mask(u_bin_idx: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return per-bounce labels from a binwise boolean mask."""
    if u_bin_idx.ndim != 2:
        raise ValueError("u_bin_idx must be a 2D array")
    if mask.ndim != 1:
        raise ValueError("mask must be a 1D array")
    return mask[u_bin_idx]


def build_rowwise_hazard_dataset(
    u_traj: np.ndarray,
    labels: np.ndarray,
    max_age: int | None = None,
    u_bin_idx: np.ndarray | None = None,
) -> dict:
    """Convert long labeled trajectories into row-wise hazard learning data."""
    if u_traj.shape != labels.shape:
        raise ValueError("u_traj and labels must have the same shape")
    if u_traj.ndim != 2:
        raise ValueError("u_traj and labels must be 2D arrays")
    if u_traj.shape[1] < 2:
        raise ValueError("need at least two hits to define next-bounce labels")
    if max_age is not None and max_age < 0:
        raise ValueError("max_age must be non-negative or None")
    if u_bin_idx is not None and u_bin_idx.shape != u_traj.shape:
        raise ValueError("u_bin_idx must match u_traj shape when provided")

    rows_u = []
    rows_log1pa = []
    rows_y = []
    rows_seg_id = []
    rows_age = []
    rows_particle_id = []
    rows_hit_idx = []
    rows_bin_idx = []

    seg_id = 0
    n_particles, n_hits1 = u_traj.shape
    last_trainable_hit = n_hits1 - 2

    for particle_id in range(n_particles):
        hit_idx = 0
        while hit_idx <= last_trainable_hit:
            if labels[particle_id, hit_idx]:
                hit_idx += 1
                continue

            stretch_start = hit_idx
            while hit_idx < n_hits1 and not labels[particle_id, hit_idx]:
                hit_idx += 1
            stretch_stop = hit_idx

            observed_stop = min(stretch_stop - 1, last_trainable_hit)
            if max_age is not None:
                observed_stop = min(observed_stop, stretch_start + max_age)
            if observed_stop < stretch_start:
                seg_id += 1
                continue

            for n in range(stretch_start, observed_stop + 1):
                age = n - stretch_start
                rows_u.append(float(u_traj[particle_id, n]))
                rows_log1pa.append(np.log1p(age))
                rows_y.append(1 if labels[particle_id, n + 1] else 0)
                rows_seg_id.append(seg_id)
                rows_age.append(age)
                rows_particle_id.append(particle_id)
                rows_hit_idx.append(n)
                if u_bin_idx is None:
                    rows_bin_idx.append(-1)
                else:
                    rows_bin_idx.append(int(u_bin_idx[particle_id, n]))

            seg_id += 1

    u_arr = np.array(rows_u, dtype=np.float64)
    log1pa_arr = np.array(rows_log1pa, dtype=np.float64)

    return {
        "X_full": np.stack([u_arr, log1pa_arr], axis=1),
        "X_base": u_arr.reshape(-1, 1),
        "y": np.array(rows_y, dtype=np.int8),
        "seg_id": np.array(rows_seg_id, dtype=np.int32),
        "age": np.array(rows_age, dtype=np.int32),
        "particle_id": np.array(rows_particle_id, dtype=np.int32),
        "hit_idx": np.array(rows_hit_idx, dtype=np.int32),
        "bin_idx": np.array(rows_bin_idx, dtype=np.int32),
    }


def summarize_rowwise_dataset(dataset: dict) -> dict:
    """Return a compact summary for CLI reporting."""
    y = dataset["y"]
    age = dataset["age"]
    u = dataset["X_full"][:, 0]
    seg_id = dataset["seg_id"]

    return {
        "n_rows": int(len(y)),
        "positive_rate": float(y.mean()) if len(y) else float("nan"),
        "n_positive": int(y.sum()),
        "n_segments": int(len(np.unique(seg_id))) if len(seg_id) else 0,
        "u_min": float(u.min()) if len(u) else float("nan"),
        "u_max": float(u.max()) if len(u) else float("nan"),
        "age_max": int(age.max()) if len(age) else -1,
    }