"""Export a dedicated stratified (u0, psi0) sweep for estimating u_s.

This sweep is not sampled from the long-time dynamical measure. Instead it uses
uniform phase seeds at fixed initial energies to probe whether the deterministic
collision-free Fermi map appears globally stochastic below a threshold u_s.

The output stores per-seed summary diagnostics and optionally a thinned tail
Poincare sample for later plotting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from map import step
from trajectories import load_cal_config

TWO_PI = 2.0 * np.pi


def make_u0_grid(u_min: float, u_max: float, n_u: int) -> np.ndarray:
    """Return a log-spaced initial-energy grid."""
    return np.logspace(np.log10(u_min), np.log10(u_max), int(n_u))


def make_psi0_grid(n_psi: int) -> np.ndarray:
    """Return a uniform phase grid in [0, 2*pi)."""
    return np.linspace(0.0, TWO_PI, int(n_psi), endpoint=False)


def _phase_entropy_from_hist(hist: np.ndarray) -> np.ndarray:
    total = hist.sum(axis=1, keepdims=True)
    p = np.zeros_like(hist, dtype=np.float64)
    np.divide(hist, np.maximum(total, 1), out=p, where=total > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log(p), 0.0)
    H = -np.sum(p * logp, axis=1)
    norm = np.log(hist.shape[1]) if hist.shape[1] > 1 else 1.0
    return np.where(total[:, 0] > 0, H / norm, np.nan)


def run_us_sweep(cfg: dict, out_dir: Path, thin_stride: int = 50, n_phase_bins: int = 64) -> Path:
    """Run the deterministic map on a stratified (u0, psi0) grid and save diagnostics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "us_sweep.npz"

    u0_grid = make_u0_grid(cfg["u_min"], cfg["u_max"], cfg["us_n_u"])
    psi0_grid = make_psi0_grid(cfg["us_n_psi"])
    n_u = len(u0_grid)
    n_psi = len(psi0_grid)
    n_hits = int(cfg["us_n_hits"])
    burn_in = int(cfg["us_burn_in"])
    n_tail = max(n_hits - burn_in, 0)
    n_keep = 0 if thin_stride <= 0 or n_tail == 0 else (n_tail + thin_stride - 1) // thin_stride

    phase_hist = np.zeros((n_u, n_psi, n_phase_bins), dtype=np.int32)
    visited_fraction = np.full((n_u, n_psi), np.nan)
    phase_entropy = np.full((n_u, n_psi), np.nan)
    mean_u_tail = np.full((n_u, n_psi), np.nan)
    std_u_tail = np.full((n_u, n_psi), np.nan)
    lag1_acf_cospsi = np.full((n_u, n_psi), np.nan)
    tail_count = np.zeros((n_u, n_psi), dtype=np.int32)

    u_tail_thin = np.full((n_u, n_psi, n_keep), np.nan, dtype=np.float32)
    psi_tail_thin = np.full((n_u, n_psi, n_keep), np.nan, dtype=np.float32)

    print(
        f"[u_s sweep] Running {n_u} energy levels x {n_psi} phase seeds x {n_hits} hits "
        f"(burn-in {burn_in}, thin_stride {thin_stride})"
    )

    for iu, u0 in enumerate(u0_grid):
        u = np.full(n_psi, u0, dtype=np.float64)
        psi = psi0_grid.copy()

        sum_u = np.zeros(n_psi, dtype=np.float64)
        sum_u2 = np.zeros(n_psi, dtype=np.float64)
        sum_x = np.zeros(n_psi, dtype=np.float64)
        sum_x2 = np.zeros(n_psi, dtype=np.float64)
        sum_xx1 = np.zeros(n_psi, dtype=np.float64)
        pair_count = np.zeros(n_psi, dtype=np.int32)
        prev_x = np.zeros(n_psi, dtype=np.float64)
        have_prev = np.zeros(n_psi, dtype=bool)
        keep_idx = np.zeros(n_psi, dtype=np.int32)

        for hit in range(n_hits):
            u, psi, _ = step(u, psi, cfg["A"], cfg["omega"], cfg["L"])
            if hit < burn_in:
                continue

            tail_idx = hit - burn_in
            bin_idx = np.clip((psi / TWO_PI * n_phase_bins).astype(np.int32), 0, n_phase_bins - 1)
            np.add.at(phase_hist[iu], (np.arange(n_psi), bin_idx), 1)

            sum_u += u
            sum_u2 += u * u
            x = np.cos(psi)
            sum_x += x
            sum_x2 += x * x

            active_prev = have_prev
            if np.any(active_prev):
                sum_xx1[active_prev] += prev_x[active_prev] * x[active_prev]
                pair_count[active_prev] += 1
            prev_x = x
            have_prev[:] = True
            tail_count[iu] += 1

            if n_keep > 0 and (tail_idx % thin_stride == 0):
                idx = keep_idx.copy()
                valid = idx < n_keep
                if np.any(valid):
                    u_tail_thin[iu, np.where(valid)[0], idx[valid]] = u[valid].astype(np.float32)
                    psi_tail_thin[iu, np.where(valid)[0], idx[valid]] = psi[valid].astype(np.float32)
                keep_idx += 1

        counts = tail_count[iu].astype(np.float64)
        valid_counts = counts > 0
        mean_u_tail[iu, valid_counts] = sum_u[valid_counts] / counts[valid_counts]
        var_u = np.full(n_psi, np.nan, dtype=np.float64)
        var_u[valid_counts] = sum_u2[valid_counts] / counts[valid_counts] - mean_u_tail[iu, valid_counts] ** 2
        std_u_tail[iu, valid_counts] = np.sqrt(np.maximum(var_u[valid_counts], 0.0))

        phase_entropy[iu] = _phase_entropy_from_hist(phase_hist[iu])
        visited_fraction[iu] = np.count_nonzero(phase_hist[iu] > 0, axis=1) / float(n_phase_bins)

        mu_x = np.full(n_psi, np.nan, dtype=np.float64)
        var_x = np.full(n_psi, np.nan, dtype=np.float64)
        mu_x[valid_counts] = sum_x[valid_counts] / counts[valid_counts]
        var_x[valid_counts] = sum_x2[valid_counts] / counts[valid_counts] - mu_x[valid_counts] ** 2
        valid_acf = (pair_count > 0) & np.isfinite(var_x) & (var_x > 1e-12)
        lag1_acf_cospsi[iu, valid_acf] = (
            sum_xx1[valid_acf] / pair_count[valid_acf] - mu_x[valid_acf] ** 2
        ) / var_x[valid_acf]

        if (iu + 1) % max(1, n_u // 10) == 0 or iu == n_u - 1:
            print(f"  completed {iu + 1}/{n_u} energy levels")

    np.savez_compressed(
        out_path,
        u0_grid=u0_grid,
        psi0_grid=psi0_grid,
        phase_hist_tail=phase_hist,
        visited_fraction=visited_fraction,
        phase_entropy=phase_entropy,
        mean_u_tail=mean_u_tail,
        std_u_tail=std_u_tail,
        lag1_acf_cospsi=lag1_acf_cospsi,
        tail_count=tail_count,
        u_tail_thin=u_tail_thin,
        psi_tail_thin=psi_tail_thin,
        thin_stride=np.array([thin_stride], dtype=np.int32),
        n_phase_bins=np.array([n_phase_bins], dtype=np.int32),
        us_n_hits=np.array([n_hits], dtype=np.int32),
        us_burn_in=np.array([burn_in], dtype=np.int32),
        A=np.array([float(cfg["A"])], dtype=np.float64),
        omega=np.array([float(cfg["omega"])], dtype=np.float64),
        L=np.array([float(cfg["L"])], dtype=np.float64),
    )
    print(f"[u_s sweep] Saved {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a stratified (u0, psi0) sweep for estimating u_s")
    ap.add_argument("--config", default=None, help="Optional YAML override for base/calibration config")
    ap.add_argument("--out-dir", default="results/us_sweep")
    ap.add_argument("--u-min", type=float, default=None)
    ap.add_argument("--u-max", type=float, default=None)
    ap.add_argument("--n-u", type=int, default=80)
    ap.add_argument("--n-psi", type=int, default=192)
    ap.add_argument("--n-hits", type=int, default=20000)
    ap.add_argument("--burn-in", type=int, default=5000)
    ap.add_argument("--thin-stride", type=int, default=50)
    ap.add_argument("--n-phase-bins", type=int, default=64)
    args = ap.parse_args()

    cfg = load_cal_config(args.config)
    cfg["u_min"] = float(args.u_min if args.u_min is not None else cfg["u_min"])
    cfg["u_max"] = float(args.u_max if args.u_max is not None else cfg["u_max"])
    cfg["us_n_u"] = int(args.n_u)
    cfg["us_n_psi"] = int(args.n_psi)
    cfg["us_n_hits"] = int(args.n_hits)
    cfg["us_burn_in"] = int(args.burn_in)

    run_us_sweep(cfg, Path(args.out_dir), thin_stride=int(args.thin_stride), n_phase_bins=int(args.n_phase_bins))


if __name__ == "__main__":
    main()