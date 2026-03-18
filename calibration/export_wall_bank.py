"""Export a collision-free Fermi-map trajectory bank with proxy labels.

This is the real wall-side data source for the survival-learning pipeline.
It runs the exact discrete Fermi map, computes the energy-bin chaos mask from
phase entropy + phase mixing lag, then stores the full trajectory bank:

    u_n, psi_n, M_n

along with the binwise diagnostics needed by downstream survival learning.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from chaos_mask import build_chaotic_mask
from first_passage import build_proxy_labels
from mixing_diagnostics import compute_phase_entropy
from phase_mixing import compute_phase_acf, estimate_tau_mix_from_acf
from trajectories import load_cal_config, make_energy_bins, generate_trajectories


def export_wall_bank(cfg: dict, out_dir: Path, save_time: bool = False) -> Path:
    """Generate and save a self-contained real Fermi-map trajectory bank."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wall_bank.npz"

    print("[1/5] Generating collision-free Fermi trajectories ...")
    data = generate_trajectories(cfg)
    u_traj = data["u_traj"]
    psi_traj = data["psi_traj"]

    print("[2/5] Building energy bins ...")
    u_edges, u_centers = make_energy_bins(cfg["u_min"], cfg["u_max"], cfg["n_u_bins"])
    u_bin_idx = np.clip(np.digitize(u_traj, u_edges) - 1, 0, len(u_centers) - 1).astype(np.int32)

    print("[3/5] Computing phase-mixing diagnostics ...")
    C_acf, counts_acf = compute_phase_acf(
        u_traj,
        psi_traj,
        u_edges,
        max_lag=int(cfg["acf_max_lag"]),
    )
    tau_lag = estimate_tau_mix_from_acf(
        C_acf,
        counts_acf,
        threshold=float(cfg["acf_threshold"]),
        n_min=int(cfg["n_km_samples_min"]),
    )
    entropy = compute_phase_entropy(
        u_traj,
        psi_traj,
        u_edges,
        n_psi_bins=int(cfg.get("n_psi_bins_uniformity", 36)),
        n_min=int(cfg["n_km_samples_min"]),
    )

    print("[4/5] Building chaos mask and proxy labels ...")
    chaos_mask = build_chaotic_mask(
        entropy["entropy_norm"],
        tau_lag,
        entropy_threshold=float(cfg.get("entropy_threshold", 0.8)),
    )
    proxy_labels = build_proxy_labels(u_traj, u_edges, chaos_mask)

    print("[5/5] Saving trajectory bank ...")
    save_kwargs = {
        "u_traj": u_traj,
        "psi_traj": psi_traj,
        "u_bin_idx": u_bin_idx,
        "proxy_labels": proxy_labels.astype(np.int8),
        "u_edges": u_edges,
        "u_centers": u_centers,
        "chaos_mask": chaos_mask.astype(np.int8),
        "tau_lag": tau_lag,
        "entropy": entropy["entropy"],
        "entropy_norm": entropy["entropy_norm"],
        "entropy_counts": entropy["counts"],
        "counts_acf": counts_acf,
        "acf_max_lag": np.array([int(cfg["acf_max_lag"])], dtype=np.int32),
        "acf_threshold": np.array([float(cfg["acf_threshold"])], dtype=np.float64),
        "entropy_threshold": np.array([float(cfg.get("entropy_threshold", 0.8))], dtype=np.float64),
        "A": np.array([float(cfg["A"])], dtype=np.float64),
        "omega": np.array([float(cfg["omega"])], dtype=np.float64),
        "L": np.array([float(cfg["L"])], dtype=np.float64),
        "seed": np.array([int(cfg["seed"])], dtype=np.int64),
    }
    if save_time:
        save_kwargs["t_traj"] = data["t_traj"]

    np.savez_compressed(out_path, **save_kwargs)

    proxy_frac = float(proxy_labels.mean())
    n_chaotic = int(chaos_mask.sum())
    print(f"  Saved: {out_path}")
    print(f"  Trajectories   : {u_traj.shape[0]} particles x {u_traj.shape[1] - 1} hits")
    print(f"  Chaos mask     : {n_chaotic}/{len(chaos_mask)} bins")
    print(f"  Proxy fraction : {proxy_frac:.1%} of bounces")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Export real Fermi-map trajectory bank with proxy labels")
    ap.add_argument("--config", default=None, help="Optional calibration override YAML")
    ap.add_argument("--out-dir", default="results/real_wall_bank")
    ap.add_argument("--save-time", action="store_true", help="Also store t_traj in the bank")
    args = ap.parse_args()

    cfg = load_cal_config(args.config)
    export_wall_bank(cfg, Path(args.out_dir), save_time=args.save_time)


if __name__ == "__main__":
    main()