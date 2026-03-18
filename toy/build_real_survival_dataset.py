"""Build a row-wise hazard dataset from a real Fermi-map trajectory bank.

This mirrors toy/build_survival_dataset.py but uses collision-free Fermi-map
trajectories labeled by the binwise Markov proxy. Each row corresponds to one
non-proxy bounce with target y = 1 if proxy entry occurs at the next bounce.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from calibration.survival_dataset import build_labels_from_mask, build_rowwise_hazard_dataset, summarize_rowwise_dataset


def _load_labels(raw, label_source: str, mask_path: str | None) -> tuple[np.ndarray, dict]:
    metadata = {"label_source": np.array([label_source])}

    if label_source == "proxy":
        return raw["proxy_labels"].astype(bool), metadata

    if mask_path is None:
        raise ValueError("--mask-path is required when --label-source=core-mask")

    mask_npz = np.load(mask_path, allow_pickle=False)
    if "core_mask" in mask_npz.files:
        mask = mask_npz["core_mask"].astype(bool)
    elif "mask" in mask_npz.files:
        mask = mask_npz["mask"].astype(bool)
    else:
        raise ValueError(f"No core_mask or mask array found in {mask_path}")

    if "u_bin_idx" in raw.files:
        labels = build_labels_from_mask(raw["u_bin_idx"], mask)
    else:
        u_edges = raw["u_edges"]
        u_traj = raw["u_traj"]
        u_bin_idx = np.clip(np.digitize(u_traj, u_edges) - 1, 0, len(mask) - 1)
        labels = build_labels_from_mask(u_bin_idx, mask)

    metadata["core_mask"] = mask.astype(np.int8)
    metadata["mask_path"] = np.array([str(mask_path)])
    return labels, metadata


def _print_summary(dataset: dict) -> None:
    summary = summarize_rowwise_dataset(dataset)
    print(f"\n  Total rows      : {summary['n_rows']:,}")
    print(f"  Positive rate   : {summary['positive_rate']:.4f}  ({summary['n_positive']:,} entries)")
    print(f"  u coverage      : [{summary['u_min']:.3f}, {summary['u_max']:.3f}]")
    print(f"  Age range       : [0, {summary['age_max']}]")
    print(f"  Unique segments : {summary['n_segments']:,}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build hazard dataset from real Fermi-map trajectories")
    ap.add_argument("--data", required=True, help="Path to wall_bank.npz")
    ap.add_argument("--out-dir", default="results/real_wall_hazard")
    ap.add_argument(
        "--label-source",
        choices=["proxy", "core-mask"],
        default="proxy",
        help="Which event definition to use when building labels",
    )
    ap.add_argument(
        "--mask-path",
        default=None,
        help="Path to diagnostics/core_mask.npz when --label-source=core-mask",
    )
    ap.add_argument(
        "--max-age",
        type=int,
        default=None,
        help="Optional maximum age to retain in the row-wise dataset",
    )
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/2] Loading wall bank from {args.data} ...")
    raw = np.load(args.data, allow_pickle=False)
    u_traj = raw["u_traj"]
    labels, label_meta = _load_labels(raw, args.label_source, args.mask_path)
    u_bin_idx = raw["u_bin_idx"] if "u_bin_idx" in raw.files else None
    print(f"  {u_traj.shape[0]:,} particles x {u_traj.shape[1] - 1:,} hits loaded")
    print(f"  Label source     : {args.label_source}")

    print("[2/2] Building row-wise hazard dataset ...")
    dataset = build_rowwise_hazard_dataset(
        u_traj=u_traj,
        labels=labels,
        max_age=args.max_age,
        u_bin_idx=u_bin_idx,
    )
    _print_summary(dataset)

    save_kwargs = dict(dataset)
    save_kwargs.update(label_meta)
    for key in ["u_edges", "u_centers", "chaos_mask", "tau_lag", "entropy_norm"]:
        if key in raw.files:
            save_kwargs[key] = raw[key]

    out_path = out_dir / "dataset.npz"
    np.savez_compressed(out_path, **save_kwargs)
    print(f"\n  Saved dataset to {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()