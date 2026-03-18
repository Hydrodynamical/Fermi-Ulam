"""
build_survival_dataset.py — Convert segments to row-wise hazard learning data.

For each segment, emit one row per bounce before entry or reset:

    X_full  : [u, log(1+a)]   — features for full model
    X_base  : [u]             — features for energy-only baseline
    y       : 1 iff entry occurs at the *next* bounce, 0 otherwise
    seg_id  : segment index (for survival reconstruction)
    age     : current bounce index a

Usage
-----
python toy/build_survival_dataset.py \
    --data results/toy/segments.npz \
    --out-dir results/toy
"""

import argparse
import pathlib

import numpy as np


def build_dataset(segments: dict) -> dict:
    """
    Convert segment arrays into row-wise hazard data.

    Parameters
    ----------
    segments : dict from toy_generator.load_segments()
        Keys: u0 (N,), z (N,), T (N,), L (N,)

    Returns
    -------
    dict with keys
        X_full  : (R, 2)  [u, log(1+a)]
        X_base  : (R, 1)  [u]
        y       : (R,)    bool — entry at next bounce?
        seg_id  : (R,)    int32 — segment index
        age     : (R,)    int32 — current bounce a
    """
    u0 = segments["u0"]   # (N,)
    T  = segments["T"]    # (N,) entry age; np.inf if censored
    L  = segments["L"]    # (N,) segment length (number of bounces observed)
    N  = len(u0)

    rows_u        = []
    rows_log1pa   = []
    rows_y        = []
    rows_seg_id   = []
    rows_age      = []

    for i in range(N):
        u    = float(u0[i])
        t    = T[i]        # entry age or inf
        leng = int(L[i])   # segment length (= entry age for entered, = reset age for censored)

        # For entered segments: emit rows for a = 0, ..., T  (y=1 only at a=T)
        # For censored segments: emit rows for a = 0, ..., L-1 (all y=0)
        entered = np.isfinite(t)
        n_rows  = int(t) + 1 if entered else leng

        for a in range(n_rows):
            rows_u.append(u)
            rows_log1pa.append(np.log1p(a))
            rows_y.append(1 if (entered and a == int(t)) else 0)
            rows_seg_id.append(i)
            rows_age.append(a)

    u_arr      = np.array(rows_u,      dtype=np.float64)
    log1pa_arr = np.array(rows_log1pa, dtype=np.float64)
    y_arr      = np.array(rows_y,      dtype=np.int8)
    seg_arr    = np.array(rows_seg_id, dtype=np.int32)
    age_arr    = np.array(rows_age,    dtype=np.int32)

    X_full = np.stack([u_arr, log1pa_arr], axis=1)
    X_base = u_arr.reshape(-1, 1)

    return {
        "X_full": X_full,
        "X_base": X_base,
        "y":      y_arr,
        "seg_id": seg_arr,
        "age":    age_arr,
    }


def _print_summary(ds: dict) -> None:
    R    = len(ds["y"])
    pos  = int(ds["y"].sum())
    u    = ds["X_full"][:, 0]
    print(f"\n  Total rows      : {R:,}")
    print(f"  Positive rate   : {pos/R:.4f}  ({pos:,} entries)")
    print(f"  u coverage      : [{u.min():.3f}, {u.max():.3f}]")
    print(f"  Age range       : [0, {int(ds['age'].max())}]")
    print(f"  Unique segments : {len(np.unique(ds['seg_id'])):,}")


def main():
    ap = argparse.ArgumentParser(description="Build hazard dataset from toy segments")
    ap.add_argument("--data",    required=True, help="Path to segments.npz")
    ap.add_argument("--out-dir", default="results/toy")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/2] Loading segments from {args.data} ...")
    raw = np.load(args.data, allow_pickle=False)
    segments = {k: raw[k] for k in raw.files}
    print(f"  {len(segments['u0']):,} segments loaded")

    print("[2/2] Building dataset ...")
    ds = build_dataset(segments)
    _print_summary(ds)

    out_path = out_dir / "dataset.npz"
    np.savez_compressed(out_path, **ds)
    print(f"\n  Saved dataset to {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
