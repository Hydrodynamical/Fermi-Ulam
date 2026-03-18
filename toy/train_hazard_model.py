"""Train MLP hazard models on toy or real row-wise hazard datasets.

Two models are trained and compared:

    model_full  : features [u, log(1+a)]  — full age-structured model
    model_base  : features [u]            — energy-only baseline

For real datasets the split is performed by seg_id, not by row, so the same
non-proxy excursion cannot leak into both train and test sets.
"""

import argparse
import pathlib
import sys

import joblib
import numpy as np

_REPO = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from calibration.hazard_models import PolynomialLogisticHazardModel


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_model() -> PolynomialLogisticHazardModel:
    """Return a fresh hazard classifier with a sklearn-like API."""
    return PolynomialLogisticHazardModel(
        degree=3,
        learning_rate=0.03,
        max_epochs=60,
        batch_size=8192,
        l2=1e-4,
        random_state=0,
        validation_fraction=0.1,
        n_iter_no_change=8,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_test:  np.ndarray,
    name:    str,
) -> tuple:
    """
    Fit a model on (X_train, y_train), evaluate on (X_test, y_test).

    Returns
    -------
    model     : fitted Pipeline
    metrics   : dict with log_loss, brier_score
    y_pred    : (n_test,) predicted probabilities on test set
    """
    print(f"  Training {name} ...")
    model = make_model()
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_test)[:, 1]
    eps = 1e-8
    ll = float(-np.mean(y_test * np.log(y_pred + eps) + (1.0 - y_test) * np.log(1.0 - y_pred + eps)))
    bs = float(np.mean((y_pred - y_test) ** 2))
    n_iter = model.n_iter_

    print(f"    Iterations    : {n_iter}")
    print(f"    Log-loss      : {ll:.5f}")
    print(f"    Brier score   : {bs:.5f}")

    metrics = {"log_loss": ll, "brier_score": bs, "n_iter": n_iter}
    return model, metrics, y_pred


def _print_calibration(y_test: np.ndarray, y_pred: np.ndarray, name: str) -> None:
    """Print mean-calibration error in 10 equal-frequency bins."""
    ece = compute_ece(y_test, y_pred)
    print(f"    ECE (10 bins) : {ece:.4f}  [{name}]")


def compute_ece(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    """Return the mean absolute calibration gap in quantile bins."""
    order = np.argsort(y_pred)
    bins = np.array_split(order, n_bins)
    gaps = []
    for idx in bins:
        if len(idx) == 0:
            continue
        gaps.append(abs(float(y_true[idx].mean()) - float(y_pred[idx].mean())))
    return float(np.mean(gaps)) if gaps else float("nan")


def split_indices(
    y: np.ndarray,
    seg_id: np.ndarray | None,
    test_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return train/test row indices, grouped by segment when available."""
    if seg_id is None:
        rng = np.random.default_rng(seed)
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)
        n_pos_test = int(round(test_frac * len(pos_idx)))
        n_neg_test = int(round(test_frac * len(neg_idx)))
        te_idx = np.concatenate([pos_idx[:n_pos_test], neg_idx[:n_neg_test]])
        tr_idx = np.concatenate([pos_idx[n_pos_test:], neg_idx[n_neg_test:]])
        rng.shuffle(tr_idx)
        rng.shuffle(te_idx)
        return tr_idx, te_idx, "row-stratified"

    rng = np.random.default_rng(seed)
    groups, inverse = np.unique(seg_id, return_inverse=True)
    group_counts = np.bincount(inverse)
    perm = rng.permutation(len(groups))
    target_rows = int(round(test_frac * len(seg_id)))
    picked = []
    running = 0
    for group_pos in perm:
        picked.append(group_pos)
        running += int(group_counts[group_pos])
        if running >= target_rows:
            break

    test_group_ids = groups[np.array(picked, dtype=np.int32)]
    test_mask = np.isin(seg_id, test_group_ids)
    te_idx = np.where(test_mask)[0]
    tr_idx = np.where(~test_mask)[0]
    return tr_idx, te_idx, "group-by-segment"


def maybe_cap_rows(
    idx: np.ndarray,
    y: np.ndarray,
    max_rows: int | None,
    seed: int,
) -> np.ndarray:
    """Optionally downsample a split while preserving class balance when possible."""
    if max_rows is None or len(idx) <= max_rows:
        return idx

    rng = np.random.default_rng(seed)
    pos_idx = idx[y[idx] == 1]
    neg_idx = idx[y[idx] == 0]
    target_pos = int(round(max_rows * len(pos_idx) / max(len(idx), 1)))
    target_pos = min(max(target_pos, 1 if len(pos_idx) else 0), len(pos_idx))
    target_neg = min(max_rows - target_pos, len(neg_idx))

    if target_pos + target_neg < max_rows and len(pos_idx) > target_pos:
        extra = min(max_rows - (target_pos + target_neg), len(pos_idx) - target_pos)
        target_pos += extra
    if target_pos + target_neg < max_rows and len(neg_idx) > target_neg:
        extra = min(max_rows - (target_pos + target_neg), len(neg_idx) - target_neg)
        target_neg += extra

    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    keep = np.concatenate([pos_idx[:target_pos], neg_idx[:target_neg]])
    rng.shuffle(keep)
    return keep


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Train hazard models on toy or real datasets")
    ap.add_argument("--data",      required=True, help="Path to dataset.npz")
    ap.add_argument("--out-dir",   default="results/toy")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed",      type=int,   default=0)
    ap.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional cap on training rows after the split",
    )
    ap.add_argument(
        "--max-test-rows",
        type=int,
        default=None,
        help="Optional cap on test rows after the split",
    )
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Loading dataset from {args.data} ...")
    raw = np.load(args.data, allow_pickle=False)
    X_full = raw["X_full"].astype(np.float32)  # (R, 2) [u, log(1+a)]
    X_base = raw["X_base"].astype(np.float32)  # (R, 1) [u]
    y      = raw["y"].astype(np.int32)
    seg_id = raw["seg_id"].astype(np.int32) if "seg_id" in raw.files else None
    R      = len(y)
    print(f"  {R:,} rows  |  positive rate {y.mean():.4f}")
    if seg_id is not None:
        print(f"  {len(np.unique(seg_id)):,} unique segments available")

    print(f"[2/3] Splitting {1-args.test_frac:.0%} train / {args.test_frac:.0%} test ...")
    tr_idx, te_idx, split_mode = split_indices(y, seg_id, args.test_frac, args.seed)
    tr_idx = maybe_cap_rows(tr_idx, y, args.max_train_rows, args.seed)
    te_idx = maybe_cap_rows(te_idx, y, args.max_test_rows, args.seed + 1)
    y_train, y_test = y[tr_idx], y[te_idx]
    print(f"  Split mode: {split_mode}")
    print(f"  Train: {len(tr_idx):,} rows  |  Test: {len(te_idx):,} rows")
    print(f"  Train positive rate: {y_train.mean():.4f}")
    print(f"  Test positive rate : {y_test.mean():.4f}")
    if seg_id is not None:
        print(f"  Train segments: {len(np.unique(seg_id[tr_idx])):,}")
        print(f"  Test segments : {len(np.unique(seg_id[te_idx])):,}")

    print("[3/3] Training models ...")

    model_full, metrics_full, pred_full = train_and_evaluate(
        X_full[tr_idx], X_full[te_idx], y_train, y_test, "full [u, log(1+a)]"
    )
    _print_calibration(y_test, pred_full, "full")
    metrics_full["ece"] = compute_ece(y_test, pred_full)

    model_base, metrics_base, pred_base = train_and_evaluate(
        X_base[tr_idx], X_base[te_idx], y_train, y_test, "base [u]"
    )
    _print_calibration(y_test, pred_base, "base")
    metrics_base["ece"] = compute_ece(y_test, pred_base)

    print("\n  --- Comparison ---")
    print(f"  {'Model':<12}  {'log-loss':>10}  {'Brier':>10}  {'ECE':>10}")
    print(f"  {'full':<12}  {metrics_full['log_loss']:>10.5f}  "
          f"{metrics_full['brier_score']:>10.5f}  {metrics_full['ece']:>10.5f}")
    print(f"  {'base':<12}  {metrics_base['log_loss']:>10.5f}  "
          f"{metrics_base['brier_score']:>10.5f}  {metrics_base['ece']:>10.5f}")

    # Save models
    full_path = out_dir / "model_full.pkl"
    base_path = out_dir / "model_base.pkl"
    joblib.dump(model_full, full_path)
    joblib.dump(model_base, base_path)
    print(f"\n  Saved model_full to {full_path}")
    print(f"  Saved model_base to {base_path}")

    # Save metrics
    np.savez(
        out_dir / "training_metrics.npz",
        split_mode=split_mode,
        train_rows=len(tr_idx),
        test_rows=len(te_idx),
        train_positive_rate=y_train.mean(),
        test_positive_rate=y_test.mean(),
        train_segments=(len(np.unique(seg_id[tr_idx])) if seg_id is not None else -1),
        test_segments=(len(np.unique(seg_id[te_idx])) if seg_id is not None else -1),
        log_loss_full=metrics_full["log_loss"],
        brier_full=metrics_full["brier_score"],
        ece_full=metrics_full["ece"],
        log_loss_base=metrics_base["log_loss"],
        brier_base=metrics_base["brier_score"],
        ece_base=metrics_base["ece"],
    )
    print("Done.")


if __name__ == "__main__":
    main()
