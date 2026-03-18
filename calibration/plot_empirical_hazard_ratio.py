"""Plot empirical age dependence via the hazard ratio h(u, a) / h_bar(u).

Here h_bar(u) is the age-independent empirical baseline hazard obtained by
pooling all events and all at-risk exposures within an energy bin:

    h_bar(u_b) = sum_a n_event(b, a) / sum_a n_risk(b, a)

This diagnostic is useful for deciding whether age dependence is strong enough
to justify an age-structured model beyond the scalar baseline h(u).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from calibration.first_passage import compute_residual_waiting_times, estimate_survival_hazard
from calibration.survival_dataset import build_labels_from_mask


def load_labels(bank, label_source: str, mask_path: str | None) -> tuple[np.ndarray, np.ndarray]:
    if label_source == "proxy":
        return bank["proxy_labels"].astype(bool), bank["chaos_mask"].astype(bool)

    if mask_path is None:
        raise ValueError("--mask-path is required when --label-source=core-mask")
    mask_npz = np.load(mask_path, allow_pickle=False)
    if "core_mask" in mask_npz.files:
        mask = mask_npz["core_mask"].astype(bool)
    elif "mask" in mask_npz.files:
        mask = mask_npz["mask"].astype(bool)
    else:
        raise ValueError(f"No core_mask or mask array found in {mask_path}")
    labels = build_labels_from_mask(bank["u_bin_idx"], mask)
    return labels, mask


def compute_empirical_baseline_hazard(rwt_by_bin: list[np.ndarray], counts: np.ndarray, max_age: int, n_min: int) -> tuple[np.ndarray, np.ndarray]:
    """Return pooled age-independent hazard h_bar(u) and per-age risk counts."""
    n_bins = len(rwt_by_bin)
    baseline = np.full(n_bins, np.nan)
    risk = np.full((n_bins, max_age), np.nan)

    a_arr = np.arange(max_age + 1, dtype=np.int64)
    for b in range(n_bins):
        c = int(counts[b])
        if c < n_min:
            continue
        rwt = np.sort(rwt_by_bin[b])
        n_le = np.searchsorted(rwt, a_arr, side="right")
        n_above = (c - n_le[:-1]).astype(float)
        n_at_next = np.diff(n_le).astype(float)
        risk[b] = n_above
        denom = float(np.sum(n_above))
        if denom > 0:
            baseline[b] = float(np.sum(n_at_next) / denom)

    return baseline, risk


def make_figure(
    out_path: Path,
    u_centers: np.ndarray,
    a_arr: np.ndarray,
    h_emp: np.ndarray,
    h_base: np.ndarray,
    counts: np.ndarray,
    mask: np.ndarray,
) -> None:
    ratio = np.full_like(h_emp, np.nan, dtype=np.float64)
    base_grid = np.repeat(h_base[:, None], h_emp.shape[1], axis=1)
    valid = np.isfinite(h_emp) & np.isfinite(base_grid) & (base_grid > 0)
    ratio[valid] = h_emp[valid] / base_grid[valid]

    log_ratio = np.full_like(ratio, np.nan)
    pos = ratio > 0
    log_ratio[pos] = np.log10(ratio[pos])

    supported = np.where(~mask & np.isfinite(h_base))[0]
    if len(supported) == 0:
        supported = np.where(np.isfinite(h_base))[0]
    sel = supported[np.linspace(0, len(supported) - 1, min(5, len(supported)), dtype=int)] if len(supported) else np.array([], dtype=int)
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, max(len(sel), 1)))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.subplots_adjust(hspace=0.34, wspace=0.34)

    ax = axes[0, 0]
    im = ax.imshow(
        log_ratio,
        origin="lower",
        aspect="auto",
        extent=[a_arr[0], a_arr[-1], 0, len(u_centers) - 1],
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
    )
    tick_idx = np.linspace(0, len(u_centers) - 1, min(6, len(u_centers)), dtype=int)
    ax.set_yticks(tick_idx)
    ax.set_yticklabels([f"{u_centers[i]:.2f}" for i in tick_idx])
    ax.set_xlabel("age a [bounces]")
    ax.set_ylabel("energy u")
    ax.set_title(r"$\log_{10}(h(u,a) / \bar h(u))$")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[0, 1]
    ax.semilogx(u_centers, h_base, "o-", ms=4, lw=1.4, color="tab:blue")
    ax.set_xlabel("energy u")
    ax.set_ylabel(r"$\bar h(u)$")
    ax.set_title("Empirical age-averaged baseline hazard")

    ax = axes[1, 0]
    for color, idx in zip(colors, sel):
        ax.plot(a_arr[:-1], ratio[idx], lw=1.7, color=color, label=f"u={u_centers[idx]:.2f}")
    ax.axhline(1.0, color="k", lw=0.9, ls="--")
    ax.set_xlabel("age a [bounces]")
    ax.set_ylabel(r"$h(u,a) / \bar h(u)$")
    ax.set_ylim(0, 3.0)
    ax.set_title("Hazard ratio by representative energy")
    if len(sel):
        ax.legend(fontsize=7, loc="upper right")

    ax = axes[1, 1]
    age_dependence = np.full(len(u_centers), np.nan)
    for b in range(len(u_centers)):
        row = ratio[b]
        valid_row = row[np.isfinite(row)]
        if len(valid_row) > 1:
            age_dependence[b] = float(np.std(valid_row))
    ax.semilogx(u_centers, age_dependence, "o-", ms=4, lw=1.4, color="tab:red")
    ax.set_xlabel("energy u")
    ax.set_ylabel(r"std$_a[h(u,a) / \bar h(u)]$")
    ax.set_title("Age-dependence strength by energy")

    fig.suptitle("Empirical hazard-ratio diagnostic", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot the empirical hazard ratio h(u,a) / h_bar(u)")
    ap.add_argument("--bank", required=True, help="Path to wall_bank.npz")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--max-age", type=int, default=200)
    ap.add_argument("--n-min", type=int, default=30)
    ap.add_argument("--label-source", choices=["proxy", "core-mask"], default="proxy")
    ap.add_argument("--mask-path", default=None, help="Path to core mask npz when --label-source=core-mask")
    args = ap.parse_args()

    bank = np.load(args.bank, allow_pickle=False)
    labels, mask = load_labels(bank, args.label_source, args.mask_path)
    u_centers = bank["u_centers"]
    fp = compute_residual_waiting_times(labels, bank["u_traj"], bank["u_edges"], len(u_centers), max_age=args.max_age)
    sv = estimate_survival_hazard(fp["rwt_by_bin"], fp["counts"], args.max_age, n_min=args.n_min)
    h_base, _ = compute_empirical_baseline_hazard(fp["rwt_by_bin"], fp["counts"], args.max_age, args.n_min)

    make_figure(Path(args.out), u_centers, sv["a_arr"], sv["h"], h_base, fp["counts"], mask)
    print(f"Saved hazard-ratio diagnostic to {args.out}")


if __name__ == "__main__":
    main()