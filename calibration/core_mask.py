"""Age-conditioned core-mask estimator for real Fermi-map wall banks.

Two modes are supported:

    entry
        Large-age, out-of-proxy, entry-anisotropy statistic.
    retention
        Large-age, in-proxy, short-horizon retention statistic.

The retention mode is intended to identify the interior chaotic sea more
directly by measuring phase mixing and short-horizon persistence inside the
coarse proxy itself.
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

from calibration.survival_dataset import build_labels_from_mask, compute_nonproxy_ages, compute_stretch_ages

TWO_PI = 2.0 * np.pi


def build_age_conditioned_phase_stats(
    u_traj: np.ndarray,
    psi_traj: np.ndarray,
    u_edges: np.ndarray,
    proxy_labels: np.ndarray,
    a_star: int = 10,
    n_psi_bins: int = 32,
    min_phase_bin_count: int = 20,
) -> dict:
    """Compute age-conditioned phase entropy and event-law anisotropy by energy bin.

    Ages are measured within non-proxy stretches under the coarse proxy labels.
    The per-row event is next-bounce entry into the coarse proxy.
    """
    if u_traj.shape != psi_traj.shape or u_traj.shape != proxy_labels.shape:
        raise ValueError("u_traj, psi_traj, and proxy_labels must have matching shapes")

    n_bins = len(u_edges) - 1
    psi_edges = np.linspace(0.0, TWO_PI, n_psi_bins + 1)
    log_k = np.log(n_psi_bins)

    ages = compute_nonproxy_ages(proxy_labels)
    u_bin_idx = np.clip(np.digitize(u_traj, u_edges) - 1, 0, n_bins - 1)
    next_entry = (~proxy_labels[:, :-1]) & proxy_labels[:, 1:]

    phase_hist = np.zeros((n_bins, n_psi_bins), dtype=np.float64)
    phase_rate_sum = np.zeros((n_bins, n_psi_bins), dtype=np.float64)
    phase_rate_cnt = np.zeros((n_bins, n_psi_bins), dtype=np.int64)
    counts = np.zeros(n_bins, dtype=np.int64)
    positives = np.zeros(n_bins, dtype=np.int64)
    event_rate = np.full(n_bins, np.nan)
    phase_entropy = np.full(n_bins, np.nan)
    phase_hazard_cv = np.full(n_bins, np.nan)

    flat_sel = ages[:, :-1] >= a_star
    flat_bins = u_bin_idx[:, :-1]
    flat_psi = psi_traj[:, :-1]
    flat_y = next_entry.astype(np.float64)

    for b in range(n_bins):
        sel = flat_sel & (flat_bins == b)
        count_b = int(sel.sum())
        counts[b] = count_b
        if count_b == 0:
            continue

        psi_b = flat_psi[sel]
        y_b = flat_y[sel]
        positives[b] = int(y_b.sum())
        event_rate[b] = float(y_b.mean())

        hist_b, _ = np.histogram(psi_b, bins=psi_edges)
        phase_hist[b] = hist_b

        p = hist_b / max(hist_b.sum(), 1)
        nz = p > 0
        phase_entropy[b] = float(-np.sum(p[nz] * np.log(p[nz])) / log_k)

        psi_idx = np.clip(np.digitize(psi_b, psi_edges) - 1, 0, n_psi_bins - 1)
        for k in range(n_psi_bins):
            mask_k = psi_idx == k
            cnt_k = int(mask_k.sum())
            if cnt_k == 0:
                continue
            phase_rate_cnt[b, k] = cnt_k
            phase_rate_sum[b, k] = float(y_b[mask_k].sum())

        valid_phase = phase_rate_cnt[b] >= min_phase_bin_count
        if valid_phase.any():
            rates = phase_rate_sum[b, valid_phase] / phase_rate_cnt[b, valid_phase]
            mean_rate = float(np.mean(rates))
            if mean_rate > 0:
                phase_hazard_cv[b] = float(np.std(rates) / mean_rate)

    return {
        "mode": np.array(["entry"]),
        "ages": ages,
        "u_bin_idx": u_bin_idx,
        "next_entry": next_entry.astype(np.int8),
        "phase_hist": phase_hist,
        "phase_rate_sum": phase_rate_sum,
        "phase_rate_cnt": phase_rate_cnt,
        "counts_after_age_cut": counts,
        "positive_counts": positives,
        "event_rate": event_rate,
        "phase_entropy": phase_entropy,
        "phase_hazard_cv": phase_hazard_cv,
        "a_star": np.array([a_star], dtype=np.int32),
        "n_psi_bins": np.array([n_psi_bins], dtype=np.int32),
    }


def build_retention_phase_stats(
    u_traj: np.ndarray,
    psi_traj: np.ndarray,
    u_edges: np.ndarray,
    proxy_labels: np.ndarray,
    a_star: int = 10,
    horizon: int = 5,
    n_psi_bins: int = 32,
    min_phase_bin_count: int = 20,
) -> dict:
    """Compute large-age, in-proxy phase and retention diagnostics by energy bin."""
    if u_traj.shape != psi_traj.shape or u_traj.shape != proxy_labels.shape:
        raise ValueError("u_traj, psi_traj, and proxy_labels must have matching shapes")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    n_bins = len(u_edges) - 1
    psi_edges = np.linspace(0.0, TWO_PI, n_psi_bins + 1)
    log_k = np.log(n_psi_bins)

    ages_in_proxy = compute_stretch_ages(proxy_labels, active_value=True)
    u_bin_idx = np.clip(np.digitize(u_traj, u_edges) - 1, 0, n_bins - 1)

    valid_cols = proxy_labels.shape[1] - horizon
    if valid_cols <= 0:
        raise ValueError("trajectory is shorter than the requested retention horizon")

    future_proxy = proxy_labels[:, 1 : 1 + valid_cols].copy()
    for lag in range(2, horizon + 1):
        future_proxy &= proxy_labels[:, lag : lag + valid_cols]

    sel_base = proxy_labels[:, :valid_cols] & (ages_in_proxy[:, :valid_cols] >= a_star)
    bins_base = u_bin_idx[:, :valid_cols]
    psi_base = psi_traj[:, :valid_cols]
    retain = future_proxy.astype(np.float64)

    phase_hist = np.zeros((n_bins, n_psi_bins), dtype=np.float64)
    retention_sum = np.zeros((n_bins, n_psi_bins), dtype=np.float64)
    retention_cnt = np.zeros((n_bins, n_psi_bins), dtype=np.int64)
    counts = np.zeros(n_bins, dtype=np.int64)
    retained_counts = np.zeros(n_bins, dtype=np.int64)
    retention_mean = np.full(n_bins, np.nan)
    phase_entropy = np.full(n_bins, np.nan)
    retention_phase_cv = np.full(n_bins, np.nan)

    for b in range(n_bins):
        sel = sel_base & (bins_base == b)
        count_b = int(sel.sum())
        counts[b] = count_b
        if count_b == 0:
            continue

        psi_b = psi_base[sel]
        r_b = retain[sel]
        retained_counts[b] = int(r_b.sum())
        retention_mean[b] = float(r_b.mean())

        hist_b, _ = np.histogram(psi_b, bins=psi_edges)
        phase_hist[b] = hist_b
        p = hist_b / max(hist_b.sum(), 1)
        nz = p > 0
        phase_entropy[b] = float(-np.sum(p[nz] * np.log(p[nz])) / log_k)

        psi_idx = np.clip(np.digitize(psi_b, psi_edges) - 1, 0, n_psi_bins - 1)
        for k in range(n_psi_bins):
            mask_k = psi_idx == k
            cnt_k = int(mask_k.sum())
            if cnt_k == 0:
                continue
            retention_cnt[b, k] = cnt_k
            retention_sum[b, k] = float(r_b[mask_k].sum())

        valid_phase = retention_cnt[b] >= min_phase_bin_count
        if valid_phase.any():
            rates = retention_sum[b, valid_phase] / retention_cnt[b, valid_phase]
            mean_rate = float(np.mean(rates))
            if mean_rate > 0:
                retention_phase_cv[b] = float(np.std(rates) / mean_rate)

    return {
        "mode": np.array(["retention"]),
        "ages_in_proxy": ages_in_proxy,
        "u_bin_idx": u_bin_idx,
        "retention_event": future_proxy.astype(np.int8),
        "phase_hist": phase_hist,
        "retention_sum": retention_sum,
        "retention_cnt": retention_cnt,
        "counts_after_age_cut": counts,
        "positive_counts": retained_counts,
        "retention_mean": retention_mean,
        "phase_entropy": phase_entropy,
        "phase_hazard_cv": retention_phase_cv,
        "a_star": np.array([a_star], dtype=np.int32),
        "n_psi_bins": np.array([n_psi_bins], dtype=np.int32),
        "horizon": np.array([horizon], dtype=np.int32),
    }


def build_core_mask(
    phase_entropy: np.ndarray,
    phase_hazard_cv: np.ndarray,
    counts: np.ndarray,
    positive_counts: np.ndarray,
    entropy_threshold: float = 0.9,
    hazard_cv_threshold: float = 0.25,
    min_count: int = 500,
    min_positive: int = 25,
) -> np.ndarray:
    """Return the boolean large-age core mask from per-bin diagnostics."""
    return (
        np.isfinite(phase_entropy)
        & np.isfinite(phase_hazard_cv)
        & (phase_entropy >= entropy_threshold)
        & (phase_hazard_cv <= hazard_cv_threshold)
        & (counts >= min_count)
        & (positive_counts >= min_positive)
    )


def build_retention_core_mask(
    phase_entropy: np.ndarray,
    retention_phase_cv: np.ndarray,
    retention_mean: np.ndarray,
    counts: np.ndarray,
    entropy_threshold: float = 0.9,
    retention_cv_threshold: float = 0.25,
    retention_threshold: float = 0.8,
    min_count: int = 500,
) -> np.ndarray:
    """Return the retention-defined large-age interior core mask."""
    return (
        np.isfinite(phase_entropy)
        & np.isfinite(retention_phase_cv)
        & np.isfinite(retention_mean)
        & (phase_entropy >= entropy_threshold)
        & (retention_phase_cv <= retention_cv_threshold)
        & (retention_mean >= retention_threshold)
        & (counts >= min_count)
    )


def make_core_mask_report(
    out_path: Path,
    u_centers: np.ndarray,
    stats: dict,
    core_mask: np.ndarray,
    entropy_threshold: float,
    hazard_cv_threshold: float,
    min_count: int,
) -> None:
    """Render a 2x2 diagnostic summary for the core mask."""
    counts = stats["counts_after_age_cut"]
    phase_entropy = stats["phase_entropy"]
    phase_hazard_cv = stats["phase_hazard_cv"]
    positives = stats["positive_counts"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.32, wspace=0.30)

    ax = axes[0, 0]
    ax.semilogx(u_centers, phase_entropy, "o-", ms=4, lw=1.3, color="tab:green")
    ax.axhline(entropy_threshold, color="k", lw=1.0, ls="--")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("energy u")
    ax.set_ylabel("phase entropy")
    ax.set_title("Age-conditioned phase entropy")

    ax = axes[0, 1]
    cv_pos = np.where(phase_hazard_cv > 0, phase_hazard_cv, np.nan)
    ax.semilogx(u_centers, cv_pos, "o-", ms=4, lw=1.3, color="tab:orange")
    ax.axhline(hazard_cv_threshold, color="k", lw=1.0, ls="--")
    ax.set_xlabel("energy u")
    ax.set_ylabel("phase hazard CV")
    ax.set_title("Phase anisotropy of next-entry law")

    ax = axes[1, 0]
    ax.loglog(u_centers, np.maximum(counts, 1), "o-", ms=4, lw=1.3, color="tab:blue", label="age-cut support")
    ax.loglog(u_centers, np.maximum(positives, 1), "s--", ms=4, lw=1.1, color="tab:red", label="positive events")
    ax.axhline(min_count, color="k", lw=1.0, ls="--")
    ax.set_xlabel("energy u")
    ax.set_ylabel("count")
    ax.set_title("Support after age cut")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.semilogx(u_centers, phase_entropy, "o-", ms=4, lw=1.1, color="tab:green", alpha=0.7, label="entropy")
    ax2 = ax.twinx()
    ax2.semilogx(u_centers, cv_pos, "s--", ms=4, lw=1.1, color="tab:orange", alpha=0.7, label="hazard CV")
    ax.scatter(u_centers[core_mask], np.full(int(core_mask.sum()), 0.05), marker="|", s=180, color="black", label="core mask")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("energy u")
    ax.set_ylabel("phase entropy")
    ax2.set_ylabel("phase hazard CV")
    ax.set_title("Final core mask")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, fontsize=8, loc="upper left")

    fig.suptitle("Large-age core-mask diagnostics", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def make_retention_core_mask_report(
    out_path: Path,
    u_centers: np.ndarray,
    stats: dict,
    core_mask: np.ndarray,
    entropy_threshold: float,
    retention_cv_threshold: float,
    retention_threshold: float,
    min_count: int,
) -> None:
    """Render a 2x2 diagnostic summary for the retention-based core mask."""
    counts = stats["counts_after_age_cut"]
    phase_entropy = stats["phase_entropy"]
    retention_mean = stats["retention_mean"]
    retention_phase_cv = stats["phase_hazard_cv"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.32, wspace=0.30)

    ax = axes[0, 0]
    ax.semilogx(u_centers, phase_entropy, "o-", ms=4, lw=1.3, color="tab:green")
    ax.axhline(entropy_threshold, color="k", lw=1.0, ls="--")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("energy u")
    ax.set_ylabel("phase entropy in proxy")
    ax.set_title("Large-age in-proxy phase entropy")

    ax = axes[0, 1]
    ax.semilogx(u_centers, retention_mean, "o-", ms=4, lw=1.3, color="tab:blue")
    ax.axhline(retention_threshold, color="k", lw=1.0, ls="--")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("energy u")
    ax.set_ylabel("mean retention")
    ax.set_title("Short-horizon retention level")

    ax = axes[1, 0]
    cv_pos = np.where(retention_phase_cv > 0, retention_phase_cv, np.nan)
    ax.semilogx(u_centers, cv_pos, "o-", ms=4, lw=1.3, color="tab:orange")
    ax.axhline(retention_cv_threshold, color="k", lw=1.0, ls="--")
    ax.set_xlabel("energy u")
    ax.set_ylabel("retention phase CV")
    ax.set_title("Phase anisotropy of retention")

    ax = axes[1, 1]
    ax.loglog(u_centers, np.maximum(counts, 1), "o-", ms=4, lw=1.3, color="tab:purple", label="in-proxy age-cut support")
    ax.axhline(min_count, color="k", lw=1.0, ls="--")
    ax.scatter(u_centers[core_mask], np.full(int(core_mask.sum()), min_count), marker="|", s=180, color="black", label="core mask")
    ax.set_xlabel("energy u")
    ax.set_ylabel("count")
    ax.set_title("Support and final core mask")
    ax.legend(fontsize=8)

    fig.suptitle("Retention-based core-mask diagnostics", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def run_core_mask(bank: dict, out_dir: Path, cfg: dict | None = None, verbose: bool = True) -> dict:
    """Compute and save a large-age core mask from a wall bank."""
    cfg = cfg or {}
    diagnostics_dir = out_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    mode = str(cfg.get("core_mode", "entry"))
    a_star = int(cfg.get("core_a_star", 10))
    n_psi_bins = int(cfg.get("core_n_psi_bins", 32))
    min_count = int(cfg.get("core_min_count", 500))
    min_positive = int(cfg.get("core_min_positive", 25))
    entropy_threshold = float(cfg.get("core_entropy_threshold", 0.9))
    hazard_cv_threshold = float(cfg.get("core_hazard_cv_threshold", 0.25))
    retention_threshold = float(cfg.get("core_retention_threshold", 0.8))
    horizon = int(cfg.get("core_horizon", 5))
    min_phase_bin_count = int(cfg.get("core_min_phase_bin_count", 20))
    tag = str(cfg.get("core_tag", "")).strip()

    if verbose:
        base = (
            f"[core_mask] mode={mode}, a_star={a_star}, n_psi_bins={n_psi_bins}, min_count={min_count}, "
            f"entropy_threshold={entropy_threshold:.2f}"
        )
        if mode == "retention":
            print(base + f", horizon={horizon}, retention_threshold={retention_threshold:.2f}, retention_cv_threshold={hazard_cv_threshold:.2f}")
        else:
            print(base + f", hazard_cv_threshold={hazard_cv_threshold:.2f}")

    if mode == "retention":
        stats = build_retention_phase_stats(
            bank["u_traj"],
            bank["psi_traj"],
            bank["u_edges"],
            bank["proxy_labels"].astype(bool),
            a_star=a_star,
            horizon=horizon,
            n_psi_bins=n_psi_bins,
            min_phase_bin_count=min_phase_bin_count,
        )
        core_mask = build_retention_core_mask(
            stats["phase_entropy"],
            stats["phase_hazard_cv"],
            stats["retention_mean"],
            stats["counts_after_age_cut"],
            entropy_threshold=entropy_threshold,
            retention_cv_threshold=hazard_cv_threshold,
            retention_threshold=retention_threshold,
            min_count=min_count,
        )
        npz_name = "core_mask_retention.npz"
        png_name = "core_mask_retention.png"
    elif mode == "entry":
        stats = build_age_conditioned_phase_stats(
            bank["u_traj"],
            bank["psi_traj"],
            bank["u_edges"],
            bank["proxy_labels"].astype(bool),
            a_star=a_star,
            n_psi_bins=n_psi_bins,
            min_phase_bin_count=min_phase_bin_count,
        )
        core_mask = build_core_mask(
            stats["phase_entropy"],
            stats["phase_hazard_cv"],
            stats["counts_after_age_cut"],
            stats["positive_counts"],
            entropy_threshold=entropy_threshold,
            hazard_cv_threshold=hazard_cv_threshold,
            min_count=min_count,
            min_positive=min_positive,
        )
        npz_name = "core_mask.npz"
        png_name = "core_mask.png"
    else:
        raise ValueError(f"Unsupported core mask mode: {mode}")

    if tag:
        safe_tag = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in tag)
        stem_npz = Path(npz_name).stem
        stem_png = Path(png_name).stem
        npz_name = f"{stem_npz}_{safe_tag}.npz"
        png_name = f"{stem_png}_{safe_tag}.png"

    core_labels = build_labels_from_mask(stats["u_bin_idx"], core_mask)

    np.savez_compressed(
        diagnostics_dir / npz_name,
        u_centers=bank["u_centers"],
        u_edges=bank["u_edges"],
        core_mask=core_mask.astype(np.int8),
        core_labels=core_labels.astype(np.int8),
        mode=np.array([mode]),
        phase_entropy=stats["phase_entropy"],
        phase_hazard_cv=stats["phase_hazard_cv"],
        counts_after_age_cut=stats["counts_after_age_cut"],
        positive_counts=stats["positive_counts"],
        a_star=np.array([a_star], dtype=np.int32),
        n_psi_bins=np.array([n_psi_bins], dtype=np.int32),
        min_count=np.array([min_count], dtype=np.int32),
        entropy_threshold=np.array([entropy_threshold], dtype=np.float64),
        hazard_cv_threshold=np.array([hazard_cv_threshold], dtype=np.float64),
        **({"event_rate": stats["event_rate"], "min_positive": np.array([min_positive], dtype=np.int32)} if mode == "entry" else {}),
        **({"retention_mean": stats["retention_mean"], "horizon": np.array([horizon], dtype=np.int32), "retention_threshold": np.array([retention_threshold], dtype=np.float64)} if mode == "retention" else {}),
    )
    if mode == "retention":
        make_retention_core_mask_report(
            diagnostics_dir / png_name,
            bank["u_centers"],
            stats,
            core_mask,
            entropy_threshold,
            hazard_cv_threshold,
            retention_threshold,
            min_count,
        )
    else:
        make_core_mask_report(
            diagnostics_dir / png_name,
            bank["u_centers"],
            stats,
            core_mask,
            entropy_threshold,
            hazard_cv_threshold,
            min_count,
        )

    if verbose:
        print(f"[core_mask] Saved: {diagnostics_dir / npz_name}")
        print(f"[core_mask] Saved: {diagnostics_dir / png_name}")
        print(f"[core_mask] Core bins: {int(core_mask.sum())}/{len(core_mask)}")

    return {**stats, "core_mask": core_mask, "core_labels": core_labels, "mode": mode}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build an age-conditioned core mask from a real wall bank")
    ap.add_argument("--bank", required=True, help="Path to wall_bank.npz")
    ap.add_argument("--out-dir", default=None, help="Output directory; defaults to the bank directory")
    ap.add_argument("--mode", choices=["entry", "retention"], default="entry")
    ap.add_argument("--a-star", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--n-psi-bins", type=int, default=32)
    ap.add_argument("--min-count", type=int, default=500)
    ap.add_argument("--min-positive", type=int, default=25)
    ap.add_argument("--entropy-threshold", type=float, default=0.9)
    ap.add_argument("--hazard-cv-threshold", type=float, default=0.25)
    ap.add_argument("--retention-threshold", type=float, default=0.8)
    ap.add_argument("--min-phase-bin-count", type=int, default=20)
    ap.add_argument("--tag", default="", help="Optional suffix for output filenames")
    args = ap.parse_args()

    bank_path = Path(args.bank)
    out_dir = Path(args.out_dir) if args.out_dir is not None else bank_path.parent
    bank = np.load(bank_path, allow_pickle=False)
    cfg = {
        "core_mode": args.mode,
        "core_a_star": args.a_star,
        "core_horizon": args.horizon,
        "core_n_psi_bins": args.n_psi_bins,
        "core_min_count": args.min_count,
        "core_min_positive": args.min_positive,
        "core_entropy_threshold": args.entropy_threshold,
        "core_hazard_cv_threshold": args.hazard_cv_threshold,
        "core_retention_threshold": args.retention_threshold,
        "core_min_phase_bin_count": args.min_phase_bin_count,
        "core_tag": args.tag,
    }
    run_core_mask(bank, out_dir, cfg=cfg, verbose=True)


if __name__ == "__main__":
    main()