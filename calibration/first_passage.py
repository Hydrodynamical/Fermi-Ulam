"""Residual waiting-time law into the binwise Markov proxy.

For each bounce n0 where the energy bin is NOT in the chaos mask (M_n0 = 0),
we record how many additional bounces until the first M_n = 1 (n >= n0). This
residual waiting time (RWT) is conditioned on finding the particle outside the
proxy set at an *arbitrary* time — the correct object for PDMP collision sampling,
where bulk collisions arrive at arbitrary moments.

The proxy label M_n = 1 iff u_n falls in a chaotic energy bin (chaos mask) is a
*binwise Markov proxy*, not a guarantee of true Markov validity. Sticky KAM
trajectories within nominally chaotic bins survive this filter. The outputs of this
module quantify how well the single-exponential-clock reduction is justified.

Key outputs
-----------
S[b, a]      Empirical survival P(T > a | start in bin b) — censored obs included
h[b, a]      Discrete hazard P(T = a+1 | T > a, start in bin b)
tau_rmst[b]  Restricted Mean Survival Time up to max_age = sum_a S[b, a]
tau_exp[b]   Exponential-fit timescale: 1 / slope of (log S vs a) through origin
r2[b]        R² of exponential fit on log scale
h_cv[b]      Coefficient of variation of h[b, 0:50] — non-exponentiality score

Usage
-----
    from first_passage import run_first_passage
    run_first_passage(results, cfg, out_dir)

Or stand-alone:
    python3 calibration/first_passage.py results/calibration
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Label construction
# ---------------------------------------------------------------------------

def build_proxy_labels(
    u_traj: np.ndarray,
    u_edges: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Build binwise Markov proxy labels from the chaos mask.

    Parameters
    ----------
    u_traj  : (N_p, N_h+1) energy trajectory
    u_edges : (n_bins+1,) bin edges
    mask    : (n_bins,) bool — True for chaotic bins

    Returns
    -------
    labels : (N_p, N_h+1) bool
        True iff the bounce's energy bin is in the chaos mask.
        This is a *binwise Markov proxy*, not a guarantee of Markov validity.
    """
    n_bins = len(mask)
    bin_idx = np.clip(np.digitize(u_traj, u_edges) - 1, 0, n_bins - 1)
    return mask[bin_idx]


# ---------------------------------------------------------------------------
# Residual waiting-time computation
# ---------------------------------------------------------------------------

def compute_residual_waiting_times(
    labels: np.ndarray,
    u_traj: np.ndarray,
    u_edges: np.ndarray,
    n_bins: int,
    max_age: int = 500,
    n_min: int = 30,
) -> dict:
    """Compute residual waiting times to enter the binwise Markov proxy.

    For each starting index n0 where labels[i, n0] = False, find the first
    m >= 0 such that labels[i, n0 + m] = True (using binary search on the
    precomputed true_positions for each trajectory).  Observations where no
    entry occurs within the trajectory are right-censored.

    Parameters
    ----------
    labels  : (N_p, N_h+1) bool proxy labels from build_proxy_labels
    u_traj  : (N_p, N_h+1) energy trajectory for binning starting positions
    u_edges : (n_bins+1,) bin edges
    n_bins  : int
    max_age : int — observations with T > max_age counted as censored
    n_min   : int — kept for API consistency; threshold applied later

    Returns
    -------
    dict with:
        rwt_by_bin : list[np.ndarray] length n_bins — finite RWTs (≤ max_age)
        counts     : (n_bins,) int64 — total non-chaotic starts per bin
        censored   : (n_bins,) int64 — starts with T > max_age (right-censored)
    """
    N_p, N_hm1 = labels.shape

    rwt_by_bin = [[] for _ in range(n_bins)]
    counts = np.zeros(n_bins, dtype=np.int64)
    censored = np.zeros(n_bins, dtype=np.int64)

    for i in range(N_p):
        row_labels = labels[i]
        row_u = u_traj[i]

        false_positions = np.where(~row_labels)[0]
        if len(false_positions) == 0:
            continue  # entire trajectory is already in proxy set

        # Bin each non-chaotic starting position
        b_arr = np.clip(
            np.digitize(row_u[false_positions], u_edges) - 1,
            0, n_bins - 1,
        )
        np.add.at(counts, b_arr, 1)

        true_positions = np.where(row_labels)[0]
        if len(true_positions) == 0:
            # Never enters proxy set — all censored
            np.add.at(censored, b_arr, 1)
            continue

        # Vectorised binary search: for each false_position find the first
        # true_position at or after it.
        idx = np.searchsorted(true_positions, false_positions, side='left')
        found = idx < len(true_positions)
        safe_idx = np.clip(idx, 0, len(true_positions) - 1)
        T = np.where(found, true_positions[safe_idx] - false_positions, N_hm1)

        in_window = found & (T <= max_age)
        np.add.at(censored, b_arr, (~in_window).astype(np.int64))

        if not in_window.any():
            continue

        # Accumulate finite RWTs grouped by starting bin — sort then split
        valid_b = b_arr[in_window]
        valid_T = T[in_window]
        order = np.argsort(valid_b, kind='stable')
        srt_b = valid_b[order]
        srt_T = valid_T[order]
        split_at = np.searchsorted(srt_b, np.arange(1, n_bins))
        for b, grp in enumerate(np.split(srt_T, split_at)):
            if len(grp):
                rwt_by_bin[b].append(grp)

    rwt_by_bin = [
        np.concatenate(parts).astype(np.int64) if parts
        else np.empty(0, dtype=np.int64)
        for parts in rwt_by_bin
    ]

    return {"rwt_by_bin": rwt_by_bin, "counts": counts, "censored": censored}


# ---------------------------------------------------------------------------
# Survival and hazard estimation
# ---------------------------------------------------------------------------

def estimate_survival_hazard(
    rwt_by_bin: list,
    counts: np.ndarray,
    max_age: int,
    n_min: int = 30,
) -> dict:
    """Estimate empirical survival and discrete hazard from RWT samples.

    Censored observations (T > max_age) contribute to #{T > a} for all
    a ≤ max_age, which is correct for right-censored data when the censoring
    threshold equals max_age.

    S[b, a]     = #{T > a} / counts[b]
    h[b, a]     = #{T = a+1} / #{T > a}
    tau_rmst[b] = Σ_a S[b, a]  (Restricted Mean Survival Time up to max_age)

    Parameters
    ----------
    rwt_by_bin : list[np.ndarray] of finite RWT samples (≤ max_age) per bin
    counts     : (n_bins,) total starts per bin (including censored)
    max_age    : int
    n_min      : int — bins with counts < n_min get NaN

    Returns
    -------
    dict with: S (n_bins, max_age+1), h (n_bins, max_age),
               tau_rmst (n_bins,), a_arr (max_age+1,)
    """
    n_bins = len(rwt_by_bin)
    a_arr = np.arange(max_age + 1, dtype=np.int64)

    S = np.full((n_bins, max_age + 1), np.nan)
    h = np.full((n_bins, max_age), np.nan)
    tau_rmst = np.full(n_bins, np.nan)

    for b in range(n_bins):
        c = int(counts[b])
        if c < n_min:
            continue

        rwt = np.sort(rwt_by_bin[b])  # sorted finite RWTs ≤ max_age

        # #{T ≤ a}: searchsorted gives position of first element > a
        n_le = np.searchsorted(rwt, a_arr, side='right')   # (max_age+1,)

        # S[b, a] = #{T > a} / c; right-censored obs have T > max_age ≥ a
        S[b] = (c - n_le) / c

        # Hazard: h[b, a] = #{T = a+1} / #{T > a}  for a = 0 .. max_age-1
        n_above = (c - n_le[:-1]).astype(float)   # #{T > a}
        n_at_next = np.diff(n_le).astype(float)   # #{T = a+1}
        with np.errstate(invalid='ignore', divide='ignore'):
            h[b] = np.where(n_above > 0, n_at_next / n_above, np.nan)

        # Restricted Mean Survival Time
        tau_rmst[b] = float(np.sum(S[b]))

    return {"S": S, "h": h, "tau_rmst": tau_rmst, "a_arr": a_arr}


# ---------------------------------------------------------------------------
# Exponential fit and non-exponentiality score
# ---------------------------------------------------------------------------

def fit_exponential_tau(
    S: np.ndarray,
    a_arr: np.ndarray,
    mask: np.ndarray,
    h: np.ndarray | None = None,
    n_min_pts: int = 10,
) -> dict:
    """Fit S(a) ~ exp(-a / tau_exp) and compute non-exponentiality score.

    Fitting method: OLS of y = -log S vs a through the origin.
        slope = Σ(a · y) / Σ(a²)
        tau_exp = 1 / slope          (NOTE: tau_exp = 1/slope, not slope itself)

    Non-exponentiality score h_cv: coefficient of variation of h[b, 0:50].
    A flat hazard (h_cv ≈ 0) supports the exponential-clock reduction.

    Parameters
    ----------
    S        : (n_bins, max_age+1) survival function
    a_arr    : (max_age+1,) lag axis
    mask     : (n_bins,) bool — chaotic bins only
    h        : (n_bins, max_age) hazard (optional; used for h_cv)
    n_min_pts: int — minimum finite points for the fit

    Returns
    -------
    dict with: tau_exp (n_bins,), r2 (n_bins,), h_cv (n_bins,)
    """
    n_bins = S.shape[0]
    tau_exp = np.full(n_bins, np.nan)
    r2 = np.full(n_bins, np.nan)
    h_cv = np.full(n_bins, np.nan)

    for b in range(n_bins):
        S_row = S[b]
        if np.all(np.isnan(S_row)):
            continue

        # --- Exponential fit on log S ---
        valid = (S_row > 0) & np.isfinite(S_row) & (a_arr > 0)
        if valid.sum() < n_min_pts:
            continue
        a_v = a_arr[valid].astype(float)
        y_v = -np.log(S_row[valid])

        denom = float(np.dot(a_v, a_v))
        if denom == 0:
            continue
        slope = float(np.dot(a_v, y_v)) / denom
        if slope <= 0:
            continue
        tau_exp[b] = 1.0 / slope

        # R² on log scale
        y_fit = a_v * slope
        ss_res = float(np.sum((y_v - y_fit) ** 2))
        ss_tot = float(np.sum((y_v - np.mean(y_v)) ** 2))
        r2[b] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        # --- Non-exponentiality score from hazard ---
        if h is not None:
            h_row = h[b]
            early_len = min(len(h_row), 50)
            h_early = h_row[:early_len]
            valid_h = h_early[np.isfinite(h_early)]
            if len(valid_h) > 1:
                mu_h = float(np.mean(valid_h))
                if mu_h > 0:
                    h_cv[b] = float(np.std(valid_h)) / mu_h

    return {"tau_exp": tau_exp, "r2": r2, "h_cv": h_cv}


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_first_passage_report(
    out_dir,
    u_centers: np.ndarray,
    sv: dict,
    ex: dict,
    mask: np.ndarray,
    out_path=None,
) -> None:
    """2×2 diagnostic figure for the residual waiting-time analysis.

    Panels
    ------
    (0,0)  Survival curves S(u, a) for 4 representative chaotic bins, semilog-y.
           Dashed overlay = exp(-a / tau_exp[b]).
    (0,1)  Discrete hazard h(u, a) for same 4 bins, linear scale.
           Dashed overlay = constant 1/tau_exp[b].
    (1,0)  tau_rmst vs tau_exp vs u, semilog-x.
    (1,1)  R²(u) left axis and h_cv(u) right axis vs u, semilog-x.

    Saved as out_dir/first_passage.png.
    """
    out_dir = Path(out_dir)
    if out_path is None:
        out_path = out_dir / "first_passage.png"

    S = sv["S"]
    h = sv["h"]
    tau_rmst = sv["tau_rmst"]
    a_arr = sv["a_arr"]
    tau_exp = ex["tau_exp"]
    r2 = ex["r2"]
    h_cv = ex["h_cv"]

    # --- Pick 4 representative non-chaotic starting bins ---
    # Starting positions are non-chaotic by construction; data lives in ~mask bins.
    start_idx = np.where(~mask & np.isfinite(tau_rmst))[0]
    if len(start_idx) == 0:
        print("[first_passage] No non-chaotic starting bins with data — skipping figure.")
        return
    n_rep = min(4, len(start_idx))
    if n_rep < 4:
        rep_bins = start_idx[:n_rep]
    else:
        fracs = [0.125, 0.375, 0.625, 0.875]
        rep_bins = start_idx[
            [int(f * (len(start_idx) - 1)) for f in fracs]
        ]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"][:n_rep]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Residual waiting-time into binwise Markov proxy", fontsize=11)

    a_float = a_arr.astype(float)

    # --- Panel (0,0): Survival ---
    ax = axes[0, 0]
    for b, col in zip(rep_bins, colors):
        S_row = S[b]
        label = f"u={u_centers[b]:.2f}"
        valid = np.isfinite(S_row) & (S_row > 0)
        ax.semilogy(a_float[valid], S_row[valid], lw=1.5, color=col, label=label)
        if np.isfinite(tau_exp[b]):
            ax.semilogy(a_float, np.exp(-a_float / tau_exp[b]),
                        lw=0.8, ls="--", color=col, alpha=0.6)
    ax.set_xlabel("a (bounces)")
    ax.set_ylabel("S(u, a)")
    ax.set_title("Survival S(u, a)")
    ax.legend(fontsize=7)
    ax.set_xlim(0, min(int(a_arr[-1]), 200))

    # --- Panel (0,1): Hazard ---
    ax = axes[0, 1]
    a_h = a_float[:-1]   # lag axis for h (length max_age)
    for b, col in zip(rep_bins, colors):
        h_row = h[b]
        label = f"u={u_centers[b]:.2f}"
        valid = np.isfinite(h_row)
        if valid.any():
            ax.plot(a_h[valid], h_row[valid], lw=1.5, color=col, label=label)
        if np.isfinite(tau_exp[b]):
            ax.axhline(1.0 / tau_exp[b], lw=0.8, ls="--", color=col, alpha=0.6)
    ax.set_xlabel("a (bounces)")
    ax.set_ylabel("h(u, a)")
    ax.set_title("Discrete hazard h(u, a)")
    ax.legend(fontsize=7)
    ax.set_xlim(0, min(int(a_arr[-1]), 100))

    # --- Panel (1,0): tau_rmst vs tau_exp vs u ---
    # Non-chaotic starting bins (where data lives) in color; chaotic bins in gray.
    ax = axes[1, 0]
    nc_u = np.where(~mask, u_centers, np.nan)   # non-chaotic (starting bins)
    ch_u = np.where(mask, u_centers, np.nan)    # chaotic (destination bins)
    ax.semilogx(nc_u, np.where(~mask, tau_rmst, np.nan),
                "o", ms=5, color="tab:blue", label=r"$\tau_\mathrm{rmst}$")
    ax.semilogx(nc_u, np.where(~mask, tau_exp, np.nan),
                "x", ms=6, mew=1.5, color="tab:orange", label=r"$\tau_\mathrm{exp}$")
    ax.semilogx(ch_u, np.where(mask, tau_rmst, np.nan),
                "o", ms=4, color="silver", label="chaotic (0 starts)")
    ax.set_xlabel("u")
    ax.set_ylabel("bounces")
    ax.set_title(r"$\tau_\mathrm{rmst}$ vs $\tau_\mathrm{exp}$ vs u")
    ax.legend(fontsize=7)

    # --- Panel (1,1): R² and h_cv vs u ---
    ax = axes[1, 1]
    ax2 = ax.twinx()
    ax.semilogx(nc_u, np.where(~mask, r2, np.nan),
                "o-", ms=4, lw=1.2, color="tab:purple", label=r"$R^2$")
    ax2.semilogx(nc_u, np.where(~mask, h_cv, np.nan),
                 "s--", ms=4, lw=1.2, color="tab:gray", label="h_cv")
    ax.axhline(0.9, color="tab:purple", ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel("u")
    ax.set_ylabel(r"$R^2$ (log-linear fit)", color="tab:purple")
    ax2.set_ylabel("h_cv (non-exponentiality)", color="tab:gray")
    ax.set_ylim(-0.1, 1.05)
    ax.set_title(r"Fit quality: $R^2$ and $h_\mathrm{cv}$")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[first_passage] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_first_passage(
    results: dict,
    cfg: dict,
    out_dir,
    verbose: bool = True,
) -> dict:
    """Top-level runner for the residual waiting-time diagnostic.

    Parameters
    ----------
    results : dict from run_calibration() — must contain u_traj, u_centers, u_edges
    cfg     : merged config dict
    out_dir : calibration output directory (diagnostics/ created inside)
    verbose : print progress

    Returns
    -------
    dict with all computed arrays
    """
    out_dir = Path(out_dir)
    diag_dir = out_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    u_traj    = results["u_traj"]
    u_centers = results["u_centers"]
    u_edges   = results["u_edges"]
    n_bins    = len(u_centers)
    max_age   = int(cfg.get("fp_max_age", 500))

    # --- Load or rebuild chaos mask ---
    mask = _load_mask(diag_dir, results, cfg, verbose)

    if verbose:
        n_chaotic = int(mask.sum())
        print(f"  Chaos mask: {n_chaotic}/{n_bins} chaotic bins")

    # --- Build proxy labels ---
    labels = build_proxy_labels(u_traj, u_edges, mask)
    if verbose:
        proxy_frac = float(labels.mean())
        print(f"  Proxy labels: {proxy_frac:.1%} of bounces already in chaotic set")

    # --- Residual waiting times ---
    if verbose:
        print(f"  Computing residual waiting times (max_age={max_age}) ...")
    fp = compute_residual_waiting_times(labels, u_traj, u_edges, n_bins, max_age=max_age)

    total_starts = int(fp["counts"].sum())
    total_censored = int(fp["censored"].sum())
    if verbose:
        cens_pct = 100 * total_censored / max(total_starts, 1)
        print(f"  Total non-chaotic starts: {total_starts:,}  "
              f"censored: {total_censored:,} ({cens_pct:.1f}%)")
        # Per-bin RWT summary for chaotic bins
        chaotic_bins = np.where(mask)[0]
        if len(chaotic_bins):
            print(f"  {'bin':>4}  {'u':>8}  {'starts':>7}  {'finite':>7}  "
                  f"{'cens':>6}  {'median_T':>8}")
            for b in chaotic_bins:
                c = int(fp["counts"][b])
                if c == 0:
                    continue
                n_fin = len(fp["rwt_by_bin"][b])
                cens  = int(fp["censored"][b])
                med   = int(np.median(fp["rwt_by_bin"][b])) if n_fin > 0 else -1
                med_s = f"{med:8d}" if n_fin > 0 else "      --"
                print(f"  {b:>4}  {u_centers[b]:>8.3g}  {c:>7d}  {n_fin:>7d}  "
                      f"{cens:>6d}  {med_s}")

    # --- Survival and hazard ---
    sv = estimate_survival_hazard(fp["rwt_by_bin"], fp["counts"], max_age)
    if verbose:
        chaotic_idx = np.where(~mask & np.isfinite(sv["tau_rmst"]))[0]
        if len(chaotic_idx):
            mid = chaotic_idx[len(chaotic_idx) // 2]
            a50 = min(50, sv["S"].shape[1] - 1)
            print(f"  Survival spot-check bin {mid} (u={u_centers[mid]:.3g}): "
                  f"S(0)={sv['S'][mid, 0]:.3f}  "
                  f"S(10)={sv['S'][mid, min(10, a50)]:.3f}  "
                  f"S(50)={sv['S'][mid, a50]:.3f}")

    # --- Exponential fit ---
    ex = fit_exponential_tau(sv["S"], sv["a_arr"], mask, h=sv["h"])

    # --- Per-bin table ---
    if verbose:
        _print_per_bin_table(u_centers, mask, fp, sv, ex)

    # --- Save ---
    np.savez_compressed(
        diag_dir / "first_passage.npz",
        S=sv["S"],
        h=sv["h"],
        tau_rmst=sv["tau_rmst"],
        tau_exp=ex["tau_exp"],
        r2=ex["r2"],
        h_cv=ex["h_cv"],
        a_arr=sv["a_arr"],
        counts=fp["counts"],
        censored=fp["censored"],
    )
    if verbose:
        print("  Saved first_passage.npz")

    # --- Figure ---
    make_first_passage_report(diag_dir, u_centers, sv, ex, mask)

    ret = {**sv, **ex, **fp}
    return ret


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _print_per_bin_table(
    u_centers: np.ndarray,
    mask: np.ndarray,
    fp: dict,
    sv: dict,
    ex: dict,
) -> None:
    """Print a formatted per-bin summary table to stdout."""
    n_bins = len(u_centers)
    tau_rmst = sv["tau_rmst"]
    tau_exp  = ex["tau_exp"]
    r2       = ex["r2"]
    h_cv     = ex["h_cv"]
    counts   = fp["counts"]
    censored = fp["censored"]

    def _fmt(v, fmt=".1f"):
        return f"{v:{fmt}}" if np.isfinite(v) else "    --"

    print(f"\n  First-passage per-bin results:")
    print(f"  {'bin':>4}  {'u':>8}  {'M':>1}  {'starts':>7}  "
          f"{'cens%':>6}  {'tau_rmst':>9}  {'tau_exp':>8}  {'R2':>6}  {'h_cv':>6}")
    print("  " + "-" * 72)
    for b in range(n_bins):
        c = int(counts[b])
        cens = int(censored[b])
        cens_pct = f"{100 * cens / c:5.1f}%" if c > 0 else "    --"
        m_char = "C" if mask[b] else "."
        print(f"  {b:>4}  {u_centers[b]:>8.3g}  {m_char:>1}  {c:>7d}  "
              f"{cens_pct:>6}  {_fmt(tau_rmst[b]):>9}  "
              f"{_fmt(tau_exp[b]):>8}  {_fmt(r2[b], '.3f'):>6}  "
              f"{_fmt(h_cv[b], '.3f'):>6}")

    active = ~mask & np.isfinite(tau_rmst)
    if active.any():
        print(f"\n  tau_rmst range (non-chaotic starting bins): "
              f"[{np.nanmin(tau_rmst[active]):.1f}, {np.nanmax(tau_rmst[active]):.1f}] bounces")
        good_r2 = int(np.sum(np.isfinite(r2[active]) & (r2[active] > 0.9)))
        print(f"  R²>0.9: {good_r2}/{int(active.sum())} non-chaotic starting bins")


def _load_mask(diag_dir: Path, results: dict, cfg: dict, verbose: bool) -> np.ndarray:
    """Load chaos mask from chaos_mask.npz, rebuild, or fall back to all-True."""
    n_bins = len(results["u_centers"])
    chaos_npz = diag_dir / "chaos_mask.npz"

    if chaos_npz.exists():
        try:
            d = np.load(chaos_npz)
            return d["mask"].astype(bool)
        except Exception:
            pass

    # Rebuild from mixing_diagnostics.npz + chaos_mask.py
    try:
        from chaos_mask import build_chaotic_mask
        mix_npz = diag_dir / "mixing_diagnostics.npz"
        if mix_npz.exists():
            mix = np.load(mix_npz)
            entropy_norm = mix["entropy_norm"]
        else:
            entropy_norm = results.get("entropy_norm", None)
        tau_lag = results["tau_lag"]
        if entropy_norm is not None:
            threshold = float(cfg.get("entropy_threshold", 0.8))
            mask = build_chaotic_mask(entropy_norm, tau_lag, entropy_threshold=threshold)
            if verbose:
                print("  [first_passage] Rebuilt chaos mask from entropy_norm + tau_lag.")
            return mask
    except Exception:
        pass

    if verbose:
        print("  [first_passage] Chaos mask unavailable — using all-True mask.")
    return np.ones(n_bins, dtype=bool)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys as _sys

    if len(_sys.argv) < 2:
        print("Usage: python3 first_passage.py <out_dir>")
        _sys.exit(1)

    _out_dir = Path(_sys.argv[1])
    _cal_npz = _out_dir / "calibration_results.npz"
    if not _cal_npz.exists():
        print(f"calibration_results.npz not found in {_out_dir}")
        _sys.exit(1)

    _cal = np.load(_cal_npz)
    _results = {k: _cal[k] for k in _cal.files}

    # u_traj is not saved in calibration_results.npz (too large);
    # fall back to the diagnostic subset saved by run_calibration.
    if "u_traj" not in _results:
        _diag_npz = _out_dir / "diag_trajectories.npz"
        if not _diag_npz.exists():
            print(
                "u_traj not found in calibration_results.npz and "
                "diag_trajectories.npz is missing.\n"
                "Re-run via run_calibration.py to regenerate u_traj."
            )
            _sys.exit(1)
        _diag = np.load(_diag_npz)
        _results["u_traj"]   = _diag["u_traj"]
        _results["psi_traj"] = _diag["psi_traj"]
        print(
            f"  Note: using diagnostic subset "
            f"({_results['u_traj'].shape[0]} particles from diag_trajectories.npz)"
        )

    run_first_passage(_results, cfg={}, out_dir=_out_dir, verbose=True)
