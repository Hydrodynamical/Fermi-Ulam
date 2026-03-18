"""Lichtenberg-Lieberman (LL) validation for the calibration experiment.

Tests whether the empirical Fermi-Ulam transport coefficients are consistent
with the textbook random-phase prediction (Lichtenberg & Lieberman, §5.4):

    D(u) ≡ E[(Δu)² | u] ∝ u          (second moment per bounce)
    B(u) ≡ E[Δu    | u] ≈ const       (first moment per bounce)
    B(u) ≈ ½ dD/du                    (Hamiltonian detailed-balance identity)

Under these conditions the stationary density satisfies:

    P(u) ≈ const  on the chaotic interval   [flat in energy]
    P(s) ∝ s      in speed  s = √(2u)       [linear in speed]

All tests use one-bounce increments (Δn = 1), NOT a coarse lag.
All tests are restricted to the chaotic sea (entropy mask).

Functions
---------
compute_one_bounce_transport   -- B(u) and D(u) at Δn=1, chaotic bins only
check_landau_relation          -- residual R(u) = B - ½ D'
fit_ll_shapes                  -- fit D ∝ u, B ≈ const; compute ll_ratio
compare_stationary_density     -- f_emp vs flat (u) and linear (s)
make_ll_validation_report      -- 2×2 diagnostic figure
run_ll_validation              -- top-level runner

Usage
-----
Runs automatically as part of run_calibration.py diagnostics.

Stand-alone:
    python3 calibration/ll_validation.py results/calibration
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from chaos_mask import build_chaotic_mask


# ---------------------------------------------------------------------------
# Core transport computation
# ---------------------------------------------------------------------------

def compute_one_bounce_transport(
    u_traj: np.ndarray,
    u_edges: np.ndarray,
    mask: np.ndarray | None = None,
    n_min: int = 50,
) -> dict:
    """Per-bounce first and second moments of Δu = u_{n+1} − u_n.

    Uses Δn = 1 (one wall bounce), NOT the coarse lag m_arr used elsewhere.
    Bins by the starting energy u_n.  If mask is provided, only processes
    bins where mask[b] = True; others are returned as NaN.

    Parameters
    ----------
    u_traj   : (N_particles, N_hits+1)
    u_edges  : (n_bins+1,)
    mask     : (n_bins,) bool or None (treat all bins as valid)
    n_min    : minimum samples per bin

    Returns
    -------
    dict with keys:
      B       : (n_bins,)  E[Δu | u_n in bin]          — per-bounce drift
      D       : (n_bins,)  E[(Δu)² | u_n in bin]       — raw second moment
      counts  : (n_bins,)  int64
    """
    N_p, N_h1 = u_traj.shape
    N_h = N_h1 - 1
    n_bins = len(u_edges) - 1

    if mask is None:
        mask = np.ones(n_bins, dtype=bool)

    sum1 = np.zeros(n_bins)   # sum of Δu
    sum2 = np.zeros(n_bins)   # sum of (Δu)²
    cnt  = np.zeros(n_bins, dtype=np.int64)

    for i in range(N_p):
        u_i = u_traj[i]
        u_base = u_i[:N_h]          # u_n,  shape (N_h,)
        u_next = u_i[1:]            # u_{n+1}
        du = u_next - u_base        # Δu

        # Bin each base hit
        bin_idx = np.searchsorted(u_edges, u_base, side="right") - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        for b in range(n_bins):
            if not mask[b]:
                continue
            sel = bin_idx == b
            if not sel.any():
                continue
            dub = du[sel]
            sum1[b] += dub.sum()
            sum2[b] += (dub ** 2).sum()
            cnt[b]  += sel.sum()

    B = np.full(n_bins, np.nan)
    D = np.full(n_bins, np.nan)
    for b in range(n_bins):
        if cnt[b] >= n_min:
            B[b] = sum1[b] / cnt[b]
            D[b] = sum2[b] / cnt[b]

    return {"B": B, "D": D, "counts": cnt}


# ---------------------------------------------------------------------------
# Landau / Hamiltonian identity check
# ---------------------------------------------------------------------------

def check_landau_relation(
    u_centers: np.ndarray,
    B: np.ndarray,
    D: np.ndarray,
) -> dict:
    """Verify B(u) ≈ ½ dD/du (Hamiltonian detailed-balance identity).

    For a Hamiltonian system in the random-phase chaotic sea, the FP
    coefficients must satisfy B = ½ D' exactly.  Deviations indicate
    either non-Hamiltonian corrections or that we are outside the chaotic sea.

    Returns
    -------
    dict with keys:
      D_deriv  : (n_bins,)  numerical derivative dD/du (central differences)
      residual : (n_bins,)  R(u) = B(u) − ½ dD/du
    """
    n = len(u_centers)
    D_deriv = np.full(n, np.nan)

    for b in range(n):
        # Central difference where possible
        if b == 0:
            # forward difference
            if np.isfinite(D[b]) and np.isfinite(D[b + 1]):
                D_deriv[b] = (D[b + 1] - D[b]) / (u_centers[b + 1] - u_centers[b])
        elif b == n - 1:
            # backward difference
            if np.isfinite(D[b]) and np.isfinite(D[b - 1]):
                D_deriv[b] = (D[b] - D[b - 1]) / (u_centers[b] - u_centers[b - 1])
        else:
            if np.isfinite(D[b - 1]) and np.isfinite(D[b + 1]):
                D_deriv[b] = (D[b + 1] - D[b - 1]) / (u_centers[b + 1] - u_centers[b - 1])

    residual = B - 0.5 * D_deriv

    return {"D_deriv": D_deriv, "residual": residual}


# ---------------------------------------------------------------------------
# Shape fitting
# ---------------------------------------------------------------------------

def fit_ll_shapes(
    u_centers: np.ndarray,
    B: np.ndarray,
    D: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """Fit LL functional forms D ∝ u and B ≈ const on the chaotic interval.

    LL predicts (in their normalisation): D(u) = 2u, B(u) = 1.
    For a general normalisation, the functional forms should hold and
    ll_ratio = c_B / (½ c_D) should be ≈ 1.

    Parameters
    ----------
    u_centers : (n_bins,)
    B, D      : (n_bins,)  one-bounce transport coefficients
    mask      : (n_bins,) bool  chaotic bins

    Returns
    -------
    dict with keys:
      c_D      : float  slope of D(u)/u on chaotic interval
      c_B      : float  mean B(u) on chaotic interval
      ll_ratio : float  c_B / (½ c_D) — should be ~1 for Hamiltonian RP
      D_fit    : (n_bins,)  c_D * u_centers
      B_fit    : (n_bins,)  c_B * ones
    """
    valid = mask & np.isfinite(B) & np.isfinite(D) & (D > 0)

    if valid.sum() < 2:
        nan = np.float64(np.nan)
        return {
            "c_D": nan, "c_B": nan, "ll_ratio": nan,
            "D_fit": np.full(len(u_centers), np.nan),
            "B_fit": np.full(len(u_centers), np.nan),
        }

    u_v = u_centers[valid]
    D_v = D[valid]
    B_v = B[valid]

    # D = c_D * u  (no intercept, OLS through origin)
    c_D = float(np.dot(D_v, u_v) / np.dot(u_v, u_v))

    c_B = float(np.mean(B_v))

    ll_ratio = float(c_B / (0.5 * c_D)) if c_D != 0 else np.nan

    D_fit = c_D * u_centers
    B_fit = np.full(len(u_centers), c_B)

    return {
        "c_D": np.float64(c_D),
        "c_B": np.float64(c_B),
        "ll_ratio": np.float64(ll_ratio),
        "D_fit": D_fit,
        "B_fit": B_fit,
    }


# ---------------------------------------------------------------------------
# Stationary density comparison
# ---------------------------------------------------------------------------

def compare_stationary_density(
    u_traj: np.ndarray,
    u_edges: np.ndarray,
    u_centers: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """Compare empirical stationary density to LL prediction on chaotic bins.

    LL predicts:
      P(u) ≈ const  (flat in energy)     → histogram in u should be flat
      P(s) ∝ s      (linear in speed)    → histogram in s = √(2u) should be linear

    All comparisons are restricted to bins where mask[b] = True.

    Returns
    -------
    dict with keys:
      f_emp_u  : (n_bins,)   empirical density in u (chaotic bins only; 0 elsewhere)
      f_ll_u   : (n_bins,)   flat LL prediction in u
      f_emp_s  : (n_s_bins,) empirical density in s
      f_ll_s   : (n_s_bins,) linear LL prediction in s
      s_centers: (n_s_bins,)
      s_edges  : (n_s_bins+1,)
      l1_u     : float  L1 distance between f_emp_u and f_ll_u (chaotic bins)
      l1_s     : float  L1 distance between f_emp_s and f_ll_s
    """
    n_bins = len(u_centers)
    du = np.diff(u_edges)

    # --- density in u ---
    counts_u = np.zeros(n_bins, dtype=np.int64)
    for i in range(u_traj.shape[0]):
        idx = np.searchsorted(u_edges, u_traj[i], side="right") - 1
        idx = np.clip(idx, 0, n_bins - 1)
        np.add.at(counts_u, idx, 1)

    # Zero out non-chaotic bins, then convert to density
    counts_masked = np.where(mask, counts_u, 0)
    total = counts_masked.sum()
    f_emp_u = np.zeros(n_bins)
    if total > 0:
        f_emp_u = counts_masked / (total * du)

    # LL flat prediction: uniform on chaotic bins
    f_ll_u = np.zeros(n_bins)
    chaotic_width = du[mask].sum()
    if chaotic_width > 0:
        f_ll_u[mask] = 1.0 / chaotic_width

    # L1 in u restricted to chaotic bins
    l1_u = float(np.sum(np.abs(f_emp_u[mask] - f_ll_u[mask]) * du[mask]))

    # --- density in s ---
    s_edges = np.sqrt(2.0 * u_edges)
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    ds = np.diff(s_edges)
    n_s = len(s_centers)

    # Histogram all samples in speed; mark chaotic via their u-bin
    counts_s = np.zeros(n_s, dtype=np.int64)
    counts_s_mask = np.zeros(n_s, dtype=np.int64)
    for i in range(u_traj.shape[0]):
        s_i = np.sqrt(2.0 * u_traj[i])
        b_u = np.searchsorted(u_edges, u_traj[i], side="right") - 1
        b_u = np.clip(b_u, 0, n_bins - 1)
        b_s = np.searchsorted(s_edges, s_i, side="right") - 1
        b_s = np.clip(b_s, 0, n_s - 1)
        np.add.at(counts_s, b_s, 1)
        chaotic_hit = mask[b_u]
        np.add.at(counts_s_mask, b_s, chaotic_hit.astype(int))

    total_s = counts_s_mask.sum()
    f_emp_s = np.zeros(n_s)
    if total_s > 0:
        f_emp_s = counts_s_mask / (total_s * ds)

    # LL linear prediction: P(s) ∝ s on chaotic s-bins
    # Chaotic s-bins: those whose centre corresponds to a chaotic u-bin
    chaotic_s = mask  # same indexing since s_edges = sqrt(2*u_edges)
    f_ll_s = np.zeros(n_s)
    norm_s = np.dot(s_centers[chaotic_s], ds[chaotic_s])
    if norm_s > 0:
        f_ll_s[chaotic_s] = s_centers[chaotic_s] / norm_s

    l1_s = float(np.sum(np.abs(f_emp_s[chaotic_s] - f_ll_s[chaotic_s]) * ds[chaotic_s]))

    return {
        "f_emp_u":   f_emp_u,
        "f_ll_u":    f_ll_u,
        "f_emp_s":   f_emp_s,
        "f_ll_s":    f_ll_s,
        "s_centers": s_centers,
        "s_edges":   s_edges,
        "l1_u":      np.float64(l1_u),
        "l1_s":      np.float64(l1_s),
    }


# ---------------------------------------------------------------------------
# Report figure
# ---------------------------------------------------------------------------

def make_ll_validation_report(
    out_dir,
    u_centers: np.ndarray,
    u_edges: np.ndarray,
    transport: dict,
    landau: dict,
    shapes: dict,
    density: dict,
    mask: np.ndarray,
    out_path=None,
) -> None:
    """2×2 diagnostic figure for the LL validation."""
    out_dir = Path(out_dir)
    if out_path is None:
        out_path = out_dir / "ll_validation.png"

    B        = transport["B"]
    D        = transport["D"]
    D_deriv  = landau["D_deriv"]
    residual = landau["residual"]
    c_D      = float(shapes["c_D"])
    c_B      = float(shapes["c_B"])
    ll_ratio = float(shapes["ll_ratio"])
    D_fit    = shapes["D_fit"]
    B_fit    = shapes["B_fit"]

    chaotic_u   = np.where(mask, u_centers, np.nan)
    nc_u        = np.where(~mask, u_centers, np.nan)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Lichtenberg-Lieberman validation    "
        f"ll_ratio = c_B / (½c_D) = {ll_ratio:.3f}  (LL predicts ≈ 1)",
        fontsize=11,
    )

    # ------------------------------------------------------------------
    # Panel (0,0): D(u) vs u — expect slope-1 log-log line
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    ax.set_title("D(u) = E[(Δu)²]  — expect D ∝ u")
    ax.loglog(nc_u, D, "o", ms=3, color="silver", label="non-chaotic")
    ax.loglog(chaotic_u, D, "o", ms=4, color="tab:blue", label="chaotic")
    ax.loglog(u_centers, D_fit, "--", lw=1.2, color="tab:red",
              label=f"D_fit = {c_D:.3g} · u")
    ax.set_xlabel("u")
    ax.set_ylabel("D(u) [per bounce]")
    ax.legend(fontsize=7)

    # ------------------------------------------------------------------
    # Panel (0,1): B(u) vs u — expect flat; also show ½D'
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    ax.set_title("B(u) = E[Δu]  vs  ½ D'(u)  — expect equal")
    ax.semilogx(nc_u, B, "o", ms=3, color="silver")
    ax.semilogx(chaotic_u, B, "o", ms=4, color="tab:green", label="B emp")
    ax.semilogx(u_centers, 0.5 * D_deriv, "--", lw=1.2, color="tab:orange",
                label="½ D' (num)")
    ax.axhline(c_B, color="tab:red", lw=1.0, ls=":", label=f"B_fit = {c_B:.3g}")
    ax.set_xlabel("u")
    ax.set_ylabel("B(u) [per bounce]")
    ax.legend(fontsize=7)

    # ------------------------------------------------------------------
    # Panel (1,0): Landau residual R = B − ½D'
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    ax.set_title(f"Landau residual R = B − ½D'   (ll_ratio = {ll_ratio:.3f})")
    ax.semilogx(nc_u, np.where(~mask, residual, np.nan), "o", ms=3,
                color="silver")
    ax.semilogx(chaotic_u, np.where(mask, residual, np.nan), "o-", ms=3,
                lw=0.8, color="tab:purple", label="R(u)")
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.set_xlabel("u")
    ax.set_ylabel("B − ½ D'")
    ax.legend(fontsize=7)

    # ------------------------------------------------------------------
    # Panel (1,1): Stationary density comparison
    # ------------------------------------------------------------------
    ax  = axes[1, 1]
    ax2 = ax.twinx()
    ax.set_title(
        f"Stationary density   "
        f"L1(u)={float(density['l1_u']):.3f}  L1(s)={float(density['l1_s']):.3f}"
    )
    # u-density (left axis)
    ax.plot(u_centers, density["f_emp_u"], color="tab:blue", lw=1.2, label="emp P(u)")
    ax.plot(u_centers, density["f_ll_u"], "--", color="tab:blue", lw=0.9, label="LL: flat")
    ax.set_xscale("log")
    ax.set_xlabel("u  (left axis)  /  s  (right axis)")
    ax.set_ylabel("P(u) density", color="tab:blue")
    # s-density (right axis)
    s_c = density["s_centers"]
    ax2.plot(s_c, density["f_emp_s"], color="tab:orange", lw=1.2, label="emp P(s)")
    ax2.plot(s_c, density["f_ll_s"], "--", color="tab:orange", lw=0.9,
             label="LL: ∝ s")
    ax2.set_ylabel("P(s) density", color="tab:orange")
    # combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ll_validation] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run_ll_validation(
    results: dict,
    cfg: dict,
    out_dir,
    verbose: bool = True,
) -> dict:
    """Run the full LL validation suite.

    Parameters
    ----------
    results  : dict returned by run_calibration() — needs u_traj, u_centers, u_edges,
               tau_lag, and optionally entropy_norm (from diagnostics)
    cfg      : calibration config dict
    out_dir  : calibration output directory (diagnostics/ subdir used)
    verbose  : print progress messages

    Returns
    -------
    dict with all computed arrays (also saved to diagnostics/ll_validation.npz)
    """
    out_dir   = Path(out_dir)
    diag_dir  = out_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    u_traj    = results["u_traj"]
    u_centers = results["u_centers"]
    u_edges   = results["u_edges"]
    tau_lag   = results["tau_lag"]

    # ------------------------------------------------------------------
    # Build chaos mask
    # ------------------------------------------------------------------
    mask = None
    chaos_npz = diag_dir / "chaos_mask.npz"
    if chaos_npz.exists():
        try:
            d = np.load(chaos_npz)
            mask = d["mask"].astype(bool)
            if verbose:
                print(f"  Loaded chaos mask from {chaos_npz} "
                      f"({mask.sum()}/{len(mask)} chaotic bins)")
        except Exception:
            pass

    if mask is None:
        # Try to rebuild from entropy_norm in results or mixing_diagnostics.npz
        entropy_norm = None
        mix_npz = diag_dir / "mixing_diagnostics.npz"
        if mix_npz.exists():
            try:
                entropy_norm = np.load(mix_npz)["entropy_norm"]
            except Exception:
                pass

        if entropy_norm is not None:
            threshold = cfg.get("entropy_threshold", 0.8)
            mask = build_chaotic_mask(entropy_norm, tau_lag,
                                      entropy_threshold=threshold)
            if verbose:
                print(f"  Rebuilt chaos mask: {mask.sum()}/{len(mask)} chaotic bins")
        else:
            if verbose:
                print("  [WARN] No chaos mask available; using all bins")
            mask = np.ones(len(u_centers), dtype=bool)

    # ------------------------------------------------------------------
    # Step 1: one-bounce transport coefficients
    # ------------------------------------------------------------------
    if verbose:
        print("  Computing one-bounce transport coefficients ...")
    transport = compute_one_bounce_transport(u_traj, u_edges, mask=mask)
    B = transport["B"]
    D = transport["D"]
    n_valid = np.isfinite(B).sum()
    if verbose:
        print(f"  B, D estimated for {n_valid}/{len(B)} chaotic bins")

    # ------------------------------------------------------------------
    # Step 2: Landau relation check
    # ------------------------------------------------------------------
    if verbose:
        print("  Checking Landau relation B ≈ ½ D' ...")
    landau = check_landau_relation(u_centers, B, D)

    # ------------------------------------------------------------------
    # Step 3: Fit LL shapes
    # ------------------------------------------------------------------
    shapes = fit_ll_shapes(u_centers, B, D, mask)
    if verbose:
        print(f"  c_D = {float(shapes['c_D']):.4g},  "
              f"c_B = {float(shapes['c_B']):.4g},  "
              f"ll_ratio = {float(shapes['ll_ratio']):.4f}")

    # ------------------------------------------------------------------
    # Step 4: Stationary density comparison
    # ------------------------------------------------------------------
    if verbose:
        print("  Comparing stationary densities ...")
    density = compare_stationary_density(u_traj, u_edges, u_centers, mask)
    if verbose:
        print(f"  L1(P_emp vs flat in u) = {float(density['l1_u']):.4f}")
        print(f"  L1(P_emp vs linear in s) = {float(density['l1_s']):.4f}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    np.savez_compressed(
        diag_dir / "ll_validation.npz",
        B=B, D=D, counts=transport["counts"],
        D_deriv=landau["D_deriv"],
        residual=landau["residual"],
        c_D=shapes["c_D"],
        c_B=shapes["c_B"],
        ll_ratio=shapes["ll_ratio"],
        D_fit=shapes["D_fit"],
        B_fit=shapes["B_fit"],
        f_emp_u=density["f_emp_u"],
        f_ll_u=density["f_ll_u"],
        f_emp_s=density["f_emp_s"],
        f_ll_s=density["f_ll_s"],
        s_centers=density["s_centers"],
        l1_u=density["l1_u"],
        l1_s=density["l1_s"],
        mask=mask,
    )
    if verbose:
        print(f"  Saved ll_validation.npz")

    # ------------------------------------------------------------------
    # Report figure
    # ------------------------------------------------------------------
    make_ll_validation_report(
        diag_dir, u_centers, u_edges,
        transport, landau, shapes, density, mask,
    )

    out = {
        "transport": transport,
        "landau": landau,
        "shapes": shapes,
        "density": density,
        "mask": mask,
    }
    return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) < 2:
        print("Usage: python3 ll_validation.py <out_dir>")
        _sys.exit(1)

    _out_dir = Path(_sys.argv[1])
    _cal_npz = _out_dir / "calibration_results.npz"
    if not _cal_npz.exists():
        print(f"calibration_results.npz not found in {_out_dir}")
        _sys.exit(1)

    _cal = np.load(_cal_npz)
    _results = {k: _cal[k] for k in _cal.files}

    # load u_traj from diag_trajectories.npz if not in calibration_results.npz
    if "u_traj" not in _results:
        _diag_traj = _out_dir / "diag_trajectories.npz"
        if _diag_traj.exists():
            _dt = np.load(_diag_traj)
            _results["u_traj"] = _dt["u_traj"]
            _results["psi_traj"] = _dt["psi_traj"]
        else:
            print("u_traj not found; run with full calibration_results.npz")
            _sys.exit(1)

    from trajectories import load_cal_config
    _cfg = load_cal_config()
    run_ll_validation(_results, _cfg, _out_dir, verbose=True)
