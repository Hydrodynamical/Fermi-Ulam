"""Non-interacting calibration experiment: extract L_wall from the Fermi-Ulam map.

Usage (from repo root)
----------------------
    python calibration/run_calibration.py
    python calibration/run_calibration.py --out-dir results/calibration
    python calibration/run_calibration.py --config config/calibration_fast.yaml
    python calibration/run_calibration.py --quiet

Pipeline
--------
1. Generate trajectories via the exact discrete Fermi-Ulam map.
2. Compute phase ACF C(k; u) per energy bin  ->  tau_lag(u)  ->  m(u).
3. Estimate Kramers-Moyal coefficients b(u), a(u) at coarse lag m(u).
4. Solve FP steady-state f_FP(u) analytically from b and a.
5. Compare f_FP to the empirical histogram f_emp(u).
6. Save calibration_results.npz and five diagnostic plots.

Outputs (in --out-dir)
----------------------
  calibration_results.npz       all arrays (u_centers, C_acf, tau_lag, m_arr,
                                 b, a, f_fp, f_emp, ...)
  diag_trajectories.npz         raw (u, psi) for n_diag_particles particles
  poincare_section.png
  phase_acf.png
  tau_mix.png
  km_coefficients.png
  fp_comparison.png

Scientific purpose
------------------
The outputs b(u), a(u), tau_lag(u) are the intrinsic wall-heating coefficients
that later parameterise the PDMP wall operator L_wall in the two-population
model  f = f^0 + f^1.  This experiment is the calibration step that must
precede the interacting (collision) experiments.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running as a script from any working directory.
_CALIB = Path(__file__).parent
_REPO = _CALIB.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_CALIB))   # calibration/ must come before src/ to shadow name conflicts

from trajectories import load_cal_config, make_energy_bins, generate_trajectories
from phase_mixing import compute_phase_acf, estimate_tau_mix_from_acf, choose_coarse_lag
from kramers_moyal import estimate_km_coefficients
from fokker_planck import (
    steady_state_fp,
    empirical_energy_density,
    compare_fp_to_empirical,
)
from plotting import (
    plot_poincare_section,
    plot_phase_acf,
    plot_tau_mix,
    plot_km_coefficients,
    plot_fp_comparison,
)


def run_calibration(cfg: dict, out_dir: Path, verbose: bool = True) -> dict:
    """Execute the full calibration pipeline.

    Parameters
    ----------
    cfg     : merged config dict (from load_cal_config)
    out_dir : directory for outputs (created if absent)
    verbose : print progress messages

    Returns
    -------
    dict with keys: u_centers, C_acf, tau_lag, m_arr, b, a, f_fp, f_emp, metrics
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Trajectories
    # ------------------------------------------------------------------
    if verbose:
        print("\n[1/5] Generating trajectories ...")
    data = generate_trajectories(cfg)
    u_traj = data["u_traj"]
    psi_traj = data["psi_traj"]

    u_edges, u_centers = make_energy_bins(cfg["u_min"], cfg["u_max"], cfg["n_u_bins"])

    if cfg.get("save_diagnostics", True):
        n_d = min(cfg.get("n_diag_particles", 10), u_traj.shape[0])
        np.savez_compressed(
            out_dir / "diag_trajectories.npz",
            u_traj=u_traj[:n_d],
            psi_traj=psi_traj[:n_d],
        )
        if verbose:
            print(f"  Saved {n_d} diagnostic trajectories.")

    # ------------------------------------------------------------------
    # Step 2: Phase mixing time
    # ------------------------------------------------------------------
    if verbose:
        print("[2/5] Computing phase ACF ...")
    C, counts = compute_phase_acf(
        u_traj, psi_traj, u_edges,
        max_lag=cfg["acf_max_lag"],
    )
    tau_lag = estimate_tau_mix_from_acf(
        C, counts,
        threshold=cfg["acf_threshold"],
        n_min=cfg["n_km_samples_min"],
    )
    m_arr = choose_coarse_lag(
        tau_lag,
        safety=cfg["m_safety_factor"],
        m_min=cfg["m_min"],
    )
    if verbose:
        finite = np.isfinite(tau_lag)
        n_finite = finite.sum()
        print(
            f"  tau_lag: {n_finite}/{len(tau_lag)} bins resolved; "
            f"range [{np.nanmin(tau_lag):.1f}, {np.nanmax(tau_lag):.1f}] bounces"
        )
        print(f"  m range: [{m_arr.min()}, {m_arr.max()}]")

    # ------------------------------------------------------------------
    # Step 3: Kramers-Moyal coefficients
    # ------------------------------------------------------------------
    if verbose:
        print("[3/5] Estimating Kramers-Moyal coefficients ...")
    km = estimate_km_coefficients(
        u_traj, u_edges, m_arr,
        n_min=cfg["n_km_samples_min"],
    )
    b, a = km["b"], km["a"]
    n_good = np.isfinite(b).sum()
    if verbose:
        print(f"  b/a estimated for {n_good}/{len(b)} bins")

    # ------------------------------------------------------------------
    # Step 4: FP steady-state
    # ------------------------------------------------------------------
    if verbose:
        print("[4/5] Solving FP steady-state ...")
    f_fp = steady_state_fp(u_centers, b, a)
    f_emp = empirical_energy_density(u_traj, u_edges)

    # ------------------------------------------------------------------
    # Step 5: Comparison
    # ------------------------------------------------------------------
    metrics = compare_fp_to_empirical(u_centers, f_fp, f_emp)
    if verbose:
        print(
            f"  FP vs empirical:  L1 = {metrics['l1_error']:.4f}, "
            f"L2 = {metrics['l2_error']:.4f}, "
            f"KL = {metrics['kl_divergence']:.5f}"
        )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    np.savez_compressed(
        out_dir / "calibration_results.npz",
        u_centers=u_centers,
        u_edges=u_edges,
        C_acf=C,
        counts_acf=counts,
        tau_lag=tau_lag,
        m_arr=m_arr,
        b=b,
        a=a,
        km_counts=km["counts"],
        f_fp=f_fp,
        f_emp=f_emp,
    )
    if verbose:
        print(f"  Saved calibration_results.npz to {out_dir}/")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    if verbose:
        print("[5/5] Generating plots ...")

    plot_poincare_section(
        u_traj, psi_traj, n_particles=5,
        out_path=out_dir / "poincare_section.png",
    )
    plot_phase_acf(
        C, u_centers,
        out_path=out_dir / "phase_acf.png",
    )
    plot_tau_mix(
        u_centers, tau_lag, m_arr,
        out_path=out_dir / "tau_mix.png",
    )
    plot_km_coefficients(
        u_centers, b, a,
        out_path=out_dir / "km_coefficients.png",
    )
    plot_fp_comparison(
        u_centers, f_fp, f_emp, metrics,
        out_path=out_dir / "fp_comparison.png",
    )

    if verbose:
        print(f"\nAll outputs written to: {out_dir}/")

    return {
        "u_centers": u_centers,
        "u_edges": u_edges,
        "u_traj": u_traj,
        "psi_traj": psi_traj,
        "C_acf": C,
        "tau_lag": tau_lag,
        "m_arr": m_arr,
        "b": b,
        "a": a,
        "f_fp": f_fp,
        "f_emp": f_emp,
        "metrics": metrics,
    }


def run_diagnostics(results: dict, cfg: dict, out_dir: Path, verbose: bool = True) -> None:
    """Run the second-pass diagnostic layer over calibration trajectory data.

    Creates out_dir/diagnostics/ and writes 5 .npz files plus a diagnostic_report.png.
    Each module is called inside its own try/except so one failure does not abort the rest.

    Parameters
    ----------
    results : dict returned by run_calibration() — must contain u_traj, psi_traj, etc.
    cfg     : merged config dict
    out_dir : calibration output directory (diagnostics/ created inside)
    verbose : print progress messages
    """
    diag_dir = out_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    u_traj    = results["u_traj"]
    psi_traj  = results["psi_traj"]
    u_centers = results["u_centers"]
    u_edges   = results["u_edges"]
    m_arr     = results["m_arr"]
    tau_lag   = results["tau_lag"]
    b         = results["b"]
    a         = results["a"]
    f_fp      = results["f_fp"]
    f_emp     = results["f_emp"]

    n_psi_unif = cfg.get("n_psi_bins_uniformity", 36)
    n_psi_heat = cfg.get("n_psi_bins_heatmap", 12)
    lag_values = tuple(cfg.get("variance_lag_values", [1, 2, 4, 8, 16, 32, 64]))
    eta_values = tuple(cfg.get("small_jump_eta_values", [0.05, 0.1, 0.2, 0.5]))
    entropy_threshold = cfg.get("entropy_threshold", 0.8)
    run_bootstrap = cfg.get("run_bootstrap", False)
    n_bootstrap = cfg.get("n_bootstrap", 50)

    if verbose:
        print("\n[Diagnostics] Writing to", diag_dir)

    # ------------------------------------------------------------------
    # 1. Mixing diagnostics
    # ------------------------------------------------------------------
    try:
        from mixing_diagnostics import (
            compute_integrated_acf_time,
            compute_phase_uniformity,
            compute_phase_entropy,
        )
        if verbose:
            print("[Diag 1/8] Mixing diagnostics ...")
        tau_int = compute_integrated_acf_time(results["C_acf"])
        unif = compute_phase_uniformity(
            u_traj, psi_traj, u_edges, m_arr, n_psi_bins=n_psi_unif,
        )
        ent = compute_phase_entropy(
            u_traj, psi_traj, u_edges, n_psi_bins=n_psi_unif,
        )
        np.savez_compressed(
            diag_dir / "mixing_diagnostics.npz",
            tau_int=tau_int,
            tv_distance=unif["tv_distance"],
            circular_modes=unif["circular_modes"],
            entropy_norm=ent["entropy_norm"],
            counts_uniform=unif["counts"],
            counts_entropy=ent["counts"],
        )
        if verbose:
            print("  Saved mixing_diagnostics.npz")
    except Exception as exc:
        print(f"  [WARN] mixing_diagnostics failed: {exc}")
        tau_int = None
        ent = {"entropy_norm": np.full(len(u_centers), np.nan)}

    # ------------------------------------------------------------------
    # 2. Increment diagnostics
    # ------------------------------------------------------------------
    try:
        from increment_diagnostics import (
            compute_increment_moments,
            compute_small_jump_ratios,
            compute_variance_vs_lag,
        )
        if verbose:
            print("[Diag 2/8] Increment diagnostics ...")
        incr = compute_increment_moments(u_traj, u_edges, m_arr)
        sjr  = compute_small_jump_ratios(u_traj, u_edges, m_arr, eta_values=eta_values)
        vvl  = compute_variance_vs_lag(u_traj, u_edges, lag_values=lag_values)
        np.savez_compressed(
            diag_dir / "increment_diagnostics.npz",
            # Per-bin increment statistics at the coarse lag
            mean_u=incr["mean_u"],
            moment_var_u=incr["var_u"],
            skew_u=incr["skew_u"],
            kurt_u=incr["kurt_u"],
            mean_logu=incr["mean_logu"],
            moment_var_logu=incr["var_logu"],
            skew_logu=incr["skew_logu"],
            kurt_logu=incr["kurt_logu"],
            quantiles_u=incr["quantiles_u"],
            quantiles_logu=incr["quantiles_logu"],
            # Small-jump ratios
            ratios=sjr["ratios"],
            eta_values=sjr["eta_values"],
            # Variance vs lag  (n_bins, n_lags) — used by _panel_var_vs_lag
            var_u=vvl["var_u"],
            var_logu=vvl["var_logu"],
            lag_values=vvl["lag_values"],
        )
        if verbose:
            print("  Saved increment_diagnostics.npz")
    except Exception as exc:
        print(f"  [WARN] increment_diagnostics failed: {exc}")

    # ------------------------------------------------------------------
    # 3. Markov tests
    # ------------------------------------------------------------------
    try:
        from markov_tests import (
            compute_phase_conditioned_moments,
            test_semigroup_consistency,
            test_lag1_autocorrelation,
        )
        if verbose:
            print("[Diag 3/8] Markov tests ...")
        pcm  = compute_phase_conditioned_moments(
            u_traj, psi_traj, u_edges, m_arr, n_psi_bins=n_psi_heat,
        )
        semi = test_semigroup_consistency(u_traj, u_edges, m_arr)
        acorr = test_lag1_autocorrelation(u_traj, u_edges, m_arr)
        np.savez_compressed(
            diag_dir / "markov_tests.npz",
            drift_map=pcm["drift_map"],
            var_map=pcm["var_map"],
            counts_map=pcm["counts_map"],
            psi_edges=pcm["psi_edges"],
            drift_ratio=semi["drift_ratio"],
            var_ratio=semi["var_ratio"],
            lag1_autocorr=acorr["lag1_autocorr"],
        )
        if verbose:
            print("  Saved markov_tests.npz")
    except Exception as exc:
        print(f"  [WARN] markov_tests failed: {exc}")

    # ------------------------------------------------------------------
    # 4. Chaos mask + masked FP
    # ------------------------------------------------------------------
    try:
        from chaos_mask import build_chaotic_mask, estimate_km_masked, compute_fp_masked
        if verbose:
            print("[Diag 4/8] Chaos mask ...")
        entropy_norm = ent["entropy_norm"]
        mask = build_chaotic_mask(
            entropy_norm, tau_lag, entropy_threshold=entropy_threshold,
        )
        km_m = estimate_km_masked(u_traj, u_edges, m_arr, mask)
        fp_m = compute_fp_masked(u_centers, km_m["b_masked"], km_m["a_masked"], f_emp)
        np.savez_compressed(
            diag_dir / "chaos_mask.npz",
            entropy_norm=entropy_norm,
            mask=mask,
            b_masked=km_m["b_masked"],
            a_masked=km_m["a_masked"],
            counts_masked=km_m["counts_masked"],
            f_fp_masked=fp_m["f_fp_masked"],
            entropy_threshold=np.float64(entropy_threshold),
        )
        if verbose:
            n_chaotic = int(mask.sum())
            print(f"  {n_chaotic}/{len(mask)} bins marked chaotic; saved chaos_mask.npz")
    except Exception as exc:
        print(f"  [WARN] chaos_mask failed: {exc}")

    # ------------------------------------------------------------------
    # 5. FP diagnostics (KS distance, log-residual, forward validation, bootstrap)
    # ------------------------------------------------------------------
    try:
        from fp_diagnostics import (
            compute_ks_distance,
            compute_log_density_residual,
            forward_validate_fp,
        )
        if verbose:
            print("[Diag 5/8] FP diagnostics ...")
        ksd = compute_ks_distance(u_centers, f_fp, f_emp, u_edges)
        ldr = compute_log_density_residual(u_centers, f_fp, f_emp)
        fwd = forward_validate_fp(u_centers, u_edges, b, a, f_emp, n_steps=10)

        save_kwargs = dict(
            ks_stat=np.float64(ksd["ks_stat"]),
            ks_u=np.float64(ksd["ks_u"]),
            cdf_fp=ksd["cdf_fp"],
            cdf_emp=ksd["cdf_emp"],
            residual=ldr["residual"],
            step_l1_to_fp=fwd["step_l1_to_fp"],
            step_l1_to_emp=fwd["step_l1_to_emp"],
        )

        if run_bootstrap:
            from fp_diagnostics import bootstrap_km_confidence
            if verbose:
                print(f"  Running bootstrap CI (n={n_bootstrap}) ...")
            bci = bootstrap_km_confidence(
                u_traj, u_edges, m_arr, n_bootstrap=n_bootstrap,
            )
            save_kwargs["b_ci"] = bci["b_ci"]
            save_kwargs["a_ci"] = bci["a_ci"]

        np.savez_compressed(diag_dir / "fp_diagnostics.npz", **save_kwargs)
        if verbose:
            print(f"  KS stat = {ksd['ks_stat']:.4f} at u = {ksd['ks_u']:.3f}")
            print("  Saved fp_diagnostics.npz")
    except Exception as exc:
        print(f"  [WARN] fp_diagnostics failed: {exc}")

    # ------------------------------------------------------------------
    # 6. Diagnostic report (8-panel figure)
    # ------------------------------------------------------------------
    try:
        from diagnostic_report import make_diagnostic_report
        if verbose:
            print("[Diag 6/8] Generating diagnostic_report.png ...")
        make_diagnostic_report(diag_dir, u_centers, u_edges)
    except Exception as exc:
        print(f"  [WARN] diagnostic_report failed: {exc}")

    # ------------------------------------------------------------------
    # 7. Lichtenberg-Lieberman validation
    # ------------------------------------------------------------------
    try:
        from ll_validation import run_ll_validation
        if verbose:
            print("[Diag 7/8] Lichtenberg-Lieberman validation ...")
        run_ll_validation(results, cfg, out_dir, verbose=verbose)
    except Exception as exc:
        print(f"  [WARN] ll_validation failed: {exc}")

    # ------------------------------------------------------------------
    # 8. Residual waiting-time law into binwise Markov proxy
    # ------------------------------------------------------------------
    try:
        from first_passage import run_first_passage
        if verbose:
            print("[Diag 8/8] Residual waiting-time law into binwise Markov proxy ...")
        run_first_passage(results, cfg, out_dir, verbose=verbose)
    except Exception as exc:
        print(f"  [WARN] first_passage failed: {exc}")

    if verbose:
        print(f"[Diagnostics] Done. All outputs in {diag_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Calibration experiment: extract wall operator from Fermi-Ulam map"
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Override YAML config path (merged on top of calibration_base.yaml)",
    )
    parser.add_argument(
        "--out-dir",
        default="results/calibration",
        metavar="DIR",
        help="Output directory for results and plots (default: results/calibration)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Skip the extended diagnostic pass (faster for quick checks)",
    )
    args = parser.parse_args()

    cfg = load_cal_config(args.config)

    if not args.quiet:
        print("=== Calibration experiment ===")
        print("Config:")
        for k, v in sorted(cfg.items()):
            print(f"  {k}: {v}")

    out_dir = Path(args.out_dir)
    results = run_calibration(cfg, out_dir, verbose=not args.quiet)

    print("\nFinal metrics:")
    for k, v in results["metrics"].items():
        print(f"  {k}: {v:.5f}")

    run_diag = cfg.get("run_diagnostics", True) and not args.no_diagnostics
    if run_diag:
        run_diagnostics(results, cfg, out_dir, verbose=not args.quiet)


if __name__ == "__main__":
    main()
