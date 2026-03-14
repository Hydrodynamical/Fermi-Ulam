"""Entry point: run one experiment and save results + plots.

Usage:
  python main.py                          # uses base config
  python main.py --config ../config/experiment_heating.yaml
  python main.py --out-dir ../results/run1
"""
import argparse
import sys
from pathlib import Path
import numpy as np

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config
from simulator import run
from plotting import plot_energy, plot_mixing_fraction, plot_velocity_hist, plot_phase_space


def parse_args():
    p = argparse.ArgumentParser(description="1D PDMP wall-heating simulator")
    p.add_argument("--config", default=None, help="Override YAML config path")
    p.add_argument("--out-dir", default="../results", help="Output directory")
    p.add_argument("--no-plots", action="store_true", help="Skip plotting")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    print("Config:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print()

    print("Running simulation...")
    data = run(cfg, verbose=True)

    # Save data
    npz_path = out_dir / "results.npz"
    save_dict = {k: v for k, v in data.items()
                 if k not in ("particles_final", "cfg")}
    save_dict.update({f"final_{k}": v
                      for k, v in data["particles_final"].items()})
    np.savez(npz_path, **save_dict)
    print(f"\nSaved results to {npz_path}")

    if not args.no_plots:
        print("Plotting...")
        plot_energy(data, out_path=out_dir / "energy.png")
        plot_mixing_fraction(data, out_path=out_dir / "mixing_fraction.png")
        v_final = data["particles_final"]["v"]
        t_final = data["t"][-1]
        plot_velocity_hist(v_final, t_final, out_path=out_dir / "velocity_hist.png")

        # Phase space snapshot at final time — need a particles-like object
        class _P:
            pass
        pf = _P()
        pf.x = data["particles_final"]["x"]
        pf.v = data["particles_final"]["v"]
        pf.sigma = data["particles_final"]["sigma"]
        plot_phase_space(pf, t_final, cfg["L"], out_path=out_dir / "phase_space.png")
        print(f"Plots saved to {out_dir}/")

    print("Done.")


if __name__ == "__main__":
    main()
