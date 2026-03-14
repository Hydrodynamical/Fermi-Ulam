"""Load and merge YAML config files."""
import yaml
from pathlib import Path


_BASE = Path(__file__).parent.parent / "config" / "base.yaml"


def load_config(override_path=None) -> dict:
    with open(_BASE) as f:
        cfg = yaml.safe_load(f)
    if override_path is not None:
        with open(override_path) as f:
            override = yaml.safe_load(f)
        cfg.update(override)
    return cfg
