"""Constants, hyper-parameters (proposal Sec. 1 & 3) and run settings.

Hyper-parameter defaults live in the `Config` dataclass. `configs/default.yaml` can
override them (see `Config.from_yaml`). Paths come from the YAML file and can be
overridden by environment variables, so the same code runs on a laptop and on Colab:
    SOILGSD_DATA_DIR    folder with the Kaggle files
    SOILGSD_CACHE_DIR   folder for SAM masks, embeddings, features
"""
import argparse
import datetime
import os
import random
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import yaml

# 11 DIN EN ISO 14688-1 sieve sizes [mm], smallest first. Curves are always 11 values in this order.
SIEVES = np.array([0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2, 6.3, 20, 63, 200], dtype=float)
LOG_WIDTHS = np.diff(np.log10(SIEVES))  # 10 widths for Eq.(1)
N_SIEVES = len(SIEVES)

# Same sieves as spelled in the Kaggle CSV headers, and the ID column name.
SIEVE_COLUMNS = ["0.002", "0.0063", "0.02", "0.063", "0.2", "0.63", "2", "6.3", "20", "63", "200"]
ID_COLUMN = "sample_id"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


@dataclass
class Config:
    tau_px: float = 25.0          # SAM-reliable grain width in px  (Eq. 2)
    mass_p: float = 2.0           # m_g ∝ a_g * b_g^p               (Path A)
    kappa: float = 3.0            # w = n_eff / (n_eff + kappa)     (Eq. 5)
    min_solidity: float = 0.85    # drop clumps / touching grains
    target_rho: float = 4.6       # Path B resampling scale [px/mm]
    feat_min_mm: float = 6.3      # Path B coarse stats use grains >= this
    n_pca: int = 3                # r in Eq. 4
    ridge_alpha: float = 10.0
    tray_crop_frac: float = 0.9   # MVP stand-in for tray detection (center crop)
    max_side: int = 4096          # downscale very large photos (rho is rescaled accordingly)
    blend_mode: str = "adaptive"  # "adaptive" | "pure_b"(w=0) | "hard"(w=1)

    # YAML key (section, name) for each field; fields missing from the YAML keep their default.
    YAML_KEYS = {
        "tau_px": ("preprocess", "tau_px"),
        "tray_crop_frac": ("preprocess", "tray_crop_frac"),
        "max_side": ("preprocess", "max_side"),
        "mass_p": ("path_a", "mass_exponent_p"),
        "min_solidity": ("path_a", "solidity_min"),
        "feat_min_mm": ("path_a", "coarse_stats_floor_mm"),
        "target_rho": ("path_b", "working_scale_px_per_mm"),
        "n_pca": ("path_b", "pca_rank_r"),
        "ridge_alpha": ("path_b", "ridge_alpha"),
        "kappa": ("stitch", "kappa"),
        "blend_mode": ("stitch", "blend_mode"),
    }

    @classmethod
    def from_yaml(cls, cfg: dict) -> "Config":
        """Build a Config from a dict returned by load_config()."""
        values = {}
        for f in fields(cls):
            section, name = cls.YAML_KEYS[f.name]
            if name in (cfg.get(section) or {}):
                values[f.name] = f.type(cfg[section][name]) if isinstance(f.type, type) else cfg[section][name]
        return cls(**values)


def load_config(path=None) -> dict:
    """Read a YAML config into a dict. Uses configs/default.yaml when path is None."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg


def _resolve(path_text) -> Path:
    """Relative paths are taken from the repository root."""
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def kaggle_dir(cfg) -> Path:
    return _resolve(os.environ.get("SOILGSD_DATA_DIR") or cfg["paths"]["kaggle_dir"])


def cache_dir(cfg) -> Path:
    return _resolve(os.environ.get("SOILGSD_CACHE_DIR") or cfg["paths"]["cache_dir"])


def new_run_dir(cfg, name) -> Path:
    """Create and return runs/<date>_<time>_<name>/ for a script's outputs."""
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _resolve(cfg["paths"]["runs_dir"]) / f"{stamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def set_seed(seed):
    """Fix the random seeds (Python, NumPy, and torch when it is installed)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def pick_device(requested="auto") -> str:
    """"auto" picks cuda when a GPU is visible."""
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def base_parser(description) -> argparse.ArgumentParser:
    """Argument parser with the flags every script shares:
    --config  YAML file (default: configs/default.yaml)
    --device  auto, cuda or cpu (only matters for SAM / DINOv2 steps)
    --limit   process only the first N items, for a quick local check
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--limit", type=int, default=None, help="only the first N items")
    return parser
