"""Constants and hyper-parameters (proposal Sec. 1 & 3)."""
from dataclasses import dataclass
import numpy as np

# 11 DIN EN ISO 14688-1 sieve sizes [mm]
SIEVES = np.array([0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2, 6.3, 20, 63, 200], dtype=float)
LOG_WIDTHS = np.diff(np.log10(SIEVES))  # 10 widths for Eq.(1)


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
