"""Path C: evidence-weighted stitching (Eq. 5)."""
import numpy as np
from .config import SIEVES, Config
from .metrics import clean_curve


def blend(pB: np.ndarray, coarse, dc: float, cfg: Config) -> np.ndarray:
    """pB: Path B curve (11,), coarse: CoarseResult, dc: crossover [mm]."""
    if not np.isfinite(dc) or cfg.blend_mode == "pure_b":
        return clean_curve(pB)
    coarse = coarse.above(dc)                  # only grains resolved at the crossover count (n_eff too)
    if coarse.mass.size == 0:
        return clean_curve(pB)
    i_c = int(np.where(SIEVES == dc)[0][0])
    F = pB[i_c]
    Gc = coarse.G(dc)
    GB = (pB - F) / max(100.0 - F, 1e-6)
    w = 1.0 if cfg.blend_mode == "hard" else coarse.n_eff / (coarse.n_eff + cfg.kappa)
    out = pB.copy()
    out[i_c:] = F + (100.0 - F) * (w * Gc[i_c:] + (1 - w) * GB[i_c:])
    return clean_curve(out)
