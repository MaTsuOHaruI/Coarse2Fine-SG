import numpy as np
from .config import SIEVES


def adaptive_crossover(rho: float, tau: float = 25.0) -> float:
    """Eq. (2): smallest sieve x_i with rho * x_i >= tau. inf if none resolvable."""
    ok = SIEVES[rho * SIEVES >= tau]
    return float(ok.min()) if ok.size else float("inf")


def soil_crossover(rhos, tau: float = 25.0) -> float:
    """Crossover for one soil whose photos may have different scales (H374):
    the largest d_c among its photos, so every photo can resolve it."""
    return max(adaptive_crossover(r, tau) for r in rhos)


def crossover_index(dc: float) -> int:
    """Index of the crossover sieve in SIEVES."""
    return int(np.where(SIEVES == dc)[0][0])


def compute_rho(px_per_mm_fullres: float, full_width: int, released_width: int) -> float:
    """True scale of a released (rescaled) photo, by width. See true_scale for rotated photos."""
    return px_per_mm_fullres * released_width / full_width


def true_scale(px_per_mm_fullres: float, released_size_px, full_size_px) -> float:
    """True scale of a released photo in px/mm. Long side is compared with long side,
    so portrait and landscape photos give the same answer (used by data.load_kaggle)."""
    return px_per_mm_fullres * max(released_size_px) / max(full_size_px)
