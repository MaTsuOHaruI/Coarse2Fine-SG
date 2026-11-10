import numpy as np
from .config import SIEVES


def adaptive_crossover(rho: float, tau: float = 25.0) -> float:
    """Eq. (2): smallest sieve x_i with rho * x_i >= tau. inf if none resolvable."""
    ok = SIEVES[rho * SIEVES >= tau]
    return float(ok.min()) if ok.size else float("inf")


def compute_rho(px_per_mm_fullres: float, full_width: int, released_width: int) -> float:
    """True scale of a released (rescaled) photo."""
    return px_per_mm_fullres * released_width / full_width
