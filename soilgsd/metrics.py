import numpy as np
from .config import LOG_WIDTHS


def emd(p_true, p_pred) -> float:
    """Eq. (1): log-weighted EMD between cumulative curves (11 values each)."""
    d = np.abs(np.asarray(p_true, float) - np.asarray(p_pred, float))[:10]
    return float((d * LOG_WIDTHS).sum())


def clean_curve(p) -> np.ndarray:
    """Force [0,100], monotone, 100 at 200 mm."""
    p = np.clip(np.asarray(p, float), 0, 100)
    p = np.maximum.accumulate(p)
    p[-1] = 100.0
    return p
