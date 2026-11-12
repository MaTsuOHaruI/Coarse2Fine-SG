import numpy as np
from .config import LOG_WIDTHS


def emd(p_true, p_pred, form: str = "left") -> float:
    """Eq. (1): log-weighted EMD between cumulative curves (11 values each). Lower is better, range [0, 500].

    form: how |difference| is integrated over log10(size).
        "left" uses the value at the left sieve of each interval (default; reproduces the
        organisers' reference scores 102.0 and 89.9), "right" the right sieve,
        "trapezoid" the average of both.
    """
    d = np.abs(np.asarray(p_true, float) - np.asarray(p_pred, float))
    if form == "left":
        h = d[:10]
    elif form == "right":
        h = d[1:]
    elif form == "trapezoid":
        h = 0.5 * (d[:-1] + d[1:])
    else:
        raise ValueError(f"unknown form: {form}")
    return float((h * LOG_WIDTHS).sum())


def mean_emd(P_true, P_pred, form: str = "left") -> float:
    """Mean EMD over several soils. In: two (n_soils, 11) arrays in percent."""
    P_true, P_pred = np.atleast_2d(np.asarray(P_true, float)), np.atleast_2d(np.asarray(P_pred, float))
    return float(np.mean([emd(t, p, form) for t, p in zip(P_true, P_pred)]))


def clean_curve(p) -> np.ndarray:
    """Force [0,100], monotone, 100 at 200 mm."""
    p = np.clip(np.asarray(p, float), 0, 100)
    p = np.maximum.accumulate(p)
    p[-1] = 100.0
    return p
