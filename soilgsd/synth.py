"""Synthetic soils so the pipeline is runnable/testable without Kaggle data.
Coarse grains = coloured ellipses (lognormal sizes); fines = speckle noise. Labels are exact."""
import numpy as np
import cv2
from .config import SIEVES
from .pipeline import Sample
from .metrics import clean_curve


def make_soil(rng, rho=4.6, size=900, fine_frac=None):
    fine_frac = rng.uniform(0.2, 0.85) if fine_frac is None else fine_frac
    img = np.full((size, size, 3), (205, 200, 190), np.uint8)
    img = np.clip(img + rng.normal(0, 6 + 20 * fine_frac, img.shape), 0, 255).astype(np.uint8)
    n, mu = rng.integers(8, 25), rng.uniform(np.log(8), np.log(25))
    grains = []
    for _ in range(n):
        d = float(np.clip(rng.lognormal(mu, 0.45), 7, 60))           # b [mm]
        a = d * rng.uniform(1.0, 1.4)
        c = rng.uniform(0.15, 0.85, 2) * size
        col = tuple(int(v) for v in rng.integers(60, 140, 3))
        cv2.ellipse(img, (int(c[0]), int(c[1])), (int(a * rho / 2), int(d * rho / 2)),
                    rng.uniform(0, 180), 0, 360, col, -1)
        grains.append((a, d))
    m = np.array([a * d ** 3 for a, d in grains]); m = m / m.sum() * (1 - fine_frac)
    # fines split evenly over classes below 2 mm (classes 1..6), so truth is known
    pi = np.zeros(11)
    pi[:7] = fine_frac / 7
    for (a, d), mm in zip(grains, m):
        pi[np.searchsorted(SIEVES, d)] += mm
    return img, clean_curve(100 * np.cumsum(pi))


def make_dataset(n=12, rho=4.6, seed=0):
    rng = np.random.default_rng(seed)
    samples, curves = [], []
    for i in range(n):
        img, c = make_soil(rng, rho)
        samples.append(Sample(f"syn{i:02d}", [(img, rho)])); curves.append(c)
    return samples, np.stack(curves)
