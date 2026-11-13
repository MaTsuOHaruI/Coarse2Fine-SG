"""Path A: training-free measurement of the resolvable (coarse) fraction."""
from dataclasses import dataclass, field
import numpy as np
import cv2
from .config import SIEVES, Config


def grain_axes(mask: np.ndarray):
    """Return (a_px, b_px, solidity): max / min Feret diameter and solidity."""
    m = np.pad(mask.astype(np.uint8), 1)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return 0.0, 0.0, 0.0
    c = max(cs, key=cv2.contourArea)
    hull = cv2.convexHull(c)[:, 0, :].astype(float)
    ang = np.deg2rad(np.arange(0, 180, 3))
    proj = hull @ np.stack([np.cos(ang), np.sin(ang)])          # (n_pts, n_ang)
    w = proj.max(0) - proj.min(0) + 1.0
    harea = cv2.contourArea(hull.astype(np.float32))
    sol = float(mask.sum() / harea) if harea > 0 else 0.0
    return float(w.max()), float(w.min()), min(sol, 1.0)


@dataclass
class CoarseResult:
    b_mm: np.ndarray = field(default_factory=lambda: np.zeros(0))
    mass: np.ndarray = field(default_factory=lambda: np.zeros(0))
    area_mm2: np.ndarray = field(default_factory=lambda: np.zeros(0))
    tray_mm2: float = 0.0

    def merge(self, o: "CoarseResult") -> "CoarseResult":
        return CoarseResult(np.r_[self.b_mm, o.b_mm], np.r_[self.mass, o.mass],
                            np.r_[self.area_mm2, o.area_mm2], self.tray_mm2 + o.tray_mm2)

    def above(self, dc: float) -> "CoarseResult":
        """Only the grains with b >= dc (the ones the crossover says are resolved on every photo)."""
        k = self.b_mm >= dc
        return CoarseResult(self.b_mm[k], self.mass[k], self.area_mm2[k], self.tray_mm2)

    @property
    def n_eff(self) -> float:
        m = self.mass
        return float(m.sum() ** 2 / (m ** 2).sum()) if m.size else 0.0

    def G(self, dc: float) -> np.ndarray:
        """Eq. (3): cumulative mass fraction over sieves >= dc (nan below).
        Pools only grains with b >= dc, so G(dc) is ~0 and G(200) = 1. (Fix: the MVP also pooled
        grains between tau/rho and dc, e.g. 5.4 to 6.3 mm on training photos.)"""
        out = np.full(len(SIEVES), np.nan)
        if not np.isfinite(dc):
            return out
        c = self.above(dc)
        if not c.mass.size:
            return out
        tot = c.mass.sum()
        for i, x in enumerate(SIEVES):
            if x >= dc:
                out[i] = c.mass[c.b_mm <= x].sum() / tot
        return out

    def stats(self, min_mm: float):
        """Path B input (iv): count, tray coverage, coarse D50 -- grains >= min_mm only."""
        k = self.b_mm >= min_mm
        n = int(k.sum())
        cov = float(self.area_mm2[k].sum() / self.tray_mm2) if self.tray_mm2 > 0 else 0.0
        if n:
            o = np.argsort(self.b_mm[k]); cm = np.cumsum(self.mass[k][o]); cm /= cm[-1]
            d50 = float(self.b_mm[k][o][np.searchsorted(cm, 0.5)])
        else:
            d50 = 0.0
        return np.array([np.log1p(n), cov, np.log1p(d50)])


def measure(rgb: np.ndarray, rho: float, segmenter, cfg: Config) -> CoarseResult:
    """Segment one cropped photo and keep reliable grains."""
    H, W = rgb.shape[:2]
    res = CoarseResult(tray_mm2=H * W / rho ** 2)
    b_l, m_l, a_l = [], [], []
    for r in segmenter.segment(rgb):
        a, b, sol = grain_axes(r.mask)
        if b < cfg.tau_px or sol < cfg.min_solidity:            # too small / clump
            continue
        y0, x0, y1, x1 = r.bbox
        if y0 <= 0 or x0 <= 0 or y1 >= H or x1 >= W:             # cut by image edge
            continue
        # TODO: drop clods whose colour matches the fine matrix
        a_mm, b_mm = a / rho, b / rho
        b_l.append(b_mm); m_l.append(a_mm * b_mm ** cfg.mass_p)
        a_l.append(r.mask.sum() / rho ** 2)
    res.b_mm, res.mass, res.area_mm2 = map(np.asarray, (b_l, m_l, a_l))
    return res
