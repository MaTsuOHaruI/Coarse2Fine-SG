from dataclasses import dataclass, replace
import numpy as np
import cv2
from .config import Config, SIEVES
from .crossover import adaptive_crossover
from .path_a import measure, CoarseResult
from .features import center_crop, resample_to, image_feats, SimpleEmbedder
from .path_b import PathB
from .path_c import blend


@dataclass
class Sample:
    sample_id: str
    photos: list            # [(path_or_ndarray(RGB), rho_px_per_mm), ...]


@dataclass
class Extracted:
    sample_id: str
    coarse: CoarseResult    # Path A, pooled over photos
    dc: float               # adaptive crossover [mm]
    feats: np.ndarray       # Path B input vector


def _load(p):
    if isinstance(p, np.ndarray):
        return p
    # np.fromfile + imdecode: cv2.imread fails on non-ASCII paths on Windows
    img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(p)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def extract(s: Sample, segmenter, cfg: Config, embedder=None) -> Extracted:
    """Run the label-independent half once per sample (cache this; it's the slow part)."""
    embedder = embedder or SimpleEmbedder()
    coarse, dcs, fv = CoarseResult(), [], []
    for p, rho in s.photos:
        img = _load(p)
        if cfg.max_side and max(img.shape[:2]) > cfg.max_side:
            sc = cfg.max_side / max(img.shape[:2])
            img = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            rho = rho * sc
        img = center_crop(img, cfg.tray_crop_frac)
        coarse = coarse.merge(measure(img, rho, segmenter, cfg))
        dcs.append(adaptive_crossover(rho, cfg.tau_px))
        fv.append(image_feats(resample_to(img, rho, cfg.target_rho), embedder))
    feats = np.r_[np.mean(fv, 0), coarse.stats(cfg.feat_min_mm)]
    return Extracted(s.sample_id, coarse, float(max(dcs)), feats)


class SoilGSD:
    def __init__(self, cfg: Config = Config()):
        self.cfg = cfg

    def fit(self, xs: list, curves: np.ndarray):
        self.b = PathB(self.cfg.n_pca, self.cfg.ridge_alpha).fit(np.stack([x.feats for x in xs]), curves)
        return self

    def predict(self, x: Extracted, mode: str = None) -> dict:
        cfg = replace(self.cfg, blend_mode=mode or self.cfg.blend_mode)
        pB = self.b.predict(x.feats)
        return {"pB": pB, "final": blend(pB, x.coarse, x.dc, cfg)}
