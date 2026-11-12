from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np
import cv2
from .config import Config, SIEVES
from .crossover import adaptive_crossover
from .path_a import measure, CoarseResult
from .features import center_crop, resample_to, image_feats, SimpleEmbedder
from .path_b import PathB
from .path_c import blend
from .segment import save_regions, load_regions


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


class _CachedRegions:
    """Stands in for a segmenter when the masks of a photo are already on disk."""
    def __init__(self, regions):
        self.regions = regions

    def segment(self, rgb):
        return self.regions


def extract(s: Sample, segmenter, cfg: Config, embedder=None, cache_dir=None) -> Extracted:
    """Run the label-independent half once per sample (the slow part).

    cache_dir: when given, each photo's masks go to cache_dir/masks/<segmenter>_<prep>/<photo>.npz
    and its feature vector to cache_dir/features/<embedder>_<prep>_rho<target>/<photo>.npy.
    Cached photos skip segmentation and feature extraction, so an interrupted run resumes.
    Masks are cached before filtering, so changing tau, solidity or p does not need a new SAM run.
    """
    embedder = embedder or SimpleEmbedder()
    prep = f"ms{cfg.max_side}_cr{cfg.tray_crop_frac}"
    coarse, dcs, fv = CoarseResult(), [], []
    for p, rho in s.photos:
        img = _load(p)
        if cfg.max_side and max(img.shape[:2]) > cfg.max_side:
            sc = cfg.max_side / max(img.shape[:2])
            img = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            rho = rho * sc
        img = center_crop(img, cfg.tray_crop_frac)
        seg, feat = segmenter, None
        if cache_dir is not None and not isinstance(p, np.ndarray):
            key = Path(p).stem
            emb_name = getattr(embedder, "name", type(embedder).__name__)
            mpath = Path(cache_dir) / "masks" / f"{segmenter.name}_{prep}" / f"{key}.npz"
            fpath = Path(cache_dir) / "features" / f"{emb_name}_{prep}_rho{cfg.target_rho}" / f"{key}.npy"
            if not mpath.exists():
                save_regions(mpath, segmenter.segment(img))
            seg = _CachedRegions(load_regions(mpath))
            if fpath.exists():
                feat = np.load(fpath)
            else:
                feat = image_feats(resample_to(img, rho, cfg.target_rho), embedder)
                fpath.parent.mkdir(parents=True, exist_ok=True)
                tmp = fpath.with_name(fpath.name + ".tmp.npy")
                np.save(tmp, feat)
                tmp.replace(fpath)
        if feat is None:
            feat = image_feats(resample_to(img, rho, cfg.target_rho), embedder)
        coarse = coarse.merge(measure(img, rho, seg, cfg))
        dcs.append(adaptive_crossover(rho, cfg.tau_px))
        fv.append(feat)
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


def add_tool_args(parser):
    """Flags shared by the scripts that run the pipeline (see make_tools)."""
    parser.add_argument("--segmenter", default="classical", choices=["classical", "sam"])
    parser.add_argument("--sam-ckpt", default=None, help="SAM checkpoint (default: sam.checkpoint under the cache dir)")
    parser.add_argument("--dino", action="store_true", help="DINOv2 embeddings instead of the 4x4 colour thumbnail")
    return parser


def make_tools(ycfg: dict, args):
    """Segmenter and embedder from the YAML config dict and the add_tool_args flags."""
    from . import config
    from .segment import make_segmenter
    device = config.pick_device(args.device)
    pre = ycfg.get("preprocess", {})
    sam = ycfg.get("sam", {})
    ckpt = args.sam_ckpt or (config.cache_dir(ycfg) / sam["checkpoint"] if sam.get("checkpoint") else None)
    seg = make_segmenter(args.segmenter, ckpt, device, sam.get("model_type", "vit_b"),
                         pre.get("tile_px", 1024), pre.get("tile_overlap_px", 256))
    if args.dino:
        from .features import Dinov2Embedder
        emb = Dinov2Embedder(device=device)
    else:
        emb = SimpleEmbedder()
    return seg, emb
