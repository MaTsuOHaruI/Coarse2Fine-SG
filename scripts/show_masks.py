"""Draw cached masks over their photo, coloured by what Path A (path_a.measure) does with each one.

Colours: green = kept, grey = too small (b < tau_px), orange = low solidity (clump),
red = cut by the image edge. Rules and order are the same as in path_a.measure.
The photo is prepared exactly as in pipeline.extract (max_side resize, centre crop), so the
cached boxes line up with the image.

Usage:  uv run python scripts/show_masks.py [--segmenter sam|classical] [--mask-dir DIR]
                                            [--photos STEM ...] [--limit N]
Writes: runs/<stamp>_overlays/<photo>.png (long side at most 2048 px)
        runs/<stamp>_overlays/regions.csv (one row per region: photo, sample_id, label,
        a_px, b_px, solidity, b_mm)
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from soilgsd import config, data
from soilgsd.config import ID_COLUMN, Config
from soilgsd.features import center_crop
from soilgsd.path_a import grain_axes
from soilgsd.pipeline import _load
from soilgsd.segment import load_regions

# BGR is not used anywhere here: images stay RGB until cv2.imwrite, which gets a converted copy.
COLOURS_RGB = {
    "kept": (40, 200, 60),
    "too_small": (150, 150, 150),
    "low_solidity": (255, 150, 0),
    "edge_cut": (230, 30, 30),
}
MAX_OUT_SIDE_PX = 2048


def default_mask_dir(ycfg: dict, cfg: Config, segmenter: str) -> Path:
    """Cache folder that pipeline.extract writes for this config: masks/<segmenter name>_<prep>."""
    pre, sam = ycfg.get("preprocess", {}), ycfg.get("sam", {})
    if segmenter == "sam":
        name = f"sam_{sam.get('model_type', 'vit_b')}_t{pre.get('tile_px', 1024)}_o{pre.get('tile_overlap_px', 256)}"
    else:
        name = "classical_d8_a30"     # ClassicalSegmenter() defaults
    prep = f"ms{cfg.max_side}_cr{cfg.tray_crop_frac}"
    return config.cache_dir(ycfg) / "masks" / f"{name}_{prep}"


def prepare_photo(path, rho_px_per_mm: float, cfg: Config):
    """Same steps as pipeline.extract before segmentation.
    Returns (rgb uint8 (H, W, 3), rho in px/mm after any resize)."""
    img = _load(path)
    if cfg.max_side and max(img.shape[:2]) > cfg.max_side:
        sc = cfg.max_side / max(img.shape[:2])
        img = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        rho_px_per_mm = rho_px_per_mm * sc
    return center_crop(img, cfg.tray_crop_frac), rho_px_per_mm


def classify(region, H: int, W: int, cfg: Config):
    """Label one region with the first path_a.measure rule that drops it, else "kept".
    Returns (label, a_px, b_px, solidity)."""
    a, b, sol = grain_axes(region.mask)
    if b < cfg.tau_px:
        return "too_small", a, b, sol
    if sol < cfg.min_solidity:
        return "low_solidity", a, b, sol
    y0, x0, y1, x1 = region.bbox
    if y0 <= 0 or x0 <= 0 or y1 >= H or x1 >= W:
        return "edge_cut", a, b, sol
    return "kept", a, b, sol


def draw(rgb: np.ndarray, regions: list, labels: list) -> np.ndarray:
    """Masks filled at 40 % opacity plus a 2 px outline, in the colour of their label.
    Input and output: RGB uint8 (H, W, 3)."""
    out = rgb.copy()
    for r, lab in zip(regions, labels):
        y0, x0, y1, x1 = r.bbox
        colour = np.array(COLOURS_RGB[lab], dtype=np.float32)
        sub = out[y0:y1, x0:x1]
        sub[r.mask] = (0.6 * sub[r.mask] + 0.4 * colour).astype(np.uint8)
        cs, _ = cv2.findContours(r.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(sub, cs, -1, COLOURS_RGB[lab], 2)
    return out


def add_legend(rgb: np.ndarray, counts: dict, title: str) -> np.ndarray:
    """Write the photo name and the count per label in the top-left corner (in place)."""
    lines = [title] + [f"{lab}: {counts.get(lab, 0)}" for lab in COLOURS_RGB]
    scale = max(rgb.shape[:2]) / 1500
    for i, text in enumerate(lines):
        y = int((40 + 45 * i) * scale)
        colour = (255, 255, 255) if i == 0 else COLOURS_RGB[lines[i].split(":")[0]]
        cv2.putText(rgb, text, (int(20 * scale), y), cv2.FONT_HERSHEY_SIMPLEX, 1.2 * scale, (0, 0, 0),
                    int(6 * scale) + 1, cv2.LINE_AA)
        cv2.putText(rgb, text, (int(20 * scale), y), cv2.FONT_HERSHEY_SIMPLEX, 1.2 * scale, colour,
                    int(2 * scale) + 1, cv2.LINE_AA)
    return rgb


def main():
    parser = config.base_parser(__doc__)
    parser.add_argument("--segmenter", default="sam", choices=["sam", "classical"],
                        help="which cache folder to read when --mask-dir is not given")
    parser.add_argument("--mask-dir", default=None, help="folder of cached .npz masks")
    parser.add_argument("--photos", nargs="*", default=None,
                        help="photo file stems to draw (default: every cached photo)")
    args = parser.parse_args()
    ycfg = config.load_config(args.config)
    cfg = Config.from_yaml(ycfg)
    mask_dir = Path(args.mask_dir) if args.mask_dir else default_mask_dir(ycfg, cfg, args.segmenter)
    if not mask_dir.is_dir():
        raise SystemExit(f"no mask folder at {mask_dir}")

    kaggle = config.kaggle_dir(ycfg)
    photos = pd.concat([data.list_photos(kaggle, "train"), data.list_photos(kaggle, "test")])
    photos["stem"] = [Path(p).stem for p in photos["path"]]
    cached = {p.stem for p in mask_dir.glob("*.npz") if not p.name.endswith(".tmp.npz")}
    todo = photos[photos["stem"].isin(cached)]
    if args.photos:
        todo = todo[todo["stem"].isin(args.photos)]
    if args.limit:
        todo = todo.head(args.limit)
    if todo.empty:
        raise SystemExit(f"no matching cached photos in {mask_dir}")

    run_dir = config.new_run_dir(ycfg, "overlays")
    print(f"{len(todo)} photos | masks {mask_dir} | tau {cfg.tau_px} px | solidity {cfg.min_solidity}")
    rows = []
    for _, ph in todo.iterrows():
        rgb, rho = prepare_photo(ph["path"], ph["rho_px_per_mm"], cfg)
        H, W = rgb.shape[:2]
        regions = load_regions(mask_dir / f"{ph['stem']}.npz")
        labels = []
        for r in regions:
            lab, a, b, sol = classify(r, H, W, cfg)
            labels.append(lab)
            rows.append({"photo": ph["stem"], ID_COLUMN: ph[ID_COLUMN], "label": lab, "a_px": round(a, 1),
                         "b_px": round(b, 1), "solidity": round(sol, 3), "b_mm": round(b / rho, 2)})
        counts = pd.Series(labels, dtype=str).value_counts().to_dict()
        img = add_legend(draw(rgb, regions, labels), counts, f"{ph['stem']}  ({ph[ID_COLUMN]})")
        sc = MAX_OUT_SIDE_PX / max(H, W)
        if sc < 1:
            img = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        out = run_dir / f"{ph['stem']}.png"
        ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))   # imencode: non-ASCII paths
        if not ok:
            raise RuntimeError(f"could not encode {out}")
        buf.tofile(str(out))
        print(f"{ph['stem']}: {len(regions)} regions, kept {counts.get('kept', 0)}")

    pd.DataFrame(rows).to_csv(run_dir / "regions.csv", index=False)
    print(f"\nwritten to {run_dir}")


if __name__ == "__main__":
    main()
