"""Segmentation back-ends. All return list[Region] for one (tile of a) photo."""
from dataclasses import dataclass
from typing import Protocol
import numpy as np
import cv2


@dataclass
class Region:
    mask: np.ndarray          # bool crop, tight to bbox
    bbox: tuple               # (y0, x0, y1, x1) in image coords


class Segmenter(Protocol):
    def segment(self, rgb: np.ndarray) -> list: ...


class ClassicalSegmenter:
    """Dependency-light stand-in for SAM: bg-colour distance + watershed.
    Lets the whole pipeline run without GPU / weights."""

    def __init__(self, min_dist: int = 8, min_area: int = 30):
        self.min_dist, self.min_area = min_dist, min_area

    def segment(self, rgb):
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed
        from scipy import ndimage as ndi
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab = cv2.GaussianBlur(lab, (0, 0), 1.5)
        border = np.concatenate([lab[:6].reshape(-1, 3), lab[-6:].reshape(-1, 3),
                                 lab[:, :6].reshape(-1, 3), lab[:, -6:].reshape(-1, 3)])
        bg = np.median(border, axis=0)
        dist = np.linalg.norm(lab - bg, axis=2)
        d8 = np.clip(dist / max(dist.max(), 1e-6) * 255, 0, 255).astype(np.uint8)
        t, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fg = d8 > t
        fg = ndi.binary_opening(fg, iterations=1)
        dt = ndi.distance_transform_edt(fg)
        pk = peak_local_max(dt, min_distance=self.min_dist, labels=fg, exclude_border=False)
        markers = np.zeros(fg.shape, np.int32)
        markers[tuple(pk.T)] = np.arange(1, len(pk) + 1)
        lbl = watershed(-dt, markers, mask=fg)
        out = []
        for i, sl in enumerate(ndi.find_objects(lbl), start=1):
            if sl is None:
                continue
            m = lbl[sl] == i
            if m.sum() < self.min_area:
                continue
            out.append(Region(m, (sl[0].start, sl[1].start, sl[0].stop, sl[1].stop)))
        return out


class SamSegmenter:
    """segment-anything automatic mask generator, tiled at 1024 px with overlap.
    MVP merge rule: drop masks touching an *interior* tile border (they are cut grains);
    the overlap guarantees such grains are whole in a neighbouring tile."""

    def __init__(self, checkpoint: str, model_type: str = "vit_b", device: str = "cuda",
                 tile: int = 1024, overlap: int = 256):
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
        self.gen = SamAutomaticMaskGenerator(sam, points_per_side=32, min_mask_region_area=50)
        self.tile, self.overlap = tile, overlap

    def segment(self, rgb):
        H, W = rgb.shape[:2]
        step = self.tile - self.overlap
        ys = list(range(0, max(H - self.tile, 0) + 1, step)) or [0]
        xs = list(range(0, max(W - self.tile, 0) + 1, step)) or [0]
        if ys[-1] + self.tile < H: ys.append(H - self.tile)
        if xs[-1] + self.tile < W: xs.append(W - self.tile)
        out, seen = [], []
        for y0 in ys:
            for x0 in xs:
                y1, x1 = min(y0 + self.tile, H), min(x0 + self.tile, W)
                for m in self.gen.generate(rgb[y0:y1, x0:x1]):
                    seg = m["segmentation"]
                    yy, xx = np.where(seg)
                    if yy.size == 0: continue
                    ty0, tx0, ty1, tx1 = yy.min(), xx.min(), yy.max() + 1, xx.max() + 1
                    cut = ((ty0 == 0 and y0 > 0) or (tx0 == 0 and x0 > 0) or
                           (ty1 == y1 - y0 and y1 < H) or (tx1 == x1 - x0 and x1 < W))
                    if cut: continue
                    box = (y0 + ty0, x0 + tx0, y0 + ty1, x0 + tx1)
                    cy, cx = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                    if any(abs(cy - a) < 5 and abs(cx - b) < 5 for a, b in seen):  # dedupe overlap
                        continue
                    seen.append((cy, cx))
                    out.append(Region(seg[ty0:ty1, tx0:tx1], box))
        return out
