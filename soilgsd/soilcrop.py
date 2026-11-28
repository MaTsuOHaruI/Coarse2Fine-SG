# NEW file (Path B). Drop into soilgsd/ ; nothing existing is changed.
"""Soil-only crop: the largest axis-aligned rectangle whose rows and columns are almost all soil.

Soil = pixels close (in blurred CIELAB) to the median colour of the image centre. The tray rim
(dark), lighting strips (bright) and table are rejected. Works on the image already resampled to
4.6 px/mm. Returns the crop and the box (y0, y1, x0, x1).
"""
import numpy as np
import cv2


def soil_mask(rgb, blur_px=4.0, dark=0.5, bright=235):
    """Non-soil = tray rim / table (much darker than the soil) or lighting strips (near white).
    Brightness is blurred so grain shadows do not count as rim."""
    L = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    Lb = cv2.GaussianBlur(L, (0, 0), blur_px)
    H, W = L.shape
    med = np.median(Lb[H // 4: 3 * H // 4, W // 4: 3 * W // 4])
    white = cv2.GaussianBlur((L > bright).astype(np.float32), (0, 0), blur_px) > 0.05
    return (Lb > dark * med) & ~white


def _longest_run(ok):
    best, cur, start, bs = 0, 0, 0, 0
    for i, v in enumerate(ok):
        if v:
            if cur == 0: start = i
            cur += 1
            if cur > best: best, bs = cur, start
        else:
            cur = 0
    return bs, bs + best


def soil_box(mask, need=0.9, inset=0.03, iters=3):
    """Rows (cols) count as soil when >= need of their pixels inside the current column (row) range
    are soil; keep the longest contiguous run of soil rows and of soil cols; repeat a few times."""
    H, W = mask.shape
    y0, y1, x0, x1 = 0, H, 0, W
    for _ in range(iters):
        ny0, ny1 = _longest_run(mask[:, x0:x1].mean(1) >= need)
        nx0, nx1 = _longest_run(mask[y0:y1, :].mean(0) >= need)
        if ny1 - ny0 < 0.2 * H or nx1 - nx0 < 0.2 * W:
            break
        y0, y1, x0, x1 = ny0, ny1, nx0, nx1
    dy, dx = int(inset * (y1 - y0)), int(inset * (x1 - x0))
    return y0 + dy, y1 - dy, x0 + dx, x1 - dx


def soil_crop(rgb):
    y0, y1, x0, x1 = soil_box(soil_mask(rgb))
    return rgb[y0:y1, x0:x1], (y0, y1, x0, x1)
