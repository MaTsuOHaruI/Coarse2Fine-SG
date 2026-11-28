"""Path B ens_phone on soil-crop texture features (tray rim / light strip removed per photo).

Same model as scripts/make_submission_pathb_ext.py --model ens_phone (PathBExt loss=emd + KNNMedianCurve k=5,
both trained on one row per (soil, phone)); only the image region the 10 texture features are computed on
changes: soilgsd.soilcrop.soil_crop instead of the fixed 0.9 centre crop.

Usage:  uv run python scripts/make_submission_pathb_soilcrop.py [--loso] [--save-crops N]
  --loso         also print the single_phone LOSO score (expected 35.22; 36.24 with the centre crop)
  --save-crops   write N test photos with their crop box to the run folder (check the iPhone framing!)
Writes: runs/<stamp>_submission_pathb_soilcrop/{submission.csv, features.csv, crops_*.jpg}
"""
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from soilgsd import config
from soilgsd.config import Config, ID_COLUMN
from soilgsd.data import load_labels, load_test_ids, photo_table
from soilgsd.features import center_crop, resample_to, texture_feats
from soilgsd.metrics import clean_curve, emd
from soilgsd.path_b_ext import KNNMedianCurve, PathBExt
from soilgsd.pipeline import _load
from soilgsd.soilcrop import soil_crop
from soilgsd.submit import write_submission


def soil_image(path, rho, cfg):
    """pipeline.extract preprocessing without the centre crop, at cfg.target_rho, then the soil crop."""
    img = _load(path)
    if cfg.max_side and max(img.shape[:2]) > cfg.max_side:
        sc = cfg.max_side / max(img.shape[:2])
        img = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        rho *= sc
    return soil_crop(resample_to(img, rho, cfg.target_rho))


class EnsPhone:
    def fit(self, X, curves, groups):
        self.a = PathBExt(loss="emd").fit(X, curves, groups=groups)
        self.b = KNNMedianCurve(k=5).fit(X, curves, groups=groups)
        return self

    def predict(self, x):
        return clean_curve((self.a.predict(x) + self.b.predict(x)) / 2)


def main():
    p = config.base_parser(__doc__)
    p.add_argument("--loso", action="store_true")
    p.add_argument("--save-crops", type=int, default=0)
    a = p.parse_args()
    ycfg = config.load_config(a.config)
    cfg = Config.from_yaml(ycfg)
    root = config.kaggle_dir(ycfg)
    info = photo_table(root)
    run_dir = config.new_run_dir(ycfg, "submission_pathb_soilcrop")

    rows, saved = [], 0
    for i, r in info.iterrows():
        img, box = soil_image(r.file, r.rho, cfg)
        rows.append(dict(split=r.split, sample_id=r.sample_id, phone=r.phone, file=r.name,
                         box=box, **{f"t{j}": v for j, v in enumerate(texture_feats(img))}))
        if r.split == "test" and saved < a.save_crops:
            saved += 1
            full = resample_to(_load(r.file), r.rho, cfg.target_rho).copy()
            y0, y1, x0, x1 = box
            cv2.rectangle(full, (x0, y0), (x1, y1), (255, 0, 0), 4)
            cv2.imwrite(str(run_dir / f"crops_{Path(r.file).stem}.jpg"), cv2.cvtColor(full, cv2.COLOR_RGB2BGR))
    F = pd.DataFrame(rows)
    F.to_csv(run_dir / "features.csv", index=False)
    tcols = [f"t{j}" for j in range(10)]

    lab = load_labels(root)
    curve_of = {r[0]: clean_curve(np.asarray(r[1:], float)) for r in lab.itertuples(index=False)}
    unit = F.groupby(["split", "sample_id", "phone"], sort=False)[tcols].mean().reset_index()
    trU = unit[unit.split == "train"].reset_index(drop=True)
    X, g = trU[tcols].to_numpy(), trU.sample_id.to_numpy()
    C = np.stack([curve_of[s] for s in g])

    if a.loso:
        per = []
        for s in lab[ID_COLUMN]:
            m = EnsPhone().fit(X[g != s], C[g != s], g[g != s])
            per.append(np.mean([emd(curve_of[s], m.predict(x)) for x in X[g == s]]))
        print(f"single_phone LOSO (ens_phone, soil crop): {np.mean(per):.2f}")

    teU = unit[unit.split == "test"]
    if teU.empty:
        print("no test photos found; submission not written")
        return
    m = EnsPhone().fit(X, C, g)
    curves = {}
    for s, grp in teU.groupby("sample_id", sort=False):      # one phone per test soil; average if several
        curves[s] = clean_curve(np.mean([m.predict(x) for x in grp[tcols].to_numpy()], 0))
    write_submission(curves, load_test_ids(root), run_dir / "submission.csv")
    print(f"written {run_dir / 'submission.csv'}")


if __name__ == "__main__":
    main()