"""Check the soil-crop result: centre crop 0.9 vs soil-region crop, single_phone LOSO.

Expected (24 training soils, 127 photos):
    ens_phone single_phone LOSO   centre 36.24   soil 35.22   (paired diff -1.02, 95% CI about -2.1 to +0.1)
    nested LOSO (9 candidates)    centre 38.23   soil 37.36

single_phone: each held-out soil is predicted separately from each phone's photos; the soil's score is
the mean over its phones, the CV score the mean over soils. Training rows are one per (soil, phone)
(photo_emd: one per photo). Nested: for every held-out soil an inner LOSO over the other 23 soils picks one
of the 9 candidates; that candidate's held-out EMD is the outer score.

Needs only soilgsd (+ soilgsd/path_b_ext.py and soilgsd/soilcrop.py). Test photos are not needed.
Usage:  uv run python scripts/verify_soilcrop.py
Takes a few minutes (texture features of 127 photos x 2 crops, then about 24 x 23 x 9 small fits).
"""
from dataclasses import replace

import cv2
import numpy as np
import pandas as pd

from soilgsd import config
from soilgsd.config import Config, ID_COLUMN
from soilgsd.data import load_labels, photo_table
from soilgsd.features import center_crop, resample_to, texture_feats
from soilgsd.metrics import clean_curve, emd
from soilgsd.path_b_ext import KNNMedianCurve, PathBExt
from soilgsd.pipeline import _load
from soilgsd.soilcrop import soil_crop
from sklearn.preprocessing import StandardScaler


def prepared(path, rho, cfg):
    """pipeline.extract preprocessing up to (not including) the crop: max_side downscale."""
    img = _load(path)
    if cfg.max_side and max(img.shape[:2]) > cfg.max_side:
        sc = cfg.max_side / max(img.shape[:2])
        img = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        rho *= sc
    return img, rho


def feats_centre(path, rho, cfg):
    img, rho = prepared(path, rho, cfg)
    return texture_feats(resample_to(center_crop(img, cfg.tray_crop_frac), rho, cfg.target_rho))


def feats_soil(path, rho, cfg):
    img, rho = prepared(path, rho, cfg)
    return texture_feats(soil_crop(resample_to(img, rho, cfg.target_rho))[0])


# ---- heads: fit(X, curves, groups) / predict(rows of one unit) ----
class KNNWeighted:
    """k nearest soils (soil distance = its nearest row), weighted median curve, weights 1/(d+1e-3)."""
    def __init__(self, k):
        self.k = k

    def fit(self, X, C, groups):
        self.sc = StandardScaler().fit(X)
        self.X, self.C, self.g = self.sc.transform(X), C, np.asarray(groups)
        return self

    def predict(self, x):
        q = self.sc.transform(np.atleast_2d(x)).mean(0)
        d = np.linalg.norm(self.X - q, axis=1)
        best = {}
        for i in np.argsort(d):
            best.setdefault(self.g[i], i)
            if len(best) == self.k:
                break
        idx = np.array(list(best.values()))
        w = 1 / (d[idx] + 1e-3)
        w /= w.sum()
        out = np.empty(self.C.shape[1])
        for j in range(len(out)):
            o = np.argsort(self.C[idx, j])
            out[j] = self.C[idx, j][o][np.searchsorted(np.cumsum(w[o]), 0.5)]
        return clean_curve(out)


class Avg:
    def __init__(self, *makers):
        self.makers = makers

    def fit(self, X, C, groups):
        self.ms = [m().fit(X, C, groups) for m in self.makers]
        return self

    def predict(self, x):
        return clean_curve(np.mean([m.predict(x) for m in self.ms], 0))


emd_head = lambda: PathBExt(loss="emd")
CANDIDATES = {                         # name -> (maker, training rows)
    "ens_phone": (lambda: Avg(emd_head, lambda: KNNMedianCurve(k=5)), "phone"),
    "phone_knn5": (lambda: KNNMedianCurve(k=5), "phone"),
    "phone_knn7": (lambda: KNNMedianCurve(k=7), "phone"),
    "phone_knnw5": (lambda: KNNWeighted(5), "phone"),
    "phone_knnw7": (lambda: KNNWeighted(7), "phone"),
    "phone_emd": (emd_head, "phone"),
    "photo_emd": (emd_head, "photo"),
    "ens_emd_knnw5": (lambda: Avg(emd_head, lambda: KNNWeighted(5)), "phone"),
    "ens_emd_knn5_knn3": (lambda: Avg(emd_head, lambda: KNNMedianCurve(k=5), lambda: KNNMedianCurve(k=3)), "phone"),
}


def loso(maker, level, P, curve_of, soils):
    """P: DataFrame, one row per photo: sample_id, phone, f0..f9. -> Series soil -> mean EMD over its phones."""
    fc = [c for c in P.columns if c.startswith("f")]
    if level == "phone":
        R = P.groupby(["sample_id", "phone"], sort=False)[fc].mean().reset_index()
    else:
        R = P
    X, g = R[fc].to_numpy(), R.sample_id.to_numpy()
    C = np.stack([curve_of[s] for s in g])
    out = {}
    for s in soils:
        m = maker().fit(X[g != s], C[g != s], g[g != s])
        units = P[P.sample_id == s].groupby("phone", sort=False)[fc]
        rows = [u.mean().to_numpy()[None] if level == "phone" else u.to_numpy() for _, u in units]
        out[s] = np.mean([emd(curve_of[s], m.predict(x)) for x in rows])
    return pd.Series(out)


def paired_boot(a, b, n=4000, seed=0):
    d = np.asarray(a) - np.asarray(b)
    bs = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    args = config.base_parser(__doc__).parse_args()
    ycfg = config.load_config(args.config)
    cfg = Config.from_yaml(ycfg)
    root = config.kaggle_dir(ycfg)
    info = photo_table(root)
    tr = info[info.split == "train"].reset_index(drop=True)
    lab = load_labels(root)
    curve_of = {r[0]: clean_curve(np.asarray(r[1:], float)) for r in lab.itertuples(index=False)}
    soils = lab[ID_COLUMN].tolist()
    print(f"{len(tr)} training photos, {len(soils)} soils; computing texture features ...", flush=True)

    tables = {}
    for name, fn in (("centre", feats_centre), ("soil", feats_soil)):
        F = np.stack([fn(r.file, r.rho, cfg) for r in tr.itertuples()])
        tables[name] = pd.concat([tr[["sample_id", "phone"]],
                                  pd.DataFrame(F, columns=[f"f{j}" for j in range(F.shape[1])])], axis=1)

    rows, per_soil = [], {}
    for name, P in tables.items():
        res = {c: loso(mk, lvl, P, curve_of, soils) for c, (mk, lvl) in CANDIDATES.items()}
        per_soil[name] = res["ens_phone"]
        nested, picks = [], []
        for s in soils:                                   # inner LOSO on the other 23 soils
            Pi, inner_soils = P[P.sample_id != s], [x for x in soils if x != s]
            inner = {c: loso(mk, lvl, Pi, curve_of, inner_soils).mean() for c, (mk, lvl) in CANDIDATES.items()}
            best = min(inner, key=inner.get)
            picks.append(best)
            nested.append(res[best][s])
        rows.append(dict(crop=name, ens_phone=res["ens_phone"].mean(), nested=np.mean(nested)))
        print(f"\n[{name}] single_phone LOSO per candidate:")
        for c in CANDIDATES:
            print(f"  {c:20s} {res[c].mean():6.2f}")
        print(f"  nested picks: {pd.Series(picks).value_counts().to_dict()}", flush=True)

    print("\n" + pd.DataFrame(rows).round(2).to_string(index=False))
    m, lo, hi = paired_boot(per_soil["soil"], per_soil["centre"])
    print(f"\nens_phone soil - centre: {m:+.2f}  (95% bootstrap {lo:+.2f} to {hi:+.2f})")
    print("expected: ens_phone 36.24 -> 35.22, nested 38.23 -> 37.36")


if __name__ == "__main__":
    main()
