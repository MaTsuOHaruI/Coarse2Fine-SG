"""Loader for the Kaggle layout:
  root/ppm_updated.csv  (phone,camera,width,height,ppm  -- full-resolution scale)
  root/Training_labels_updated.csv, root/sample_submission.csv
  root/Training-All_Photos_updated/**/*.jpg, root/Test_All_Photos/**/*.jpg
File name -> (phone, sample_id) is inferred by matching names (see parse_name);
run `python -m soilgsd check` to inspect the result before training."""
import re, difflib, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from .pipeline import Sample
from .metrics import clean_curve

IMG_EXT = {".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff"}
_UML = {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", str(s)).lower()
    for k, v in _UML.items():
        s = s.replace(k, v)
    return "".join(c for c in s if c.isalnum())


def _stem(name: str) -> str:
    stem = Path(name).stem
    return re.sub(r"[\s_\-]*\(\d+\)\s*$", "", stem)        # drop trailing " (3)"


def parse_name(fname, ids_norm: dict, phones: list):
    """-> (phone_row_index|None, sample_id|None). phones: [(alias_norm, row_idx)] longest first."""
    n = norm(_stem(fname))
    ph, rem = None, n
    for alias, ri in phones:
        if n.startswith(alias):
            ph, rem = ri, n[len(alias):]
            break
    cands = [k for k in ids_norm if k and k in rem]
    if cands:
        return ph, ids_norm[max(cands, key=len)]
    m = difflib.get_close_matches(rem, list(ids_norm), n=1, cutoff=0.6)
    return ph, (ids_norm[m[0]] if m else None)


def load_kaggle(root):
    root = Path(root)
    ppm = pd.read_csv(root / "ppm_updated.csv")
    lab = pd.read_csv(root / "Training_labels_updated.csv")
    sub = pd.read_csv(root / "sample_submission.csv")
    aliases = sorted({(norm(v), i) for i, r in ppm.iterrows() for v in (r.phone, r.camera)},
                     key=lambda t: -len(t[0]))
    rows = []
    for split, folder, ids in (("train", "Training-All_Photos_updated", lab.sample_id.astype(str)),
                               ("test", "Test_All_Photos", sub.sample_id.astype(str))):
        idn = {norm(i): i for i in ids}
        for f in sorted((root / folder).rglob("*")):
            if f.suffix.lower() not in IMG_EXT:
                continue
            ri, sid = parse_name(f.name, idn, aliases)
            w, h = Image.open(f).size
            if ri is None:                                   # fallback: match by image size
                hit = ppm.index[[(set((w, h)) == {a, b}) for a, b in zip(ppm.width, ppm.height)]]
                ri = hit[0] if len(hit) == 1 else None
            rho = None
            if ri is not None:                               # long-side ratio: orientation-proof
                rho = float(ppm.ppm[ri]) * max(w, h) / max(ppm.width[ri], ppm.height[ri])
            rows.append(dict(split=split, file=str(f), name=f.name, sample_id=sid,
                             phone=None if ri is None else ppm.phone[ri], w=w, h=h, rho=rho))
    info = pd.DataFrame(rows)
    bad = info[info.sample_id.isna() | info.rho.isna()]
    if len(bad):
        raise ValueError("could not resolve sample_id / phone for:\n" + bad[["split", "name", "sample_id", "phone"]].to_string())

    def build(split, ids):
        d = info[info.split == split]
        out = []
        for sid in ids:
            g = d[d.sample_id == sid]
            if g.empty:
                raise ValueError(f"no photos found for {split} sample {sid!r}")
            out.append(Sample(str(sid), list(zip(g.file, g.rho))))
        return out

    tr_ids = lab.sample_id.astype(str).tolist()
    curves = np.stack([clean_curve(r.values[:11].astype(float)) for _, r in lab.iloc[:, 1:].iterrows()])
    return dict(train=build("train", tr_ids), curves=curves,
                test=build("test", sub.sample_id.astype(str).tolist()),
                sub_columns=list(sub.columns), info=info)
