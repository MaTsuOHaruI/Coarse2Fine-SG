"""Loader for the Kaggle layout:
  root/ppm_updated.csv  (phone,camera,width,height,ppm  -- full-resolution scale, one row per phone)
  root/Training_labels_updated.csv, root/sample_submission.csv
  root/Training-All_Photos_updated/**/*.jpg, root/Test_All_Photos/**/*.jpg
File name -> (phone, sample_id) is inferred by matching names (see parse_name);
run `python -m soilgsd check` to inspect the result before training.
Everything downstream groups by sample_id, never by photo."""
import re, difflib, unicodedata, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from .config import ID_COLUMN, SIEVE_COLUMNS
from .crossover import true_scale
from .pipeline import Sample
from .metrics import clean_curve

IMG_EXT = {".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff"}
_UML = {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"}
SPLIT_FOLDERS = {"train": "Training-All_Photos_updated", "test": "Test_All_Photos"}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", str(s)).lower()
    for k, v in _UML.items():
        s = s.replace(k, v)
    return "".join(c for c in s if c.isalnum())


def _stem(name: str) -> str:
    stem = Path(name).stem
    return re.sub(r"[\s_\-]*\(\d+\)\s*$", "", stem)        # drop trailing " (3)"


def parse_name(fname, ids_norm: dict, phones: list):
    """-> (phone_row_index|None, sample_id|None, how). phones: [(alias_norm, row_idx)] longest first.
    how is "exact" (ID found inside the name), "fuzzy" (closest-name guess) or "none"."""
    n = norm(_stem(fname))
    ph, rem = None, n
    for alias, ri in phones:
        if n.startswith(alias):
            ph, rem = ri, n[len(alias):]
            break
    cands = [k for k in ids_norm if k and k in rem]
    if cands:
        return ph, ids_norm[max(cands, key=len)], "exact"
    m = difflib.get_close_matches(rem, list(ids_norm), n=1, cutoff=0.6)
    return ph, (ids_norm[m[0]] if m else None), ("fuzzy" if m else "none")


def load_labels(root) -> pd.DataFrame:
    """Training labels: sample_id and the eleven sieve columns (percent passing), one row per soil."""
    df = pd.read_csv(Path(root) / "Training_labels_updated.csv", dtype={ID_COLUMN: str})
    df.columns = [str(c).strip() for c in df.columns]
    return df[[ID_COLUMN] + SIEVE_COLUMNS]


def load_test_ids(root) -> list:
    """Sample IDs Kaggle expects, in the order of sample_submission.csv."""
    return pd.read_csv(Path(root) / "sample_submission.csv", dtype={ID_COLUMN: str})[ID_COLUMN].tolist()


def photo_table(root) -> pd.DataFrame:
    """One row per photo of both splits: split, file, name, sample_id, phone, w, h, rho, match.
    Raises if any photo cannot be resolved; warns for every name resolved by the fuzzy fallback."""
    root = Path(root)
    ppm = pd.read_csv(root / "ppm_updated.csv")
    aliases = sorted({(norm(v), i) for i, r in ppm.iterrows() for v in (r.phone, r.camera)},
                     key=lambda t: -len(t[0]))
    rows = []
    for split, ids in (("train", load_labels(root)[ID_COLUMN]), ("test", load_test_ids(root))):
        idn = {norm(i): i for i in ids}
        for f in sorted((root / SPLIT_FOLDERS[split]).rglob("*")):
            if f.suffix.lower() not in IMG_EXT:
                continue
            ri, sid, how = parse_name(f.name, idn, aliases)
            with Image.open(f) as im:                        # reads only the header
                w, h = im.size
            if ri is None:                                   # fallback: match by image size
                hit = ppm.index[[(set((w, h)) == {a, b}) for a, b in zip(ppm.width, ppm.height)]]
                ri = hit[0] if len(hit) == 1 else None
            rho = None
            if ri is not None:                               # long-side ratio: orientation-proof
                rho = true_scale(float(ppm.ppm[ri]), (w, h), (ppm.width[ri], ppm.height[ri]))
            rows.append(dict(split=split, file=str(f), name=f.name, sample_id=sid,
                             phone=None if ri is None else ppm.phone[ri], w=w, h=h, rho=rho, match=how))
    info = pd.DataFrame(rows)
    bad = info[info.sample_id.isna() | info.rho.isna()]
    if len(bad):
        raise ValueError("could not resolve sample_id / phone for:\n" + bad[["split", "name", "sample_id", "phone"]].to_string())
    fuzzy = info[info.match == "fuzzy"]
    if len(fuzzy):
        warnings.warn("sample_id guessed by closest-name match, check these:\n"
                      + fuzzy[["split", "name", "sample_id"]].to_string())
    return info


def list_photos(root, split) -> pd.DataFrame:
    """Photos of one split ("train" or "test") sorted by sample_id then file name. Columns:
    path, file, split, sample_id, phone, width_px, height_px, rho_px_per_mm."""
    d = photo_table(root)
    d = d[d.split == split].rename(columns={"file": "path", "name": "file", "w": "width_px",
                                            "h": "height_px", "rho": "rho_px_per_mm"})
    cols = ["path", "file", "split", ID_COLUMN, "phone", "width_px", "height_px", "rho_px_per_mm"]
    return d[cols].sort_values([ID_COLUMN, "file"]).reset_index(drop=True)


def load_kaggle(root):
    """Everything the pipeline needs: train/test Samples, training curves (n, 11), info table."""
    root = Path(root)
    lab = load_labels(root)
    sub = pd.read_csv(root / "sample_submission.csv", dtype={ID_COLUMN: str})
    info = photo_table(root)

    def build(split, ids):
        d = info[info.split == split]
        out = []
        for sid in ids:
            g = d[d.sample_id == sid]
            if g.empty:
                raise ValueError(f"no photos found for {split} sample {sid!r}")
            out.append(Sample(str(sid), list(zip(g.file, g.rho))))
        return out

    tr_ids = lab[ID_COLUMN].tolist()
    curves = np.stack([clean_curve(r) for r in lab[SIEVE_COLUMNS].to_numpy(float)])
    return dict(train=build("train", tr_ids), curves=curves,
                test=build("test", sub[ID_COLUMN].tolist()),
                sub_columns=list(sub.columns), info=info, labels=lab)
