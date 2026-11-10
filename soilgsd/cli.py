import argparse
import numpy as np
import pandas as pd
from dataclasses import replace
from .config import Config
from .crossover import adaptive_crossover
from .pipeline import extract, SoilGSD
from .segment import ClassicalSegmenter
from .cv import loso


def main():
    ap = argparse.ArgumentParser("soilgsd")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("demo", "check", "loso", "predict"):
        p = sub.add_parser(name)
        p.add_argument("--root", default="soil-grain-size-from-photos", help="Kaggle data folder")
        p.add_argument("--sam-ckpt", default=None, help="SAM checkpoint (omit -> classical segmenter)")
        p.add_argument("--sam-device", default="cuda")
        p.add_argument("--max-side", type=int, default=4096, help="downscale photos above this long side (0=off)")
        if name == "predict":
            p.add_argument("--out", default="submission.csv")
            p.add_argument("--blend", default="adaptive", choices=["adaptive", "pure_b", "hard"],
                           help="Path C mode (pure_b = Path B only, w=0)")
    a = ap.parse_args()
    cfg = replace(Config(), max_side=a.max_side)

    if a.cmd == "demo":
        from .synth import make_dataset
        S, C = make_dataset(12)
        seg = ClassicalSegmenter()
        X = [extract(s, seg, cfg) for s in S]
        for k, (m, _) in loso(X, C, cfg).items():
            print(f"{k:11s} LOSO-EMD = {m:7.2f}")
        return

    from .data import load_kaggle
    D = load_kaggle(a.root)
    info = D["info"]
    if a.cmd == "check":
        info = info.assign(dc=[adaptive_crossover(r, cfg.tau_px) for r in info.rho])
        pd.set_option("display.width", 200, "display.max_rows", 500, "display.max_colwidth", 48)
        print(info[["split", "name", "sample_id", "phone", "w", "h", "rho", "dc"]].to_string())
        print("\nphotos per sample (train):", info[info.split == "train"].groupby("sample_id").size().describe()[["min", "max"]].to_dict())
        print("train: %d samples / %d photos | test: %d samples / %d photos" % (
            len(D["train"]), (info.split == "train").sum(), len(D["test"]), (info.split == "test").sum()))
        return

    if a.sam_ckpt:
        from .segment import SamSegmenter
        seg = SamSegmenter(a.sam_ckpt, device=a.sam_device)
    else:
        seg = ClassicalSegmenter()

    X = [extract(s, seg, cfg) for s in D["train"]]
    if a.cmd == "loso":
        for k, (m, _) in loso(X, D["curves"], cfg).items():
            print(f"{k:11s} LOSO-EMD = {m:7.2f}")
    else:
        m = SoilGSD(cfg).fit(X, D["curves"])
        rows = [[s.sample_id, *np.round(m.predict(extract(s, seg, cfg), a.blend)["final"], 4)] for s in D["test"]]
        pd.DataFrame(rows, columns=D["sub_columns"]).to_csv(a.out, index=False)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
