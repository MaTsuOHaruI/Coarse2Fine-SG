import argparse
import numpy as np
import pandas as pd
from dataclasses import replace
from . import config
from .config import Config
from .crossover import adaptive_crossover
from .pipeline import extract, SoilGSD
from .segment import ClassicalSegmenter, make_segmenter
from .cv import loso
from .submit import write_submission


def main():
    ap = argparse.ArgumentParser("soilgsd")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("demo", "check", "loso", "predict"):
        p = sub.add_parser(name)
        p.add_argument("--config", default=None, help="YAML config (default configs/default.yaml)")
        p.add_argument("--root", default=None, help="Kaggle data folder (default: paths.kaggle_dir in the config)")
        p.add_argument("--sam-ckpt", default=None, help="SAM checkpoint (omit -> classical segmenter)")
        p.add_argument("--sam-device", default="cuda")
        p.add_argument("--max-side", type=int, default=None, help="downscale photos above this long side (0=off)")
        p.add_argument("--no-cache", action="store_true", help="do not read or write cache/ masks and features")
        if name == "predict":
            p.add_argument("--out", default="submission.csv")
            p.add_argument("--blend", default="adaptive", choices=["adaptive", "pure_b", "hard"],
                           help="Path C mode (pure_b = Path B only, w=0)")
    a = ap.parse_args()
    ycfg = config.load_config(a.config)
    cfg = Config.from_yaml(ycfg)
    if a.max_side is not None:
        cfg = replace(cfg, max_side=a.max_side)
    root = a.root or config.kaggle_dir(ycfg)
    cache = None if a.no_cache else config.cache_dir(ycfg)

    if a.cmd == "demo":
        from .synth import make_dataset
        S, C = make_dataset(12)
        seg = ClassicalSegmenter()
        X = [extract(s, seg, cfg) for s in S]
        for k, (m, _) in loso(X, C, cfg).items():
            print(f"{k:11s} LOSO-EMD = {m:7.2f}")
        return

    from .data import load_kaggle
    D = load_kaggle(root)
    info = D["info"]
    if a.cmd == "check":
        info = info.assign(dc=[adaptive_crossover(r, cfg.tau_px) for r in info.rho])
        pd.set_option("display.width", 200, "display.max_rows", 500, "display.max_colwidth", 48)
        print(info[["split", "name", "sample_id", "phone", "w", "h", "rho", "dc"]].to_string())
        print("\nphotos per sample (train):", info[info.split == "train"].groupby("sample_id").size().describe()[["min", "max"]].to_dict())
        print("train: %d samples / %d photos | test: %d samples / %d photos" % (
            len(D["train"]), (info.split == "train").sum(), len(D["test"]), (info.split == "test").sum()))
        return

    pre = ycfg.get("preprocess", {})
    seg = make_segmenter("sam" if a.sam_ckpt else "classical", a.sam_ckpt, a.sam_device,
                         ycfg.get("sam", {}).get("model_type", "vit_b"),
                         pre.get("tile_px", 1024), pre.get("tile_overlap_px", 256))

    X = [extract(s, seg, cfg, cache_dir=cache) for s in D["train"]]
    if a.cmd == "loso":
        for k, (m, _) in loso(X, D["curves"], cfg).items():
            print(f"{k:11s} LOSO-EMD = {m:7.2f}")
    else:
        m = SoilGSD(cfg).fit(X, D["curves"])
        curves = {s.sample_id: np.round(m.predict(extract(s, seg, cfg, cache_dir=cache), a.blend)["final"], 4)
                  for s in D["test"]}
        write_submission(curves, [s.sample_id for s in D["test"]], a.out)   # validity checks before saving
        print("wrote", a.out)


if __name__ == "__main__":
    main()
