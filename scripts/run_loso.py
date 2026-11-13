"""Leave-one-soil-out score of the full pipeline on the 24 training soils.

Methods: mean_curve (reference 89.9), pure_b (Path B only, w = 0), adaptive (stitch with
w = n_eff / (n_eff + kappa)), hard (w = 1). Masks and features come from the cache
(make_cache.py); missing ones are computed and cached on the way.

Usage:  uv run python scripts/run_loso.py [--segmenter classical|sam] [--dino] [--limit N soils]
Writes: runs/<stamp>_loso_<segmenter>/{summary.csv, per_soil.csv}
"""

import pandas as pd

from soilgsd import config
from soilgsd.config import Config, ID_COLUMN
from soilgsd.cv import loso
from soilgsd.data import load_kaggle
from soilgsd.pipeline import add_tool_args, extract, make_tools


def main():
    args = add_tool_args(config.base_parser(__doc__)).parse_args()
    ycfg = config.load_config(args.config)
    config.set_seed(ycfg["seed"])
    cfg = Config.from_yaml(ycfg)
    seg, emb = make_tools(ycfg, args)

    D = load_kaggle(config.kaggle_dir(ycfg))
    train, curves = D["train"], D["curves"]
    if args.limit:
        train, curves = train[: args.limit], curves[: args.limit]
    X = [extract(s, seg, cfg, emb, cache_dir=config.cache_dir(ycfg)) for s in train]

    res = loso(X, curves, cfg)
    summary = pd.DataFrame([{"method": k, "loso_emd": m} for k, (m, _) in res.items()])
    per_soil = pd.DataFrame({ID_COLUMN: [x.sample_id for x in X], **{k: e for k, (_, e) in res.items()}})
    print(f"soils: {len(X)} | segmenter {seg.name} | embedder {getattr(emb, 'name', '')} | {cfg}")
    print(summary.round(2).to_string(index=False))

    run_dir = config.new_run_dir(ycfg, f"loso_{args.segmenter}")
    summary.to_csv(run_dir / "summary.csv", index=False)
    per_soil.to_csv(run_dir / "per_soil.csv", index=False)
    print(f"\nwritten to {run_dir}")


if __name__ == "__main__":
    main()
