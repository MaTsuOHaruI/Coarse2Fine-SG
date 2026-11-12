"""Precompute masks and features for every photo into the cache (the slow, GPU part).

Resumable: photos already in the cache are skipped, so on Colab just rerun after a timeout.
Later steps (run_loso.py, make_submission.py) read the cache and run on a CPU.

Usage:  uv run python scripts/make_cache.py [--segmenter classical|sam] [--dino] [--split train|test|all] [--limit N]
Writes: <cache_dir>/masks/<segmenter>_<prep>/<photo>.npz, <cache_dir>/features/<embedder>_<prep>_rho<scale>/<photo>.npy
"""

import time

from soilgsd import config
from soilgsd.config import Config
from soilgsd.data import load_kaggle
from soilgsd.pipeline import Sample, add_tool_args, extract, make_tools


def main():
    parser = add_tool_args(config.base_parser(__doc__))
    parser.add_argument("--split", default="all", choices=["train", "test", "all"])
    args = parser.parse_args()
    ycfg = config.load_config(args.config)
    cfg = Config.from_yaml(ycfg)
    cache = config.cache_dir(ycfg)
    seg, emb = make_tools(ycfg, args)

    D = load_kaggle(config.kaggle_dir(ycfg))
    samples = (D["train"] if args.split in ("train", "all") else []) + (D["test"] if args.split in ("test", "all") else [])
    photos = [(s.sample_id, p) for s in samples for p in s.photos]   # one photo at a time
    if args.limit:
        photos = photos[: args.limit]

    print(f"{len(photos)} photos | segmenter {seg.name} | embedder {getattr(emb, 'name', '')} | cache {cache}")
    for i, (sample_id, photo) in enumerate(photos, 1):
        t0 = time.time()
        extract(Sample(sample_id, [photo]), seg, cfg, emb, cache_dir=cache)
        print(f"[{i}/{len(photos)}] {sample_id}  {time.time() - t0:6.1f} s")


if __name__ == "__main__":
    main()
