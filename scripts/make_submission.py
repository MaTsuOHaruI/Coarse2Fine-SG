"""Write a Kaggle submission file.

Models: mean_curve (mean training curve for every test soil), or the pipeline fitted on all
24 training soils: pure_b (Path B only), adaptive (stitched), hard (w = 1).

Usage:  uv run python scripts/make_submission.py [--model mean_curve|pure_b|adaptive|hard]
                                                  [--segmenter classical|sam] [--dino]
Writes: runs/<stamp>_submission_<model>/submission.csv

The file is checked (range, non-decreasing, 100 at 200 mm, sample IDs) before it
is saved. Uploading to Kaggle is done by hand.
"""

from soilgsd import baselines, config, data
from soilgsd.config import Config, SIEVE_COLUMNS
from soilgsd.pipeline import SoilGSD, add_tool_args, extract, make_tools
from soilgsd.submit import write_submission


def main():
    parser = add_tool_args(config.base_parser(__doc__))
    parser.add_argument("--model", default="mean_curve", choices=["mean_curve", "pure_b", "adaptive", "hard"])
    args = parser.parse_args()
    ycfg = config.load_config(args.config)
    config.set_seed(ycfg["seed"])
    kaggle_dir = config.kaggle_dir(ycfg)
    test_ids = data.load_test_ids(kaggle_dir)

    if args.model == "mean_curve":
        # The final model is fitted on all 24 training soils; here that is just their mean.
        labels = data.load_labels(kaggle_dir)
        curve = baselines.mean_curve(labels[SIEVE_COLUMNS].to_numpy(dtype=float))
        curves_by_id = {sample_id: curve for sample_id in test_ids}
    else:
        cfg = Config.from_yaml(ycfg)
        seg, emb = make_tools(ycfg, args)
        cache = config.cache_dir(ycfg)
        D = data.load_kaggle(kaggle_dir)
        X = [extract(s, seg, cfg, emb, cache_dir=cache) for s in D["train"]]
        model = SoilGSD(cfg).fit(X, D["curves"])
        curves_by_id = {s.sample_id: model.predict(extract(s, seg, cfg, emb, cache_dir=cache), args.model)["final"]
                        for s in D["test"]}

    run_dir = config.new_run_dir(ycfg, f"submission_{args.model}")
    df = write_submission(curves_by_id, test_ids, run_dir / "submission.csv")
    print(df.round(2).to_string(index=False))
    print(f"\nvalid submission with {len(df)} soils written to {run_dir / 'submission.csv'}")


if __name__ == "__main__":
    main()
