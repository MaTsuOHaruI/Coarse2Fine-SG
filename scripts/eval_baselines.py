"""Score the two reference baselines on the 24 Kaggle training soils.

Expected: constant baseline 102.0 (on the training labels),
          mean training curve 89.9 (leave-one-soil-out).

Usage:  uv run python scripts/eval_baselines.py [--config configs/default.yaml]
Writes: runs/<stamp>_eval_baselines/{summary.csv, per_soil_mean_curve.csv}
"""

import numpy as np
import pandas as pd

from soilgsd import baselines, config, data
from soilgsd.config import SIEVE_COLUMNS
from soilgsd.cv import leave_one_soil_out
from soilgsd.metrics import mean_emd


def main():
    args = config.base_parser(__doc__).parse_args()
    cfg = config.load_config(args.config)
    config.set_seed(cfg["seed"])
    form = cfg["metric"]["form"]

    labels = data.load_labels(config.kaggle_dir(cfg))
    P_true = labels[SIEVE_COLUMNS].to_numpy(dtype=float)

    # 1. Constant baseline: the same fixed curve for every soil, no fitting, no CV needed.
    constant = np.tile(baselines.constant_curve(), (len(labels), 1))
    constant_score = mean_emd(P_true, constant, form=form)

    # 2. Mean curve: fitted on the training fold only, scored on the held-out soil.
    def mean_curve_predictor(train_labels, held_out_id):
        return baselines.mean_curve(train_labels[SIEVE_COLUMNS].to_numpy(dtype=float))

    per_soil = leave_one_soil_out(labels, mean_curve_predictor, form=form)
    mean_curve_score = per_soil["emd"].mean()

    summary = pd.DataFrame(
        [
            {"baseline": "constant", "protocol": "training labels", "emd": constant_score, "reference": 102.0},
            {"baseline": "mean curve", "protocol": "leave-one-soil-out", "emd": mean_curve_score, "reference": 89.9},
        ]
    )
    print(f"soils: {len(labels)}   metric form: {form}")
    print(summary.round(2).to_string(index=False))

    run_dir = config.new_run_dir(cfg, "eval_baselines")
    summary.to_csv(run_dir / "summary.csv", index=False)
    per_soil.to_csv(run_dir / "per_soil_mean_curve.csv", index=False)
    print(f"\nwritten to {run_dir}")


if __name__ == "__main__":
    main()
