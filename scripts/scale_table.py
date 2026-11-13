"""List every photo with its soil, phone, size and true scale (px/mm).

Usage:  uv run python scripts/scale_table.py [--config configs/default.yaml]
Writes: runs/<stamp>_scale_table/photos.csv
"""

import pandas as pd

from soilgsd import config, data
from soilgsd.crossover import adaptive_crossover


def main():
    args = config.base_parser(__doc__).parse_args()
    cfg = config.load_config(args.config)
    kaggle_dir = config.kaggle_dir(cfg)
    tau_px = cfg["preprocess"]["tau_px"]

    photos = pd.concat([data.list_photos(kaggle_dir, "train"), data.list_photos(kaggle_dir, "test")])
    photos["d_c_mm"] = [adaptive_crossover(rho, tau_px) for rho in photos["rho_px_per_mm"]]

    summary = (
        photos.groupby(["split", "phone", "width_px", "height_px"])
        .agg(photos=("file", "count"), soils=("sample_id", "nunique"),
             rho_px_per_mm=("rho_px_per_mm", "mean"), d_c_mm=("d_c_mm", "max"))
        .round(2)
    )
    print(summary.to_string())

    run_dir = config.new_run_dir(cfg, "scale_table")
    photos.to_csv(run_dir / "photos.csv", index=False)
    print(f"\n{len(photos)} photos written to {run_dir / 'photos.csv'}")


if __name__ == "__main__":
    main()
