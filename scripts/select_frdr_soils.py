"""Choose the FRDR soils and photos for the resolution simulation, from the labels alone.

The selection is fixed before any photo is segmented, so it cannot be tuned on results.

Input: the FRDR Photogranulometry v1.0 file metadata/psd_mass_path_80_to_0_08mm_no_ballast.csv
(--frdr-csv). One row per image: sample_id, image_path, dry_or_moist, mass, water, and the
cumulative % passing at 16 sieves in columns "%_80mm" ... "%_0.08mm". The label is the same on
every image of a sample. Image names are <sample>_<type>_<dry|moist>_tray<NNN>_<NN>.jpg.

Steps:
  1. Photos: keep images in one moisture state (--state, default dry: most Kaggle photos are dry).
     A soil needs --trays different trays (default 4). Trays are separate quarters of the sample,
     so they are different spreads; images within one tray may show the same spread.
  2. Coverage: the label needs a value at 2 mm and at 6.3 mm (FRDR has 2, 5 and 10 mm).
  3. Coarse share = 100 - P(2 mm), with P read off the label by monotone PCHIP in log10(mm).
     Also reported: the 2 to 6.3 mm class, P(6.3) - P(2), which is what d_c = 2 mm adds.
  4. Stratify by coarse share (bins below) and draw at random within each bin, fixed seed.
     A bin with too few soils gives all it has; its shortfall moves to the nearest bin with spare soils.
  5. For each selected soil, draw --trays trays and one image per tray.

Usage:  uv run python scripts/select_frdr_soils.py [--frdr-csv data/frdr/psd_mass_path_80_to_0_08mm_no_ballast.csv]
Writes: runs/<stamp>_select_frdr/{selected_soils.csv, eligible_soils.csv, bin_counts.csv, selected_photos.csv}
        and copies of the soil and photo lists (ids, numbers, image paths) to --out and
        --out-photos, tracked files. An existing --out is never overwritten without --force.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from soilgsd import config

# Coarse share bins, in percent of mass above 2 mm: (low, high, soils to draw). Written down before
# any FRDR result exists. The under-10 bin holds the fine controls.
COARSE_BINS = [
    (0.0, 10.0, 4),
    (10.0, 30.0, 9),
    (30.0, 50.0, 9),
    (50.0, 70.0, 9),
    (70.0, 100.0001, 9),
]
CROSSOVER_LOW_MM = 2.0
CROSSOVER_HIGH_MM = 6.3


def sieve_columns(df, id_col):
    """Columns whose name parses as a number: the sieve openings in mm, sorted ascending.

    Returns (list of column names, numpy array of sizes in mm), same order.
    """
    pairs = []
    for col in df.columns:
        if col == id_col:
            continue
        try:
            pairs.append((float(col), col))
        except ValueError:
            continue
    pairs.sort()
    return [c for _, c in pairs], np.array([s for s, _ in pairs], dtype=float)


def passing_at(sizes_mm, passing_pct, x_mm):
    """Percentage passing at x_mm, by monotone PCHIP in log10(size).

    sizes_mm, passing_pct: 1-D arrays of the same length (NaN entries are skipped).
    Returns NaN when x_mm lies outside the labelled range, except that above the largest sieve
    a curve already at 100 stays at 100.
    """
    ok = ~np.isnan(sizes_mm) & ~np.isnan(passing_pct)
    s = np.asarray(sizes_mm, float)[ok]
    p = np.clip(np.asarray(passing_pct, float)[ok], 0.0, 100.0)
    if len(s) == 0:
        return np.nan
    if x_mm > s.max():
        return 100.0 if p[np.argmax(s)] >= 100.0 - 1e-9 else np.nan
    if x_mm < s.min():
        return np.nan
    if len(s) == 1:
        return float(p[0])
    p = np.maximum.accumulate(p)  # a label that dips by rounding would break monotone interpolation
    return float(PchipInterpolator(np.log10(s), p)(np.log10(x_mm)))


def coarse_table(labels, id_col):
    """Per soil: P(2 mm), P(6.3 mm), coarse_pct = 100 - P(2), class_2_6_pct = P(6.3) - P(2).

    labels: DataFrame, one row per soil (see module docstring).
    Returns a DataFrame with id_col and those four columns; soils without coverage get NaN.
    """
    cols, sizes = sieve_columns(labels, id_col)
    if len(cols) < 2:
        raise ValueError(f"need at least two sieve columns named in mm, found {cols}")
    rows = []
    for _, r in labels.iterrows():
        p = r[cols].to_numpy(dtype=float)
        p2 = passing_at(sizes, p, CROSSOVER_LOW_MM)
        p63 = passing_at(sizes, p, CROSSOVER_HIGH_MM)
        rows.append(
            {
                id_col: r[id_col],
                "p_2mm_pct": p2,
                "p_6_3mm_pct": p63,
                "coarse_pct": 100.0 - p2,
                "class_2_6_pct": p63 - p2,
            }
        )
    return pd.DataFrame(rows)


def assign_bin(coarse_pct, p_6_3mm_pct):
    """Index into COARSE_BINS for one soil, or -1 when it lacks coverage.

    A soil needs a value at 2 mm (coarse_pct) and at 6.3 mm (p_6_3mm_pct); without the 6.3 mm value
    it has no 2 to 6.3 mm class to test d_c = 2 mm on.
    """
    if np.isnan(coarse_pct) or np.isnan(p_6_3mm_pct):
        return -1
    for i, (low, high, _) in enumerate(COARSE_BINS):
        if low <= coarse_pct < high:
            return i
    return -1


def plan_counts(available):
    """How many soils to draw per bin.

    available: list of eligible soils per bin, same order as COARSE_BINS.
    Returns a list of counts. A bin short of its target gives all it has; the shortfall goes to
    the nearest bins (closest first, the higher bin first on a tie) that still have spare soils.
    """
    want = [n for _, _, n in COARSE_BINS]
    take = [min(w, a) for w, a in zip(want, available)]
    shortfalls = [w - t for w, t in zip(want, take)]  # fixed before any bin receives extra soils
    for i, short in enumerate(shortfalls):
        for dist in range(1, len(want)):
            for j in (i + dist, i - dist):
                if short == 0:
                    break
                if 0 <= j < len(want):
                    extra = min(short, available[j] - take[j])
                    take[j] += extra
                    short -= extra
    return take


def select_soils(table, id_col, seed):
    """Stratified random draw. table: output of coarse_table (with a "bin" column added).

    Returns the selected rows, sorted by bin then coarse share.
    """
    eligible = table[table["bin"] >= 0]
    available = [int((eligible["bin"] == i).sum()) for i in range(len(COARSE_BINS))]
    counts = plan_counts(available)
    rng = np.random.default_rng(seed)
    parts = []
    for i, n in enumerate(counts):
        group = eligible[eligible["bin"] == i].sort_values(id_col)  # fixed order before the draw
        if n > 0:
            idx = rng.choice(len(group), size=n, replace=False)
            parts.append(group.iloc[np.sort(idx)])
    picked = pd.concat(parts) if parts else eligible.iloc[:0]
    return picked.sort_values(["bin", "coarse_pct"]).reset_index(drop=True), available, counts


FRDR_ID = "sample_id"
FRDR_NAME = r"(?P<sample>\d{4})_(?P<type>[a-z]+)_(?P<state>dry|moist)_tray(?P<tray>\d{3})_(?P<index>\d{2})"


def load_frdr(csv_path, state):
    """Read the FRDR per-image CSV.

    Returns (labels, photos):
      labels: one row per sample, column sample_id (str, e.g. "0078") plus one column per sieve
              named by its size in mm ("80", ..., "0.08"), cumulative % passing.
      photos: one row per image in the given moisture state, columns sample_id, image_path,
              type, tray (str), index (str).
    """
    df = pd.read_csv(csv_path, dtype={FRDR_ID: str})  # keeps the leading zeros of the ids
    sieve_cols = [c for c in df.columns if c.startswith("%_") and c.endswith("mm") and c != "%_water"]
    if len(sieve_cols) != 16:
        raise SystemExit(f"expected 16 sieve columns, found {len(sieve_cols)}: {sieve_cols}")
    per_sample = df.groupby(FRDR_ID)[sieve_cols].nunique().max().max()
    if per_sample > 1:
        raise SystemExit("the label differs between images of one sample: check the file")
    labels = df.drop_duplicates(FRDR_ID)[[FRDR_ID] + sieve_cols].rename(
        columns={c: c[2:-2] for c in sieve_cols}  # "%_2.5mm" -> "2.5"
    )

    parts = df["image_path"].str.extract(FRDR_NAME)
    if parts["sample"].isna().any():
        raise SystemExit(f"{int(parts['sample'].isna().sum())} image paths do not follow the FRDR naming")
    if (parts["sample"] != df[FRDR_ID]).any() or (parts["state"] != df["dry_or_moist"]).any():
        raise SystemExit("image names disagree with the sample_id / dry_or_moist columns")
    photos = pd.DataFrame(
        {FRDR_ID: df[FRDR_ID], "image_path": df["image_path"], "type": parts["type"],
         "state": parts["state"], "tray": parts["tray"], "index": parts["index"]}
    )
    photos = photos[photos["state"] == state].drop(columns="state").drop_duplicates().reset_index(drop=True)
    return labels.reset_index(drop=True), photos


def select_photos(photos, soil_ids, n_trays, seed):
    """For each selected soil: draw n_trays of its trays, then one image per tray.

    photos: output of load_frdr; soil_ids: selected sample ids.
    Returns a DataFrame with the same columns as photos, n_trays rows per soil.
    """
    if len(soil_ids) == 0:
        raise SystemExit("no soil selected, so no photos to draw")
    rng = np.random.default_rng(seed + 1)
    parts = []
    for sid in soil_ids:
        soil = photos[photos[FRDR_ID] == sid]
        trays = np.sort(soil["tray"].unique())
        for tray in np.sort(rng.choice(trays, size=n_trays, replace=False)):
            images = soil[soil["tray"] == tray].sort_values("image_path")
            parts.append(images.iloc[[rng.integers(len(images))]])
    return pd.concat(parts).reset_index(drop=True)


def main():
    parser = config.base_parser(__doc__)
    parser.add_argument("--frdr-csv", default="data/frdr/psd_mass_path_80_to_0_08mm_no_ballast.csv")
    parser.add_argument("--state", default="dry", choices=["dry", "moist"])
    parser.add_argument("--trays", type=int, default=4, help="trays (= photos) per soil")
    parser.add_argument("--out", default="configs/frdr_selected_soils.csv", help="tracked copy of the soil list")
    parser.add_argument("--out-photos", default="configs/frdr_selected_photos.csv", help="tracked copy of the photo list")
    parser.add_argument("--force", action="store_true", help="overwrite an existing selection")
    args = parser.parse_args()
    cfg = config.load_config(args.config)
    seed = cfg["seed"]
    config.set_seed(seed)
    if args.trays < 1:
        raise SystemExit(f"--trays must be at least 1, got {args.trays}")

    out, out_photos = config._resolve(args.out), config._resolve(args.out_photos)
    existing = [str(f) for f in (out, out_photos) if f.exists()]
    if existing and not args.force:
        raise SystemExit(f"{', '.join(existing)} exists: the selection is fixed. Pass --force only to redo it on purpose.")

    labels, photos = load_frdr(config._resolve(args.frdr_csv), args.state)
    table = coarse_table(labels, FRDR_ID)
    table["bin"] = [assign_bin(c, p) for c, p in zip(table["coarse_pct"], table["p_6_3mm_pct"])]
    n_no_coverage = int((table["bin"] < 0).sum())

    n_trays = photos.groupby(FRDR_ID)["tray"].nunique()
    types = photos.drop_duplicates(FRDR_ID).set_index(FRDR_ID)["type"]
    table["type"] = table[FRDR_ID].map(types)
    table["n_trays"] = table[FRDR_ID].map(n_trays).fillna(0).astype(int)
    too_few = table["n_trays"] < args.trays
    print(f"soils without {args.trays} {args.state} trays: {int((too_few & (table['bin'] >= 0)).sum())} (dropped)")
    table.loc[too_few, "bin"] = -1

    picked, available, counts = select_soils(table, FRDR_ID, seed)

    bin_counts = pd.DataFrame(
        {
            "coarse_pct_bin": [f"{lo:g} to {min(hi, 100):g}" for lo, hi, _ in COARSE_BINS],
            "target": [n for _, _, n in COARSE_BINS],
            "eligible": available,
            "selected": counts,
        }
    )
    print(f"soils in label file: {len(table)}   without coverage of 2 and 6.3 mm: {n_no_coverage}")
    print(bin_counts.to_string(index=False))
    print(f"selected: {len(picked)}   types: {picked['type'].value_counts().to_dict()}")
    print(f"with an empty 2 to 6.3 mm class (< 1 %): {int((picked['class_2_6_pct'] < 1).sum())}")

    chosen = select_photos(photos, picked[FRDR_ID], args.trays, seed)
    print(f"photos selected: {len(chosen)} ({args.state}, one per tray)")

    run_dir = config.new_run_dir(cfg, "select_frdr")
    keep = [FRDR_ID, "type", "bin", "coarse_pct", "class_2_6_pct", "p_2mm_pct", "p_6_3mm_pct"]
    picked[keep].round(3).to_csv(run_dir / "selected_soils.csv", index=False)
    table.round(3).to_csv(run_dir / "eligible_soils.csv", index=False)
    bin_counts.to_csv(run_dir / "bin_counts.csv", index=False)
    chosen.to_csv(run_dir / "selected_photos.csv", index=False)

    out.parent.mkdir(parents=True, exist_ok=True)
    out_photos.parent.mkdir(parents=True, exist_ok=True)
    picked[keep].round(3).to_csv(out, index=False)
    chosen.to_csv(out_photos, index=False)
    print(f"\nwritten to {run_dir}, {out} and {out_photos}")


if __name__ == "__main__":
    main()
