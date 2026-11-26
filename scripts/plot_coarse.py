"""Path A coarse curve against the true curve above the crossover d_c, per training soil.

Both curves are conditional on the coarse fraction: the share (in %) of the mass above d_c that
passes each sieve. Path A gives this directly (CoarseResult.G); for the lab curve P it is
(P - P(d_c)) / (100 - P(d_c)) * 100. Only grains with b >= d_c count, as in path_c.blend.

Reads masks and features from the cache only (make_cache.py); it never runs a segmenter, so no
SAM checkpoint or GPU is needed. A photo without cached masks stops the script.

Usage:  uv run python scripts/plot_coarse.py [--segmenter sam|classical] [--min-coarse-pct 5]
                                             [--limit N soils]
Writes: runs/<stamp>_coarse_<segmenter>/coarse_curves.png  (soils with at least --min-coarse-pct
                                                            of their mass above d_c)
        runs/<stamp>_coarse_<segmenter>/coarse_summary.csv (every training soil: n grains,
                                                            n_eff, w, true and Path A % passing)
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from soilgsd import config
from soilgsd.config import ID_COLUMN, SIEVES, Config
from soilgsd.data import load_kaggle
from soilgsd.features import SimpleEmbedder
from soilgsd.pipeline import extract


class CacheOnlySegmenter:
    """Stands in for a segmenter in pipeline.extract: same cache name, but never segments."""

    def __init__(self, name: str):
        self.name = name

    def segment(self, rgb):
        raise SystemExit(f"no cached masks for a photo under masks/{self.name}_*; run make_cache.py first")


def segmenter_name(ycfg: dict, segmenter: str) -> str:
    """The segmenter.name that make_tools would build (it is the mask cache folder prefix)."""
    if segmenter == "classical":
        return "classical_d8_a30"     # ClassicalSegmenter() defaults
    pre, sam = ycfg.get("preprocess", {}), ycfg.get("sam", {})
    return f"sam_{sam.get('model_type', 'vit_b')}_t{pre.get('tile_px', 1024)}_o{pre.get('tile_overlap_px', 256)}"


def true_coarse_pct(curve_pct: np.ndarray, dc: float) -> np.ndarray:
    """Lab curve (11,) in % -> % of the mass above dc passing each sieve >= dc (nan below dc)."""
    i_c = int(np.where(SIEVES == dc)[0][0])
    out = np.full(len(SIEVES), np.nan)
    rest = 100.0 - curve_pct[i_c]
    if rest > 0:
        out[i_c:] = (curve_pct[i_c:] - curve_pct[i_c]) / rest * 100.0
    return out


def summarise(x, curve_pct: np.ndarray, kappa: float) -> dict:
    """One row: coarse share, grain counts, stitch weight, and true vs Path A % passing per sieve >= dc."""
    c = x.coarse.above(x.dc)
    n_eff = c.n_eff
    i_c = int(np.where(SIEVES == x.dc)[0][0])
    true_pct = true_coarse_pct(curve_pct, x.dc)
    pred_pct = c.G(x.dc) * 100.0
    row = {ID_COLUMN: x.sample_id, "dc_mm": x.dc, "coarse_mass_pct": round(100.0 - curve_pct[i_c], 2),
           "n_grains": int(c.b_mm.size), "n_eff": round(n_eff, 1),
           "w": round(n_eff / (n_eff + kappa), 3) if n_eff > 0 else 0.0,
           "max_b_mm": round(float(c.b_mm.max()), 1) if c.b_mm.size else 0.0}
    for s, t, p in zip(SIEVES[i_c + 1:-1], true_pct[i_c + 1:-1], pred_pct[i_c + 1:-1]):
        row[f"true_{s:g}"] = round(t, 1)
        row[f"pathA_{s:g}"] = round(p, 1)
    return row, true_pct, pred_pct


def plot_grid(items: list, title: str):
    """items: [(row, true_pct (11,), pred_pct (11,))]. Returns a matplotlib Figure."""
    n = len(items)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.0 * rows), squeeze=False, sharey=True)
    for ax, (row, t, p) in zip(axes.flat, items):
        k = SIEVES >= row["dc_mm"]
        ax.plot(SIEVES[k], t[k], "k-o", ms=3, label="lab sieving")
        if np.isfinite(p[k]).any():
            ax.plot(SIEVES[k], p[k], "-s", color="tab:green", ms=3, label="Path A (SAM grains)")
        ax.set_xscale("log")
        ax.set_ylim(-5, 105)
        ax.set_title(f"{row[ID_COLUMN]}  ({row['coarse_mass_pct']:.0f} % > {row['dc_mm']:g} mm)", fontsize=9)
        ax.text(0.97, 0.05, f"n={row['n_grains']}  n_eff={row['n_eff']:.0f}\nw={row['w']:.2f}  "
                f"max b={row['max_b_mm']:.0f} mm", transform=ax.transAxes, ha="right", fontsize=7)
        ax.grid(alpha=0.3)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("% of coarse mass passing")
    for ax in axes[-1]:
        ax.set_xlabel("sieve [mm]")
    axes.flat[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def main():
    parser = config.base_parser(__doc__)
    parser.add_argument("--segmenter", default="sam", choices=["sam", "classical"],
                        help="which mask cache to read")
    parser.add_argument("--min-coarse-pct", type=float, default=5.0,
                        help="plot only soils with at least this %% of their mass above d_c")
    args = parser.parse_args()
    ycfg = config.load_config(args.config)
    cfg = Config.from_yaml(ycfg)
    seg = CacheOnlySegmenter(segmenter_name(ycfg, args.segmenter))

    D = load_kaggle(config.kaggle_dir(ycfg))
    train, curves = D["train"], D["curves"]
    if args.limit:
        train, curves = train[: args.limit], curves[: args.limit]

    rows, items = [], []
    for s, curve in zip(train, curves):
        x = extract(s, seg, cfg, SimpleEmbedder(), cache_dir=config.cache_dir(ycfg))
        if not np.isfinite(x.dc):
            print(f"{s.sample_id}: no finite crossover, skipped")
            continue
        row, t, p = summarise(x, curve, cfg.kappa)
        rows.append(row)
        if row["coarse_mass_pct"] >= args.min_coarse_pct:
            items.append((row, t, p))

    run_dir = config.new_run_dir(ycfg, f"coarse_{args.segmenter}")
    table = pd.DataFrame(rows).sort_values("coarse_mass_pct", ascending=False)
    table.to_csv(run_dir / "coarse_summary.csv", index=False)
    print(table.to_string(index=False))
    if items:
        items.sort(key=lambda it: -it[0]["coarse_mass_pct"])
        fig = plot_grid(items, f"Coarse fraction above d_c: lab vs Path A ({seg.name}, tau {cfg.tau_px:g} px, "
                               f"kappa {cfg.kappa:g})")
        fig.savefig(run_dir / "coarse_curves.png", dpi=150)
    print(f"\n{len(items)} soils plotted | written to {run_dir}")


if __name__ == "__main__":
    main()
