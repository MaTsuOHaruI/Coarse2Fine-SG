"""Synthetic resolution simulation: does the stitch hold up when photos get sharper?

Training photos are about 4.6 px/mm (crossover d_c = 6.3 mm), test photos 13.9 to 19.5 px/mm
(d_c = 2 mm). No labelled photo exists at test resolution, so we make synthetic trays with exact
labels at a high native scale (25 px/mm), area-downsample them to each target scale, and run the
pipeline (classical segmenter, Path A, Path B, stitch) at every scale.

Two crossovers are compared at each scale:
  adaptive d_c   the crossover from crossover.adaptive_crossover(rho, tau) (2 mm from tau / 2 = 12.5 px/mm up)
  fixed d_c      6.3 mm everywhere (what the training labels can check)

Generator choices (kept here, not in soilgsd.synth, which stays as Haruki wrote it):
  - grains b from grain_min_mm to grain_max_mm, so the 2 to 6.3 mm class has real mass
  - true grain mass = a * b**p with p = cfg.mass_p, the same model Path A uses, so the run tests
    resolution only (it does not tune p)
  - grains never overlap and lie fully inside the centre crop (no clumps, no edge cuts)
  - fines are speckle noise, spread evenly over the classes below 2 mm (as in soilgsd.synth)

Usage:  uv run python scripts/resolution_sim.py [--n-soils 24] [--scales 25 19.5 13.9 4.6]
                                                [--overlays 2] [--limit N]
Writes: runs/<stamp>_resolution_sim/summary.csv     one row per scale
        runs/<stamp>_resolution_sim/per_soil.csv    one row per soil and scale (Path A)
        runs/<stamp>_resolution_sim/loso_per_soil.csv  per-soil LOSO EMD per method and d_c
        runs/<stamp>_resolution_sim/resolution_sim.png
        runs/<stamp>_resolution_sim/overlay_<soil>_rho<scale>.png   (first --overlays soils)
Prints the summary table and results.csv rows for the stitched EMDs.

Read the paired difference (delta_*: adaptive d_c minus fixed 6.3 mm, per soil) first. Only the
sieves from 6.3 mm up change between the two crossovers, but the difference mixes two things:
Path A's split of the 2 to 6.3 mm class, and the base curve's own value at 2 mm vs 6.3 mm (blend
anchors on base[d_c]). Three stitch bases separate them:
  pathB      the fitted Path B curve (the real pipeline)
  meanB      the training-fold mean curve (a constant base of known quality)
  oracleB    CONTROL, not a method: the held-out soil's true curve up to 2 mm, and above 2 mm the
             training-fold mean coarse shape. The 2 mm anchor is then exact, so the difference
             measures Path A's coarse curve (including its 2 to 6.3 mm split) against the mean
             coarse shape, free of the anchor effect.
The absolute EMDs are not comparable with Kaggle: Path B features cannot see the synthetic fines
(speckle is averaged away at the 4.6 px/mm working scale), so pure_b lands above the mean curve.
"""

import datetime
import subprocess
import time
from pathlib import Path
from dataclasses import replace

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_coarse import true_coarse_pct                    # sibling script (scripts/ is on sys.path)
from show_masks import COLOURS_RGB, MAX_OUT_SIDE_PX, add_legend, classify, draw
from soilgsd import config
from soilgsd.baselines import mean_curve
from soilgsd.config import ID_COLUMN, SIEVE_COLUMNS, SIEVES, Config
from soilgsd.crossover import adaptive_crossover
from soilgsd.cv import leave_one_soil_out, loso
from soilgsd.features import SimpleEmbedder, center_crop, resample_to
from soilgsd.metrics import clean_curve
from soilgsd.path_c import blend
from soilgsd.pipeline import Sample, _CachedRegions, add_tool_args, extract, make_tools

FIXED_DC_MM = 6.3
I_2MM, I_6MM = int(np.where(SIEVES == 2)[0][0]), int(np.where(SIEVES == 6.3)[0][0])
BACKGROUND_RGB = (205, 200, 190)                            # same tray colour as soilgsd.synth


def class_edges_mm(grain_min_mm: float, grain_max_mm: float) -> list:
    """Sieve classes the grains fall into, clipped to [grain_min_mm, grain_max_mm].
    Out: [(lo_mm, hi_mm), ...], smallest first, e.g. [(2, 6.3), (6.3, 20), (20, 40)]."""
    cuts = [grain_min_mm] + [s for s in SIEVES if grain_min_mm < s < grain_max_mm] + [grain_max_mm]
    return list(zip(cuts[:-1], cuts[1:]))


def draw_grains(rng, tray_mm: float, crop_frac: float, grain_mm: tuple, mass_p: float):
    """Grain list for one tray, without overlaps and fully inside the centre crop.

    Class mass shares come from a flat Dirichlet, then each class is filled with grains (b
    log-uniform inside the class) until its share of a total mass budget is reached. The budget
    is set so the grains cover about 15 to 35 % of the crop. A grain that does not fit after 200
    tries is skipped; the labels use only the grains actually placed.
    Out: list of (centre_x_mm, centre_y_mm, a_mm, b_mm, angle_deg).
    """
    edges = class_edges_mm(*grain_mm)
    share = rng.dirichlet(np.ones(len(edges)))
    lo_mm, hi_mm = tray_mm * (1 - crop_frac) / 2, tray_mm * (1 + crop_frac) / 2
    crop_area_mm2 = (hi_mm - lo_mm) ** 2
    # mass per unit area of an a x b ellipse with mass a*b^p is (4/pi) * b^(p-1) (a cancels)
    b_mid = np.array([np.sqrt(lo * hi) for lo, hi in edges])
    area_per_mass = 1.0 / (4 / np.pi * b_mid ** (mass_p - 1))
    budget = rng.uniform(0.15, 0.35) * crop_area_mm2 / float((share * area_per_mass).sum())

    wanted = []
    for (lo, hi), s in zip(edges, share):
        mass = 0.0
        while mass < s * budget:
            b = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            a = b * rng.uniform(1.0, 1.4)
            wanted.append((a, b))
            mass += a * b ** mass_p
    wanted.sort(key=lambda ab: -ab[0])                      # place the large grains first

    placed = []                                             # (x, y, radius) of placed grains
    grains = []
    for a, b in wanted:
        r = a / 2 + max(1.0, 0.1 * b)                       # bounding circle plus a visible gap
        if lo_mm + r >= hi_mm - r:
            continue
        for _ in range(200):
            x, y = rng.uniform(lo_mm + r, hi_mm - r, 2)
            if all((x - px) ** 2 + (y - py) ** 2 >= (r + pr) ** 2 for px, py, pr in placed):
                placed.append((x, y, r))
                grains.append((x, y, a, b, rng.uniform(0, 180)))
                break
    return grains


def make_tray(rng, tray_mm: float = 160.0, rho_native: float = 25.0, grain_mm: tuple = (2.0, 40.0),
              mass_p: float = 2.0, crop_frac: float = 0.9):
    """One synthetic tray at the native scale and its exact label.

    Out: rgb uint8 (S, S, 3) with S = tray_mm * rho_native px, true curve (11,) in % passing,
         a dict of label facts (fine share, share in 2 to 6.3 mm, share above 20 mm, n grains),
         and the placed grains' b in mm (n,).
    """
    size_px = int(round(tray_mm * rho_native))
    fine_frac = rng.uniform(0.2, 0.85)
    img = np.full((size_px, size_px, 3), BACKGROUND_RGB, np.float32)
    img += rng.normal(0, 6 + 20 * fine_frac, img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)

    grains = draw_grains(rng, tray_mm, crop_frac, grain_mm, mass_p)
    for x, y, a, b, ang in grains:
        col = tuple(int(v) for v in rng.integers(60, 140, 3))
        cv2.ellipse(img, (int(x * rho_native), int(y * rho_native)),
                    (int(a * rho_native / 2), int(b * rho_native / 2)), ang, 0, 360, col, -1)

    pi = np.zeros(len(SIEVES))
    pi[:I_2MM + 1] = fine_frac / (I_2MM + 1)                # fines: classes up to 2 mm, evenly
    if grains:
        m = np.array([a * b ** mass_p for _, _, a, b, _ in grains])
        m = m / m.sum() * (1 - fine_frac)
        for (_, _, _, b, _), mm in zip(grains, m):
            pi[np.searchsorted(SIEVES, b)] += mm
    else:
        pi[:I_2MM + 1] = 1.0 / (I_2MM + 1)
    curve = clean_curve(100 * np.cumsum(pi))
    facts = {"fine_pct": round(100 * fine_frac, 1),
             "pct_2_to_6.3": round(curve[I_6MM] - curve[I_2MM], 1),
             "pct_above_20": round(100 - curve[I_6MM + 1], 1),
             "n_grains_true": len(grains)}
    return img, curve, facts, np.array([b for _, _, _, b, _ in grains])


def downsample(img_native: np.ndarray, rho_native: float, rho_target: float):
    """Area-average the native tray to rho_target px/mm. Out: (rgb, rho from the actual width)."""
    img = resample_to(img_native, rho_native, rho_target)
    return img, rho_native * img.shape[1] / img_native.shape[1]


def coarse_errors_pp(coarse, curve_pct: np.ndarray, dc: float, grain_max_mm: float) -> dict:
    """Absolute error in percentage points, per sieve, between Path A's coarse curve (grains >= dc)
    and the true coarse curve (share of the mass above dc passing each sieve). Only sieves above
    dc and below grain_max_mm are kept: at or above the largest grain both curves are 100 %.
    Out: {sieve_mm: error_pp}; values are nan when Path A has no grain above dc."""
    pred = coarse.above(dc).G(dc) * 100.0
    true = true_coarse_pct(curve_pct, dc)
    return {float(sv): float(abs(pr - t)) for sv, pr, t in zip(SIEVES, pred, true)
            if dc < sv < grain_max_mm}


def oracle_fines_base(true_pct: np.ndarray, train_curves: np.ndarray) -> np.ndarray:
    """CONTROL stitch base (uses the held-out label below 2 mm, never a method).
    Up to 2 mm: the true curve. Above: the training-fold mean of the coarse shape
    (P - P(2)) / (100 - P(2)), rescaled onto the true remainder. Out: (11,) % passing."""
    P = np.asarray(train_curves, float)
    rest = np.maximum(100.0 - P[:, [I_2MM]], 1e-6)
    shape = ((P - P[:, [I_2MM]]) / rest).mean(axis=0)
    out = true_pct.copy()
    out[I_2MM:] = true_pct[I_2MM] + (100.0 - true_pct[I_2MM]) * shape[I_2MM:]
    return clean_curve(out)


def loso_fixed_base(xs: list, curves: np.ndarray, cfg: Config, base: str, dc_mm=None) -> list:
    """LOSO of the adaptive stitch on a base that is not Path B ("meanB" or "oracleB").
    dc_mm: None uses each soil's own crossover, a number forces it. Out: per-soil EMD list."""
    labels = pd.DataFrame(np.asarray(curves, float), columns=SIEVE_COLUMNS)
    labels.insert(0, ID_COLUMN, [x.sample_id for x in xs])
    by_id = {x.sample_id: (x, c) for x, c in zip(xs, curves)}
    cfg_ad = replace(cfg, blend_mode="adaptive")

    def fit_predict(train, held):
        train_curves = train[SIEVE_COLUMNS].to_numpy(float)
        x, true_pct = by_id[held]
        b = mean_curve(train_curves) if base == "meanB" else oracle_fines_base(true_pct, train_curves)
        return blend(b, x.coarse, x.dc if dc_mm is None else dc_mm, cfg_ad)

    return leave_one_soil_out(labels, fit_predict)["emd"].tolist()


def save_overlay(path, rgb_crop: np.ndarray, regions: list, cfg: Config, title: str):
    """Mask overlay in the show_masks colours (kept / too small / low solidity / edge cut)."""
    H, W = rgb_crop.shape[:2]
    labels = [classify(r, H, W, cfg)[0] for r in regions]
    out = draw(rgb_crop, regions, labels)
    counts = {lab: labels.count(lab) for lab in COLOURS_RGB}
    sc = min(1.0, MAX_OUT_SIDE_PX / max(H, W))
    if sc < 1:
        out = cv2.resize(out, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    add_legend(out, counts, title)
    cv2.imwrite(str(path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


def plot_summary(summary: pd.DataFrame):
    """Two panels over px/mm: Path A error at one sieve per crossover (median over soils), and
    the paired LOSO EMD difference adaptive d_c minus fixed 6.3 mm on each stitch base."""
    blue, orange, aqua = "#257de0", "#f47230", "#00b894"
    s = summary.sort_values("rho_px_per_mm")
    x = s["rho_px_per_mm"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for col, mark, colour, label in (("err_s20_dc6.3_median_pp", "-o", blue, "20 mm sieve, share of mass > 6.3 mm"),
                                     ("err_s6.3_dc2_median_pp", "-s", orange, "6.3 mm sieve, share of mass > 2 mm")):
        if col in s:                                        # absent for some --scales / --grain-max-mm
            ax1.plot(x, s[col], mark, color=colour, lw=2, ms=6, label=label)
    ax1.set_ylabel("Path A error, median over soils [pp]")
    ax1.set_title("Path A coarse curve vs truth (one sieve per line)", fontsize=10)
    ax2.axhline(0, color="#666666", lw=1)
    ax2.plot(x, s["delta_pathB"], "-o", color=blue, lw=2, ms=6, label="on Path B")
    ax2.plot(x, s["delta_meanB"], "-s", color=orange, lw=2, ms=6, label="on mean curve")
    ax2.plot(x, s["delta_oracleB"], "-^", color=aqua, lw=2, ms=6, label="on oracle fines (control)")
    ax2.set_ylabel("EMD, adaptive d_c minus fixed 6.3 mm")
    ax2.set_title("Paired LOSO difference\n(adaptive stitch; below 0: d_c = 2 mm helps)", fontsize=10)
    for ax in (ax1, ax2):
        ax.set_xlabel("scale [px/mm]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle("Synthetic resolution simulation (classical segmenter)", fontsize=11)
    fig.tight_layout()
    return fig


def git_commit() -> tuple:
    """(short HEAD hash, True when scripts/, soilgsd/ or configs/ differ from HEAD or have
    untracked files)."""
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        root = config.REPO_ROOT
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--",
                                str(root / "scripts"), str(root / "soilgsd"), str(root / "configs")],
                               capture_output=True, text=True, check=True).stdout.strip() != ""
        return head, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def paired_delta(a: list, b: list) -> tuple:
    """Per-soil EMD lists a and b (same soil order). Out: (mean of a - b, n soils better, n worse)."""
    d = np.asarray(a) - np.asarray(b)
    return round(float(d.mean()), 3), int((d < -1e-6).sum()), int((d > 1e-6).sum())


def main():
    parser = add_tool_args(config.base_parser(__doc__))
    parser.add_argument("--n-soils", type=int, default=24, help="number of synthetic soils")
    parser.add_argument("--scales", type=float, nargs="+", default=[25.0, 19.5, 13.9, 4.6],
                        help="target scales in px/mm (the first is not required to be native)")
    parser.add_argument("--rho-native", type=float, default=25.0, help="render scale [px/mm]")
    parser.add_argument("--tray-mm", type=float, default=160.0, help="tray side [mm]")
    parser.add_argument("--grain-min-mm", type=float, default=2.0)
    parser.add_argument("--grain-max-mm", type=float, default=40.0)
    parser.add_argument("--overlays", type=int, default=0, help="save mask overlays for the first N soils")
    parser.add_argument("--seed", type=int, default=None, help="default: seed from the config")
    args = parser.parse_args()
    if args.segmenter != "classical":
        raise SystemExit("resolution_sim runs the classical segmenter only (SAM on 4000 px trays is a Colab job)")
    size_px = int(round(args.tray_mm * args.rho_native))
    widths = [int(round(size_px * r / args.rho_native)) for r in args.scales]
    if len(set(widths)) != len(widths):
        raise SystemExit(f"--scales must give distinct image sizes, got {args.scales}")

    ycfg = config.load_config(args.config)
    cfg = replace(Config.from_yaml(ycfg), max_side=0)       # trays are 4000 px; never rescale them
    seed = ycfg.get("seed", 0) if args.seed is None else args.seed
    config.set_seed(seed)
    rng = np.random.default_rng(seed)
    seg, emb = make_tools(ycfg, args)
    n_soils = min(args.n_soils, args.limit) if args.limit else args.n_soils
    run_dir = config.new_run_dir(ycfg, "resolution_sim")

    extracted = {rho: [] for rho in args.scales}
    actual_rho = {}                                        # target scale -> scale of the resized tray
    per_soil, curves = [], []
    for i in range(n_soils):
        sid = f"syn{i:02d}"
        native, curve, facts, b_true_mm = make_tray(rng, args.tray_mm, args.rho_native,
                                                    (args.grain_min_mm, args.grain_max_mm), cfg.mass_p,
                                                    cfg.tray_crop_frac)
        curves.append(curve)
        for rho_target in args.scales:
            t0 = time.time()
            img, rho = downsample(native, args.rho_native, rho_target)
            regions = seg.segment(center_crop(img, cfg.tray_crop_frac))
            x = extract(Sample(sid, [(img, rho)]), _CachedRegions(regions), cfg, emb)
            extracted[rho_target].append(x)
            actual_rho[rho_target] = rho
            c = x.coarse.above(x.dc)
            # true grains wide enough to pass tau (Feret width is about b * rho + 1 px)
            n_resolvable = int((b_true_mm * rho + 1 >= cfg.tau_px).sum())
            e63 = coarse_errors_pp(x.coarse, curve, FIXED_DC_MM, args.grain_max_mm)
            e2 = (coarse_errors_pp(x.coarse, curve, 2.0, args.grain_max_mm)
                  if adaptive_crossover(rho, cfg.tau_px) <= 2.0 else {})
            row = {ID_COLUMN: sid, "scale_target": rho_target, "rho_px_per_mm": round(rho, 2),
                   "dc_mm": x.dc, **facts,
                   "n_regions": len(regions), "n_kept": int(x.coarse.b_mm.size),
                   "n_true_resolvable": n_resolvable,
                   "net_extra_kept": int(x.coarse.b_mm.size) - n_resolvable,   # splits minus misses
                   "n_above_dc": int(c.b_mm.size), "n_eff": round(c.n_eff, 1),
                   "w": round(c.n_eff / (c.n_eff + cfg.kappa), 3) if c.n_eff > 0 else 0.0,
                   **{f"err_s{sv:g}_dc6.3_pp": round(v, 2) for sv, v in e63.items()},
                   **{f"err_s{sv:g}_dc2_pp": round(v, 2) for sv, v in e2.items()},
                   "seconds": round(time.time() - t0, 1)}
            per_soil.append(row)
            print(f"{sid} rho {rho:5.2f}: {row['n_kept']:4d} kept ({row['net_extra_kept']:+d} net vs resolvable), "
                  f"d_c {x.dc:g}, err 20|6.3 {row.get('err_s20_dc6.3_pp')}, "
                  f"err 6.3|2 {row.get('err_s6.3_dc2_pp')}, {row['seconds']} s")
            if i < args.overlays:
                save_overlay(run_dir / f"overlay_{sid}_rho{rho_target:g}.png", center_crop(img, cfg.tray_crop_frac),
                             regions, cfg, f"{sid} at {rho:.1f} px/mm")
    curves = np.stack(curves)
    per_soil = pd.DataFrame(per_soil)
    per_soil.to_csv(run_dir / "per_soil.csv", index=False)
    err_cols = [k for k in per_soil.columns if k.startswith("err_s")]

    rows, loso_rows = [], []
    for rho_target in args.scales:
        xs = extracted[rho_target]
        rho = round(actual_rho[rho_target], 2)
        ps = per_soil[per_soil["scale_target"] == rho_target]
        res_adapt = loso(xs, curves, cfg)
        res_fixed = loso([replace(x, dc=FIXED_DC_MM) for x in xs], curves, cfg)
        per_base = {"pathB": (res_adapt["adaptive"][1], res_fixed["adaptive"][1]),
                    "hardB": (res_adapt["hard"][1], res_fixed["hard"][1])}
        for base in ("meanB", "oracleB"):
            per_base[base] = (loso_fixed_base(xs, curves, cfg, base),
                              loso_fixed_base(xs, curves, cfg, base, FIXED_DC_MM))
        row = {"scale_target": rho_target, "rho_px_per_mm": rho, "dc_adaptive_mm": xs[0].dc}
        for col in err_cols:
            v = ps[col] if col in ps else pd.Series(dtype=float)
            row[col.replace("_pp", "_median_pp")] = round(v.median(), 2) if v.notna().any() else float("nan")
            row[col.replace("_pp", "_mean_pp")] = round(v.mean(), 2) if v.notna().any() else float("nan")
            row[col.replace("_pp", "_n_soils")] = int(v.notna().sum())
        row["soils_net_extra_kept"] = int((ps["net_extra_kept"] > 0).sum())
        row["mean_n_eff"] = round(ps["n_eff"].mean(), 1)
        for base, (a, f) in per_base.items():
            d, better, worse = paired_delta(a, f)
            row.update({f"delta_{base}": d, f"better_{base}": better, f"worse_{base}": worse})
        row.update({"emd_mean_curve": round(res_adapt["mean_curve"][0], 2),
                    "emd_pure_b": round(res_adapt["pure_b"][0], 2)})
        for base, (a, f) in per_base.items():
            row[f"emd_{base}_dc6.3"] = round(float(np.mean(f)), 2)
            row[f"emd_{base}_dcadapt"] = round(float(np.mean(a)), 2)
        rows.append(row)
        for j, x in enumerate(xs):
            loso_rows.append({ID_COLUMN: x.sample_id, "rho_px_per_mm": rho,
                              "emd_mean_curve": round(res_adapt["mean_curve"][1][j], 3),
                              "emd_pure_b": round(res_adapt["pure_b"][1][j], 3),
                              **{f"emd_{b}_{tag}": round(v[k][j], 3) for b, v in per_base.items()
                                 for k, tag in ((0, "dcadapt"), (1, "dc6.3"))}})
    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / "summary.csv", index=False)
    pd.DataFrame(loso_rows).to_csv(run_dir / "loso_per_soil.csv", index=False)
    plot_summary(summary).savefig(run_dir / "resolution_sim.png", dpi=150)

    show = ["rho_px_per_mm", "dc_adaptive_mm", "err_s20_dc6.3_median_pp", "err_s6.3_dc2_median_pp",
            "err_s20_dc2_median_pp", "soils_net_extra_kept", "mean_n_eff"]
    print("\n" + summary[[k for k in show if k in summary]].to_string(index=False))
    show = ["rho_px_per_mm"] + [f"{k}_{b}" for b in ("pathB", "hardB", "meanB", "oracleB")
                                for k in ("delta", "better", "worse")]
    print("\n" + summary[show].to_string(index=False))
    show = ["rho_px_per_mm", "emd_mean_curve", "emd_pure_b"] + [f"emd_{b}_{t}" for b in ("pathB", "hardB", "meanB", "oracleB")
                                                              for t in ("dc6.3", "dcadapt")]
    print("\n" + summary[show].to_string(index=False))
    print(f"\nper soil: share of mass in 2 to 6.3 mm = {per_soil.drop_duplicates(ID_COLUMN)['pct_2_to_6.3'].mean():.1f} % "
          f"(mean), {n_soils} soils, seed {seed}")
    cfg_text = (f"synthetic segmenter={seg.name} embedder={getattr(emb, 'name', type(emb).__name__)} "
                f"n_soils={n_soils} seed={seed} tray_mm={args.tray_mm:g} "
                f"grains={args.grain_min_mm:g}-{args.grain_max_mm:g}mm rho_native={args.rho_native:g} "
                f"tau_px={cfg.tau_px:g} mass_p={cfg.mass_p:g} kappa={cfg.kappa:g} "
                f"min_solidity={cfg.min_solidity:g} n_pca={cfg.n_pca} ridge_alpha={cfg.ridge_alpha:g} "
                f"crop={cfg.tray_crop_frac:g} target_rho={cfg.target_rho:g} feat_min_mm={cfg.feat_min_mm:g} "
                f"config={Path(ycfg['_config_path']).name}")
    date, (commit, dirty) = datetime.date.today().isoformat(), git_commit()
    note = "synthetic resolution simulation" + ("; code uncommitted at run time" if dirty else "")
    print("\nresults.csv rows (pathB = adaptive stitch, hardB = hard stitch, both on Path B; "
          "meanB / oracleB = adaptive stitch on the fold mean curve / oracle-fines control):")
    for r in rows:
        for col in [k for k in r if k.startswith("emd_")]:
            name = f"{run_dir.name}/rho{r['scale_target']:g}_{col[4:]}"
            print(f'{name},{date},"{cfg_text}",{commit},{r[col]},,{note}')
    print(f"\nwritten to {run_dir}")


if __name__ == "__main__":
    main()
