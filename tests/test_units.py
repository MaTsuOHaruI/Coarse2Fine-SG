"""Tests that need no data: metric, scale, curve basis, stitch, grains, cache, submission checks."""

import cv2
import numpy as np
import pandas as pd
import pytest

from soilgsd.config import Config, ID_COLUMN, SIEVE_COLUMNS, SIEVES
from soilgsd.crossover import adaptive_crossover, soil_crossover, true_scale
from soilgsd.metrics import emd
from soilgsd.path_a import CoarseResult, grain_axes
from soilgsd.path_b import EPS, PathB, comp_to_clr, curve_to_comp, softmax
from soilgsd.path_c import blend
from soilgsd.segment import Region, load_regions, save_regions
from soilgsd.submit import check_submission, curve_problems, write_submission


def random_curve(rng):
    """A random valid curve: 11 percentages, non-decreasing, ending at 100."""
    return 100.0 * np.cumsum(rng.dirichlet(np.ones(11)))


def grains(b_mm, mass=None):
    """CoarseResult with the given minimum-Feret sizes (equal masses unless given)."""
    b = np.asarray(b_mm, float)
    m = np.ones_like(b) if mass is None else np.asarray(mass, float)
    return CoarseResult(b, m, np.ones_like(b), 1e4)


# ---- metric ----

def test_emd_is_zero_for_identical_curves():
    curve = random_curve(np.random.default_rng(0))
    assert emd(curve, curve) == 0.0


def test_emd_known_value_and_range():
    zeros = np.zeros(11)
    hundred = np.full(11, 100.0)
    # 100 points of difference over 5 decades of log10(size) = 500, the stated maximum.
    assert emd(zeros, hundred) == pytest.approx(500.0)
    assert emd(zeros, hundred, form="right") == pytest.approx(500.0)
    assert emd(zeros, hundred, form="trapezoid") == pytest.approx(500.0)


def test_emd_is_symmetric():
    rng = np.random.default_rng(1)
    a, b = random_curve(rng), random_curve(rng)
    assert emd(a, b) == pytest.approx(emd(b, a))


# ---- scale ----

def test_true_scale_of_downscaled_and_rotated_photo():
    # Motorola Edge: 11.492 px/mm at 4000 x 1800, released at 1600 x 720.
    assert true_scale(11.492, (1600, 720), (4000, 1800)) == pytest.approx(4.597, abs=1e-3)
    assert true_scale(11.492, (720, 1600), (4000, 1800)) == pytest.approx(4.597, abs=1e-3)


def test_crossover_at_train_and_test_scales():
    assert adaptive_crossover(4.6) == 6.3
    assert adaptive_crossover(13.9) == 2
    assert adaptive_crossover(19.5) == 2
    assert adaptive_crossover(0.01) == float("inf")  # nothing resolvable: blend falls back to Path B


def test_soil_crossover_takes_the_largest():
    assert soil_crossover([4.6, 12.5]) == 6.3


# ---- curve basis (Path B) ----

def test_composition_round_trip():
    curve = random_curve(np.random.default_rng(2))
    pi = curve_to_comp(curve) - EPS
    assert pi.sum() == pytest.approx(1.0)
    assert np.allclose(100.0 * np.cumsum(pi), curve)


def test_head_predicts_valid_curves():
    rng = np.random.default_rng(3)
    P = np.array([random_curve(rng) for _ in range(20)])
    X = rng.normal(size=(20, 5))
    head = PathB(r=2, alpha=1.0).fit(X, P)
    for x in [np.zeros(5), rng.normal(size=5) * 50]:          # also far outside the training range
        assert curve_problems(head.predict(x)) == []
    assert curve_problems(head.mean_curve()) == []


def test_full_rank_basis_reconstructs_smoothed_curves():
    rng = np.random.default_rng(4)
    P = np.array([random_curve(rng) for _ in range(20)])
    head = PathB(r=10).fit(rng.normal(size=(20, 3)), P)      # clr space has 10 free dimensions
    clr = np.stack([comp_to_clr(curve_to_comp(c)) for c in P])
    back = np.cumsum(softmax_rows(head.pca.mean_ + head.pca.transform(clr) @ head.pca.components_), axis=1)
    smoothed = np.cumsum(np.stack([curve_to_comp(c) / curve_to_comp(c).sum() for c in P]), axis=1)
    assert np.allclose(back, smoothed, atol=1e-6)


def softmax_rows(Z):
    return np.stack([softmax(z) for z in Z])


# ---- stitch (Path C) ----

def test_blend_output_is_valid_for_random_inputs():
    rng = np.random.default_rng(5)
    for _ in range(300):
        pB = random_curve(rng)
        dc = float(rng.choice([2, 6.3, 20]))
        coarse = grains(rng.uniform(0.5, 150, rng.integers(0, 30)), None)
        coarse.mass = rng.lognormal(0, 1, coarse.b_mm.size)
        mode = str(rng.choice(["adaptive", "hard", "pure_b"]))
        out = blend(pB, coarse, dc, Config(kappa=10.0, blend_mode=mode))
        i_c = int(np.where(SIEVES == dc)[0][0])
        assert curve_problems(out) == []
        assert np.allclose(out[:i_c], pB[:i_c])              # untouched below the crossover


def test_pure_b_returns_path_b():
    pB = random_curve(np.random.default_rng(6))
    out = blend(pB, grains([10, 30, 80]), 6.3, Config(blend_mode="pure_b"))
    assert np.allclose(out, pB)


def test_no_grains_above_crossover_returns_path_b():
    pB = random_curve(np.random.default_rng(7))
    assert np.allclose(blend(pB, grains([]), 6.3, Config()), pB)
    assert np.allclose(blend(pB, grains([3.0, 5.5]), 6.3, Config()), pB)   # all below d_c


def test_hard_stitch_uses_measured_shape():
    pB = np.array([1, 2, 5, 10, 20, 40, 60, 80, 90, 95, 100], dtype=float)
    # two equal-mass grains: one between 6.3 and 20 mm, one between 20 and 63 mm
    out = blend(pB, grains([10.0, 40.0]), 6.3, Config(blend_mode="hard"))
    assert np.allclose(out[7:], [80, 90, 100, 100])


# ---- grains (Path A) ----

def test_coarse_curve_pools_only_grains_above_crossover():
    # 5.6 mm is resolved at 4.6 px/mm (tau/rho = 5.4 mm) but is below d_c = 6.3 mm.
    G = grains([5.6, 8.0, 15.0]).G(6.3)
    assert np.isnan(G[:7]).all()
    assert G[7] == pytest.approx(0.0)
    assert G[8] == pytest.approx(1.0) and G[10] == pytest.approx(1.0)


@pytest.mark.parametrize("b_px", [25, 50, 100])
@pytest.mark.parametrize("ratio", [1.0, 1.5, 2.5])
@pytest.mark.parametrize("angle", [0, 30, 67])
def test_grain_axes_on_ellipses(b_px, ratio, angle):
    ra, rb = int(round(b_px * ratio / 2)), int(round(b_px / 2))
    a_px, b_px = 2 * ra + 1, 2 * rb + 1          # cv2.ellipse draws 2r + 1 pixels across
    img = np.zeros((int(a_px * 1.6) + 40,) * 2, np.uint8)
    c = img.shape[0] // 2
    cv2.ellipse(img, (c, c), (ra, rb), angle, 0, 360, 1, -1)
    a, b, _ = grain_axes(img.astype(bool))
    assert abs(b - b_px) / b_px < 0.05
    assert abs(a - a_px) / a_px < 0.05


# ---- mask cache ----

def test_regions_survive_the_cache(tmp_path):
    rng = np.random.default_rng(10)
    regions = [Region(rng.random((h, w)) > 0.5, (y, x, y + h, x + w))
               for h, w, y, x in [(3, 5, 0, 0), (17, 9, 40, 12), (1, 1, 7, 7)]]
    save_regions(tmp_path / "m.npz", regions)
    back = load_regions(tmp_path / "m.npz")
    assert len(back) == 3
    for r, s in zip(regions, back):
        assert np.array_equal(r.mask, s.mask) and r.bbox == s.bbox
    save_regions(tmp_path / "empty.npz", [])
    assert load_regions(tmp_path / "empty.npz") == []


# ---- submission checks ----

def test_curve_problems_accepts_and_rejects():
    good = random_curve(np.random.default_rng(8))
    assert curve_problems(good) == []
    decreasing = good.copy()
    decreasing[3] = decreasing[2] - 1
    assert "not non-decreasing" in curve_problems(decreasing)
    assert curve_problems(np.append(good[:-1], 99.0)) != []
    assert curve_problems(np.append(-5.0, good[1:])) != []
    assert curve_problems(good[:10]) != []


def test_write_submission_checks_ids_and_curves(tmp_path):
    ids = ["A", "B"]
    good = random_curve(np.random.default_rng(9))
    df = write_submission({"A": good, "B": good}, ids, tmp_path / "s.csv")
    assert list(df.columns) == [ID_COLUMN] + SIEVE_COLUMNS
    assert pd.read_csv(tmp_path / "s.csv")[ID_COLUMN].tolist() == ids

    with pytest.raises(ValueError):
        write_submission({"A": good}, ids, tmp_path / "missing.csv")
    with pytest.raises(ValueError):
        write_submission({"A": good, "B": good[::-1]}, ids, tmp_path / "bad.csv")
    with pytest.raises(ValueError):
        check_submission(df, ["A", "B", "C"])


def test_sieve_constants_agree():
    assert np.allclose([float(c) for c in SIEVE_COLUMNS], SIEVES)
