"""Acceptance tests on the real Kaggle files. Skipped when the data folder is absent."""

import numpy as np
import pytest

from soilgsd import baselines, data
from soilgsd.config import ID_COLUMN, SIEVE_COLUMNS
from soilgsd.cv import leave_one_soil_out
from soilgsd.metrics import mean_emd
from soilgsd.submit import curve_problems, write_submission


def test_labels_are_24_valid_curves(kaggle_dir):
    labels = data.load_labels(kaggle_dir)
    assert len(labels) == 24
    for curve in labels[SIEVE_COLUMNS].to_numpy(dtype=float):
        assert curve_problems(curve) == []


def test_constant_baseline_reproduces_102(kaggle_dir):
    labels = data.load_labels(kaggle_dir)
    P_true = labels[SIEVE_COLUMNS].to_numpy(dtype=float)
    constant = np.tile(baselines.constant_curve(), (len(labels), 1))
    assert mean_emd(P_true, constant) == pytest.approx(102.0, abs=0.05)


def test_mean_curve_loso_reproduces_89_9(kaggle_dir):
    labels = data.load_labels(kaggle_dir)

    def predictor(train_labels, held_out_id):
        assert held_out_id not in set(train_labels[ID_COLUMN])  # no leakage
        return baselines.mean_curve(train_labels[SIEVE_COLUMNS].to_numpy(dtype=float))

    result = leave_one_soil_out(labels, predictor)
    assert len(result) == 24
    assert result["emd"].mean() == pytest.approx(89.9, abs=0.05)


def test_train_photos_match_labels_and_scale(kaggle_dir):
    photos = data.list_photos(kaggle_dir, "train")
    assert len(photos) == 127
    assert set(photos[ID_COLUMN]) == set(data.load_labels(kaggle_dir)[ID_COLUMN])
    downscaled = photos[photos["phone"] != "Motorola Edge 60 Fusion"]
    assert downscaled["rho_px_per_mm"].between(4.4, 4.8).all()
    # The three H374 photos are full resolution.
    h374 = photos[photos[ID_COLUMN] == "H374"]
    assert len(h374) == 3
    assert h374["rho_px_per_mm"].between(12.0, 13.0).all()


def test_test_photos_map_to_the_ten_submission_ids(kaggle_dir):
    photos = data.list_photos(kaggle_dir, "test")
    assert len(photos) == 35
    assert set(photos[ID_COLUMN]) == set(data.load_test_ids(kaggle_dir))
    assert photos[ID_COLUMN].nunique() == 10
    iphone14 = photos[photos["phone"] == "iPhone 14"]["rho_px_per_mm"]
    iphone16 = photos[photos["phone"] == "iPhone 16"]["rho_px_per_mm"]
    assert iphone14.between(13.8, 14.0).all()
    assert iphone16.between(19.4, 19.6).all()


def test_mean_curve_submission_is_valid(kaggle_dir, tmp_path):
    labels = data.load_labels(kaggle_dir)
    test_ids = data.load_test_ids(kaggle_dir)
    curve = baselines.mean_curve(labels[SIEVE_COLUMNS].to_numpy(dtype=float))
    df = write_submission({i: curve for i in test_ids}, test_ids, tmp_path / "submission.csv")
    assert df[ID_COLUMN].tolist() == test_ids
