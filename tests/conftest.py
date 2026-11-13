"""Shared test fixtures. Tests that need the Kaggle files are skipped when they are absent."""

import pytest

from soilgsd import config


@pytest.fixture(scope="session")
def kaggle_dir():
    path = config.kaggle_dir(config.load_config())
    if not (path / "Training_labels_updated.csv").exists():
        pytest.skip(f"Kaggle data not found in {path}")
    return path
