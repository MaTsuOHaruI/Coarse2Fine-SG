import numpy as np
from soilgsd.config import SIEVES, Config
from soilgsd.crossover import adaptive_crossover
from soilgsd.metrics import emd
from soilgsd.synth import make_dataset
from soilgsd.segment import ClassicalSegmenter
from soilgsd.pipeline import extract, SoilGSD


def test_crossover_matches_proposal():
    assert adaptive_crossover(4.6) == 6.3      # train photos
    assert adaptive_crossover(13.9) == 2       # iPhone 14
    assert adaptive_crossover(19.5) == 2       # iPhone 16


def test_emd_zero_and_positive():
    c = np.linspace(0, 100, 11)
    assert emd(c, c) == 0 and emd(c, c * 0.9) > 0


def test_end_to_end_monotone():
    S, C = make_dataset(8, seed=1)
    cfg = Config(n_pca=2)
    X = [extract(s, ClassicalSegmenter(), cfg) for s in S]
    m = SoilGSD(cfg).fit(X[:-1], C[:-1])
    for mode in ("adaptive", "pure_b", "hard"):
        p = m.predict(X[-1], mode)["final"]
        assert p.shape == (11,) and np.all(np.diff(p) >= -1e-9) and p[-1] == 100 and p.min() >= 0
