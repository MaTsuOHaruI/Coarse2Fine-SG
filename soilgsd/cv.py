import numpy as np
from .pipeline import SoilGSD
from .metrics import emd
from .config import Config


def loso(extracted, curves, cfg=Config()):
    """Leave-one-soil-out CV. Returns mean EMD per method (+ per-sample)."""
    n = len(extracted)
    names = ["mean_curve", "pure_b", "adaptive", "hard"]
    res = {k: [] for k in names}
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        m = SoilGSD(cfg).fit([extracted[j] for j in tr], curves[tr])
        res["mean_curve"].append(emd(curves[i], m.b.mean_curve()))
        res["pure_b"].append(emd(curves[i], m.predict(extracted[i], "pure_b")["final"]))
        res["adaptive"].append(emd(curves[i], m.predict(extracted[i], "adaptive")["final"]))
        res["hard"].append(emd(curves[i], m.predict(extracted[i], "hard")["final"]))
    return {k: (float(np.mean(v)), v) for k, v in res.items()}
