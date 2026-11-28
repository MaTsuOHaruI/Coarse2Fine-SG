"""Path B candidate heads beyond the ridge/PCA head (NEW file: path_b.py is untouched).

PathBExt        same z -> curve map as PathB (Eq. 4: pi = softmax(mu + V z)), with two options
                  loss="ridge": ridge on the PCA scores, as PathB (plus optional per-row weights)
                  loss="emd"  : linear head fitted directly on a smoothed Eq. (1), started from the ridge fit
                and optional photo-level training: rows are photos, `groups` names the soil of every row,
                each soil carries a total weight of 1, and predict() averages z over the rows it is given.
KNNMedianCurve  pointwise median of the k nearest training curves in standardised feature space. A pointwise
                median of monotone curves is monotone, ends at 100 and is the L1-optimal summary of a
                neighbourhood, which suits an L1-type metric on heavy-tailed soils.
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .config import LOG_WIDTHS
from .metrics import clean_curve
from .path_b import comp_to_clr, curve_to_comp, softmax   # curve_to_comp reads path_b.EPS at call time


def emd_weights(form="left"):
    """Weights of the 11 curve points in Eq. (1) for the three integration forms of metrics.emd."""
    w = np.zeros(len(LOG_WIDTHS) + 1)
    if form == "left":
        w[:-1] = LOG_WIDTHS
    elif form == "right":
        w[1:] = LOG_WIDTHS
    elif form == "trapezoid":
        w[:-1] += LOG_WIDTHS / 2
        w[1:] += LOG_WIDTHS / 2
    else:
        raise ValueError(f"unknown form: {form}")
    return w


def _softmax_rows(L):
    e = np.exp(L - L.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


class PathBExt:
    def __init__(self, r=3, alpha=10.0, loss="ridge", lam=1.0, form="left", smooth=1.0, max_iter=300):
        """alpha: ridge penalty (also the start of the emd fit). lam: weight of sum(W**2) next to the
        mean EMD (EMD units, about 40 here). smooth: width [% points] of the smoothed |.| in the loss."""
        self.r, self.alpha, self.loss, self.lam = r, alpha, loss, lam
        self.form, self.smooth, self.max_iter = form, smooth, max_iter

    def fit(self, X, curves, groups=None):
        X, curves = np.asarray(X, float), np.asarray(curves, float)
        groups = np.arange(len(X)) if groups is None else np.asarray(groups)
        _, first, inv, cnt = np.unique(groups, return_index=True, return_inverse=True, return_counts=True)
        sw = 1.0 / cnt[inv]                                   # every soil sums to 1, whatever its photo count
        to_clr = lambda C: np.stack([comp_to_clr(curve_to_comp(c)) for c in C])
        self.pca = PCA(n_components=min(self.r, len(first) - 1)).fit(to_clr(curves[first]))   # basis from soils
        Z = self.pca.transform(to_clr(curves))
        self.sc = StandardScaler().fit(X)
        Xs = self.sc.transform(X)
        head = Ridge(alpha=self.alpha).fit(Xs, Z, sample_weight=sw)
        self.W, self.b = head.coef_.T.copy(), head.intercept_.copy()
        if self.loss == "emd":
            self._fit_emd(Xs, curves, sw)
        elif self.loss != "ridge":
            raise ValueError(f"unknown loss: {self.loss}")
        return self

    def _objective(self, theta, Xs, T, sw):
        d, r = self.W.shape
        mu, V = self.pca.mean_, self.pca.components_
        wt, s = emd_weights(self.form), self.smooth
        W, b = theta[:d * r].reshape(d, r), theta[d * r:]
        Pi = _softmax_rows(mu + (Xs @ W + b) @ V)
        diff = 100.0 * np.cumsum(Pi, 1) - T
        root = np.sqrt(diff ** 2 + s ** 2)
        loss = (sw * ((root - s) * wt).sum(1)).sum() + self.lam * (W ** 2).sum()
        GP = sw[:, None] * wt * diff / root                    # d loss / d P
        GPi = 100.0 * np.cumsum(GP[:, ::-1], 1)[:, ::-1]       # through P = 100 cumsum(pi)
        GL = Pi * (GPi - (Pi * GPi).sum(1, keepdims=True))     # through the softmax
        GZ = GL @ V.T
        return loss, np.r_[(Xs.T @ GZ + 2 * self.lam * W).ravel(), GZ.sum(0)]

    def _fit_emd(self, Xs, T, sw):
        sw = sw / sw.sum()
        res = minimize(self._objective, np.r_[self.W.ravel(), self.b], args=(Xs, T, sw), jac=True,
                       method="L-BFGS-B", options={"maxiter": self.max_iter})
        d, r = self.W.shape
        self.W, self.b = res.x[:d * r].reshape(d, r), res.x[d * r:]

    def predict(self, x):
        """x: one feature row, or all photo rows of one soil (z is averaged over the rows)."""
        z = (self.sc.transform(np.atleast_2d(x)) @ self.W + self.b).mean(0)
        pi = softmax(self.pca.mean_ + z @ self.pca.components_)
        return clean_curve(100.0 * np.cumsum(pi))


class KNNMedianCurve:
    """k nearest soils. With `groups` (soil of every row) a soil is as near as its nearest row, so several rows of
    one soil (e.g. one per phone) never fill the neighbourhood with the same curve."""
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, curves, groups=None):
        self.sc = StandardScaler().fit(X)
        self.Xs, self.C = self.sc.transform(X), np.asarray(curves, float)
        self.g = None if groups is None else np.asarray(groups)
        return self

    def predict(self, x):
        xs = self.sc.transform(np.atleast_2d(x)).mean(0)
        d = np.linalg.norm(self.Xs - xs, axis=1)
        if self.g is None:
            idx = np.argsort(d)[:self.k]
        else:
            _, first, inv = np.unique(self.g, return_index=True, return_inverse=True)
            dmin = np.full(len(first), np.inf)
            np.minimum.at(dmin, inv, d)
            idx = first[np.argsort(dmin)[:self.k]]
        return clean_curve(np.median(self.C[idx], axis=0))
