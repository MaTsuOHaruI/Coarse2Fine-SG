"""Path B: PCA-composition head. Every z gives a monotone curve; z=0 is the mean curve."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from .metrics import clean_curve

EPS = 1e-3


def curve_to_comp(P):
    P = np.asarray(P, float) / 100.0
    pi = np.diff(np.r_[0.0, P])
    return np.clip(pi, 0, None) + EPS


def comp_to_clr(pi):
    pi = pi / pi.sum()
    l = np.log(pi)
    return l - l.mean()


def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()


class PathB:
    def __init__(self, r=3, alpha=10.0):
        self.r, self.alpha = r, alpha

    def fit(self, X, curves):
        clr = np.stack([comp_to_clr(curve_to_comp(c)) for c in curves])
        self.pca = PCA(n_components=min(self.r, len(curves) - 1)).fit(clr)   # mu, V
        Z = self.pca.transform(clr)
        self.sc = StandardScaler().fit(X)
        self.head = Ridge(alpha=self.alpha).fit(self.sc.transform(X), Z)
        return self

    def predict(self, x):
        z = np.ravel(self.head.predict(self.sc.transform(np.atleast_2d(x))))
        pi = softmax(self.pca.mean_ + z @ self.pca.components_)            # Eq. (4)
        return clean_curve(100.0 * np.cumsum(pi))

    def mean_curve(self):
        """Curve at z = 0: the centre of the training curves in clr space (a geometric mean).
        Not the mean-curve baseline: that is baselines.mean_curve (arithmetic mean, LOSO 89.9 vs 92.1 for this)."""
        return clean_curve(100.0 * np.cumsum(softmax(self.pca.mean_)))
