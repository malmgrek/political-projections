"""Data analysis functionality"""

from functools import reduce

import numpy as np
from scipy.cluster import hierarchy
from scipy.stats import spearmanr
from sklearn.experimental import enable_iterative_imputer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.impute import IterativeImputer


class AffineScaler():

    def __init__(self, w=1, bias=0):
        self.w = w
        self.bias = bias

    def transform(self, X):
        return self.w * X + self.bias

    def inverse_transform(self, Y):
        return (Y - self.bias) / self.w

    def to_dict(self):
        return {"w": self.w.tolist(), "bias": self.bias.tolist()}

    @classmethod
    def from_dict(cls, x: dict):
        w = np.array(x["w"])
        bias = np.array(x["bias"])
        return AffineScaler(w=w, bias=bias)


def IntervalScaler(a, b):
    """Affine transform from a given interval to [-1, 1]

    forward: (x - 0.5 * (a + b)) / (b - a)
    inverse: y * (b - a) + 0.5 * (b - a)

    """
    a = np.array(a)
    b = np.array(b)
    w = 2 / (b - a)
    bias = (a + b) / (a - b)
    return AffineScaler(w=w, bias=bias)


def UnitScaler(X):
    u = X.mean(axis=0)
    s = X.std(axis=0)
    w = 1 / s
    bias = -u / s
    return AffineScaler(w=w, bias=bias)


def fit_kde(Y: np.ndarray):
    """Fit Scikit-Learn's KernelDensity object

    Grid searches for a satisfactory `bandwidth` parameter.

    """
    params = {'bandwidth': np.logspace(-1, 1, 20)}
    grid = GridSearchCV(KernelDensity(), params)
    grid.fit(Y)
    kde = grid.best_estimator_
    return kde


def score_density_grid(kde: KernelDensity, Y: np.ndarray, num=100):

    def lim(y, i):
        mi = y[:, i].min()
        ma = y[:, i].max()
        return [mi - 0.2 * abs(mi), ma + 0.2 * abs(ma)]

    xlim = lim(Y, 0)
    ylim = lim(Y, 1)
    (x, y) = np.meshgrid(
        np.linspace(*xlim, num=num),
        np.linspace(*ylim, num=num)
    )
    density = np.exp(
        kde.score_samples(np.c_[x.ravel(), y.ravel()])
    ).reshape(num, num)

    return (x, y, density, xlim, ylim)


def impute_missing(X, *args, **kwargs):
    """Impute missing values using Sklearn iterative imputer

    """
    imputer = IterativeImputer(*args, **kwargs).fit(X)
    return imputer.transform(X)


def order_features(X, features, corrcov):
    """Order features that behave similarly

    """
    C = np.cov(X.T) if corrcov == "cov" else spearmanr(X).correlation
    R = hierarchy.dendrogram(
        #
        # NOTE: Gives the same result. See docs of ward.
        #
        # hierarchy.ward(distance.pdist(C)),
        hierarchy.ward(C),
        orientation="bottom",
        labels=features,
        no_plot=True,
        color_threshold=None
    )
    ordered_features = R["ivl"]
    return ordered_features


#
# Intersection of N dimensional plane and xy plane
# ================================================
#
# P: n . (v - a) = 0
# xy: (0, 0, 1, 1, ..., 1) . v = 0
#
# The intersection is a straight line in xy plane, i.e.
# it is of form y = k * x + b
#
# x = 0: b * e2
# x = -b / k: - b * e1 / k
#
# => n . (b * e2 - a) = 0
# => b * n . e2 - n . a = 0
#
#    ----------------------
# => b = (n . a) / (n . e2)
#    ----------------------
#
# => n . (-b/k * e1 - a) = 0
# => -b/k = (n . a) / (n . e1)
# => k = -b * (n . e1) / (n . a)
#
#    -------------------------
# => k = - (n . e1) / (n . e2)
#    -------------------------
#


def intersect_plane_xy(n_vec, a_vec):
    """Slope and intercept of a line

    The line forms the intersection between a hyperplane
    and the xy plane.

    Equation of the hyperplane is

        n_vec . x = a_vec . x

    """
    n_dims = len(n_vec)
    e0 = np.zeros(n_dims)
    e0[0] = 1.0
    e1 = np.zeros(n_dims)
    e1[1] = 1.0
    slope = -np.dot(n_vec, e0) / np.dot(n_vec, e1)
    intercept = np.dot(n_vec, a_vec) / np.dot(n_vec, e1)
    return (slope, intercept)
