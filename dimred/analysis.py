"""Data analysis functionality"""

from functools import reduce
import itertools

import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.impute import IterativeImputer


class IntervalScaler():
    """Affine transform from a given interval to [-1, 1]

    forward: (x - 0.5 * (a + b)) / (b - a)
    inverse: y * (b - a) + 0.5 * (b - a)

    """

    def __init__(self, intervals: list, whiten=False):
        (a, b) = np.array(intervals).T
        self.a = a
        self.b = b

    def transform(self, X: np.ndarray):
        return 2 * (X - 0.5 * (self.a + self.b)) / (self.b - self.a)

    def inverse_transform(self, Y: np.ndarray):
        return 0.5 * Y * (self.b - self.a) + 0.5 * (self.b + self.a)


class StandardScaler():

    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def fit(self, X):
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        return StandardScaler(mean=mean, std=std)

    def transform(self, X):
        return (X - self.mean) / self.std

    def inverse_transform(self, Y):
        return Y * self.std + self.mean


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


def impute(X, *args, **kwargs):
    """Impute missing values using Sklearn iterative imputer

    """
    imputer = IterativeImputer(*args, **kwargs).fit(X)
    return imputer.transform(X)


def cuboid_edges(vtxs):
    """Edges of an N dimensional cuboid as vertex pairs

    """
    n_dims = len(vtxs[0])
    vtxs = list(map(tuple, vtxs))  # Enable hashing
    pairs = reduce(
        # Reduce to the pair level
        lambda cum, this: list(cum) + list(this),
        (
            # Form pairs with the base vertex
            # and sort by distance
            itertools.product(
                (v,),
                sorted(
                    vtxs,
                    key=lambda u: np.linalg.norm(
                        np.subtract(u, v)
                    )
                )[1:n_dims+1]  # Take three nearest
            ) for v in vtxs
        )
    )
    unique_pairs = reduce(
        lambda cum, this: cum if (
            this in cum or this[::-1] in cum
        ) else cum + [this],
        pairs,
        []
    )
    return unique_pairs


def intersect_plane2_cuboid(normal, a, vtxs):
    """Polygon resulting from intersection a N-cube and 2-plane

    """
    vtx_pairs = cuboid_edges(vtxs)
    points = []
    for (vi, vj) in vtx_pairs:
        t = (
            np.dot(np.subtract(vj, vi), normal) /
            np.float64(np.dot(np.subtract(a, vi), normal))
        )
        if t >= 1:
            p = np.add(vi, np.subtract(vj, vi) / t)
            points = points + [p]
    return np.array(points)
