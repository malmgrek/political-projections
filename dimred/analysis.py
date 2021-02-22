"""Data analysis functionality"""

import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.impute import IterativeImputer


class IntervalScaler():
    """Affine transform from a given interval to [-1, 1]

    TODO: Unit test

    """

    def __init__(self, intervals: list):
        (a, b) = np.array(intervals).T
        self.bias = -(a + b) / (b - a) / 2.
        self.w = 1. / (b - a)
        self.inverse_bias = (a + b) / 2.
        self.inverse_w = b - a

    def transform(self, X: np.ndarray):
        return X * self.w + self.bias

    def inverse_transform(self, X: np.ndarray):
        return X * self.inverse_w + self.inverse_bias


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


def prune():
    return
