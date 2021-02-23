"""Data analysis functionality"""

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


def project_to_axes2d(x, proj, shift=0):
    """Calculates normalized projected vector and it's orthogonal vector

    """
    y = np.dot(proj, x + shift)
    y_ = np.dot([[0, 1], [-1, 0]], y)
    y_ = y_ / np.linalg.norm(y_)
    import pdb; pdb.set_trace()
    return (y, y_)


def bounds_to_axes2d(
        interval=[-1, 1],
        dim=0,
        n_dims=3,
        transform=lambda t: t,
        **kwargs
):
    e = (np.arange(n_dims) == dim) * 1
    e0 = transform(interval[0]) * e
    e1 = transform(interval[1]) * e
    return (
        project_to_axes2d(e0, **kwargs),
        project_to_axes2d(e1, **kwargs)
    )
