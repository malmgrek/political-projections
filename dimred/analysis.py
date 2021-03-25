"""Data analysis functionality"""

from functools import reduce

import numpy as np
from scipy.cluster import hierarchy
from scipy.stats import spearmanr
from sklearn import decomposition
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
    params = {'bandwidth': np.logspace(-2, 2, 20)}
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


def analyze_pca(
        X,
        features,
        features_bounds,
        scaler,
        components=2,
        num_samples=10,
        norint=True,
        **unused
):

    bounds = np.array([features_bounds[f] for f in features])

    decomposer = decomposition.PCA(whiten=False).fit(X)
    statistics = {
        "explained_variance": (
            decomposer.explained_variance_ratio_[:components].sum()
        )
    }

    Y = decomposer.transform(X)
    Y_2d = Y[:, :2]
    U = decomposer.components_
    V = scaler.inverse_transform(U)

    # Fit KDE and sample
    kde = fit_kde(Y_2d)
    Y_samples = kde.sample(num_samples)
    (x, y, density, xlim, ylim) = score_density_grid(
        kde=kde, Y=Y_2d, num=100
    )

    # "Inverse" transform
    Y_samples_full = np.hstack((
        Y_samples,
        np.zeros((num_samples, len(features) - 2))
    ))
    X_samples = scaler.inverse_transform(
        decomposer.inverse_transform(Y_samples_full)
    )
    X_samples = (
        X_samples if norint else np.clip(np.rint(X_samples), *bounds.T)
    )

    # Deduce bound lines (slope, intercept) in 2D
    rotate = lambda x: np.dot(U, x)
    translate = lambda x, n: x - np.dot(X.mean(axis=0), n) * n
    n_dims = len(features)
    bounds_scaled = scaler.transform(bounds.T)

    def mapper(i, c):
        n_vec = (np.arange(n_dims) == i) * c
        a_vec = n_vec
        #
        # PCA transformation first shifs to zero mean and then rotates.
        # For plane's vector geometry it means that the NORMAL VECTOR is
        # just rotated and the OFFSET VECTOR is
        #
        # (1) translated in the plane normal direction
        # (2) rotated by the rotation
        #
        return intersect_plane_xy(
            rotate(n_vec),
            rotate(translate(a_vec, n_vec))
        )

    min_bounds = list(map(
        lambda args: mapper(*args), enumerate(bounds_scaled[0])
    ))
    max_bounds = list(map(
        lambda args: mapper(*args), enumerate(bounds_scaled[1])
    ))

    return {
        "reduced_bounds": (min_bounds, max_bounds),
        "decomposition": (U, V, Y_2d, Y_samples, X_samples, statistics),
        "density": (x, y, density, xlim, ylim),
    }


def analyze_ica(
        X,
        features,
        features_bounds,
        scaler,
        components=2,
        num_samples=10,
        norint=True,
        **unused
):

    bounds = np.array([features_bounds[f] for f in features])

    decomposer = decomposition.FastICA(
        n_components=components,
        random_state=np.random.RandomState(42),
        whiten=True,
        max_iter=1000,
        # tol=1e-1
    ).fit(X)

    Y = decomposer.transform(X)
    Y_2d = Y[:, :2]
    U = decomposer.components_
    V = scaler.inverse_transform(U)

    # Fit KDE and sample
    kde = fit_kde(Y_2d)
    Y_samples = kde.sample(num_samples)
    (x, y, density, xlim, ylim) = score_density_grid(
        kde=kde, Y=Y_2d, num=100
    )

    # "Inverse" transform
    X_samples = scaler.inverse_transform(
        decomposer.inverse_transform(Y_samples)
    )
    X_samples = (
        X_samples if norint else np.clip(np.rint(X_samples), *bounds.T)
    )

    return {
        "reduced_bounds": (None, None),
        "decomposition": (U, V, Y_2d, Y_samples, X_samples, None),
        "density": (x, y, density, xlim, ylim)
    }


def analyze_fa(
        X,
        features,
        features_bounds,
        scaler,
        components=2,
        num_samples=10,
        **unused
):

    bounds = np.array([features_bounds[f] for f in features])

    decomposer = decomposition.FactorAnalysis(
        n_components=components, rotation="varimax"
    ).fit(X)

    Y = decomposer.transform(X)
    Y_2d = Y[:, :2]
    U = decomposer.components_
    V = scaler.inverse_transform(U)

    # Fit KDE and sample
    kde = fit_kde(Y_2d)
    Y_samples = kde.sample(num_samples)
    (x, y, density, xlim, ylim) = score_density_grid(
        kde=kde, Y=Y_2d, num=100
    )

    # "Inverse" transform
    # TODO X_samples = ...

    return {
        "reduced_bounds": (None, None),
        "decomposition": (U, V, Y_2d, Y_samples, None, None),
        "density": (x, y, density, xlim, ylim)
    }


def calculate_pca_statistics(X):
    decomposer = decomposition.PCA().fit(X)
    return (
        decomposer.singular_values_,
        decomposer.explained_variance_ratio_
    )
