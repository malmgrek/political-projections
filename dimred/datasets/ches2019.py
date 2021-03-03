"""Data tools for CHES2019 dataset

https://www.chesdata.eu/2019-chapel-hill-expert-survey

FIXME: Fix Unnamed columns in DataFrame

"""

from io import StringIO
import logging
import os
import requests

import numpy as np
import pandas as pd
from sklearn import decomposition

from dimred import analysis


def here(*args):
    return os.path.join(os.path.dirname(__file__), *args)


#
# Manually picked subset of columns and their min/max limits, based on survey
# documentation PDF
#
features_bounds = {
    "position": [1.0, 7.0],
    "eu_salience": [0.0, 10.0],
    "eu_dissent": [0.0, 10.0],
    "eu_blur": [0.0, 10.0],
    "lrecon": [0.0, 10.0],
    "lrecon_blur": [0.0, 10.0],
    "lrecon_dissent": [0.0, 10.0],
    "lrecon_salience": [0.0, 10.0],
    "galtan": [0.0, 10.0],
    "galtan_blur": [0.0, 10.0],
    "galtan_dissent": [0.0, 10.0],
    "galtan_salience": [0.0, 10.0],
    "lrgen": [0.0, 10.0],
    "immigrate_policy": [0.0, 10.0],
    "immigra_salience": [0.0, 10.0],
    "immigrate_dissent": [0.0, 10.0],
    "multiculturalism": [0.0, 10.0],
    "multicult_salience": [0.0, 10.0],
    "multicult_dissent": [0.0, 10.0],
    "redistribution": [0.0, 10.0],
    "redist_salience": [0.0, 10.0],
    "environment": [0.0, 10.0],
    "enviro_salience": [0.0, 10.0],
    "spendvtax": [0.0, 10.0],
    "deregulation": [0.0, 10.0],
    "econ_interven": [0.0, 10.0],
    "civlib_laworder": [0.0, 10.0],
    "sociallifestyle": [0.0, 10.0],
    "religious_principles": [0.0, 10.0],
    "ethnic_minorities": [0.0, 10.0],
    "nationalism": [0.0, 10.0],
    "urban_rural": [0.0, 10.0],
    "protectionism": [0.0, 10.0],
    "regions": [0.0, 10.0],
    "russian_interference": [0.0, 10.0],
    "anti_islam_rhetoric": [0.0, 10.0],
    "people_vs_elite": [0.0, 10.0],
    "antielite_salience": [0.0, 10.0],
    "corrupt_salience": [0.0, 10.0],
    "members_vs_leadership": [0.0, 10.0],
    "eu_cohesion": [1.0, 7.0],
    "eu_foreign": [1.0, 7.0],
    "eu_intmark": [1.0, 7.0],
    "eu_budgets": [1.0, 7.0],
    "eu_asylum": [1.0, 7.0],
    "eu_econ_require": [1.0, 7.0],
    "eu_political_require": [1.0, 7.0 ],
    "eu_googov_require": [1.0, 7.0]
}


url = "https://www.chesdata.eu/s/CHES2019_experts.csv"


def read_csv(res):
    return pd.read_csv(StringIO(res.text))


def download():
    """Download dataset from web

    """
    logging.info("GET {}".format(url.format(url)))
    res = requests.get("https://www.chesdata.eu/s/CHES2019_experts.csv")
    res.raise_for_status()
    return read_csv(res)


def update(filepath=here("cache", "dump.csv")):
    """Download and save

    """
    x = download()
    logging.info("Saved to file {}".format(filepath))
    x.to_csv(filepath)
    return x


def load(filepath=here("cache", "dump.csv")):
    """Load from disk

    """
    return pd.read_csv(filepath)


def cleanup(
        x: pd.DataFrame,
        nan_floor_row=0.9,
        nan_floor_col=0.75,
        columns=list(features_bounds)+["party_id"]
):
    """Select subset of columns, fix data types, remove NaN

    """
    # Drop unwanted columns
    x = x[columns]
    # Fix data types column-wise
    x = x.apply(lambda s: pd.to_numeric(s, errors="coerce"))
    # Drop columns
    x = x.dropna(axis=1, thresh=nan_floor_col*x.shape[0])
    # Drop rows
    x = x.dropna(axis=0, thresh=nan_floor_row*x.shape[1])
    return x


def prepare(
        x: pd.DataFrame,
        groupby_feature="party_id",
) -> np.ndarray:
    """Group by parties and build weights for cells

    """
    return x.groupby(x[groupby_feature]).median()


def create_scaler(X, features, normalize_bool):
    """Construct a scaler based on features

    """
    (a, b) = np.array([features_bounds[f] for f in features]).T
    return (
        analysis.UnitScaler(X) if normalize_bool else
        analysis.IntervalScaler(a=a, b=b)
    )


def create_dataset(data, normalize: bool, impute: bool, corrcov: str):
    """Create training data, features list and scaler object

    """

    training_data = prepare(
        cleanup(
            data, nan_floor_row=0.9, nan_floor_col=0.75
        )
    )

    # Optionally impute
    X = (
        analysis.impute_missing(training_data.values, max_iter=21)
        if impute
        else training_data.dropna().values
    )
    features = list(training_data.columns)

    # So that xs -> ys
    find_permutation = lambda xs, ys: [xs.index(y) for y in ys]

    # Re-order features
    ordered_features = analysis.order_features(
        # Ordering works better with scaled data
        create_scaler(X, features, normalize).transform(X),
        features,
        corrcov
    )
    X = X[:, find_permutation(features, ordered_features)]
    # Create new scaler for further use using the re-ordered features set
    scaler = create_scaler(X, ordered_features, normalize)
    # Scale training data
    X = scaler.transform(X)

    return (X, ordered_features, scaler)


def analyze_pca(
        X,
        features,
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
    kde = analysis.fit_kde(Y_2d)
    Y_samples = kde.sample(num_samples)
    (x, y, density, xlim, ylim) = analysis.score_density_grid(
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
        return analysis.intersect_plane_xy(
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
    kde = analysis.fit_kde(Y_2d)
    Y_samples = kde.sample(num_samples)
    (x, y, density, xlim, ylim) = analysis.score_density_grid(
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
    kde = analysis.fit_kde(Y_2d)
    Y_samples = kde.sample(num_samples)
    (x, y, density, xlim, ylim) = analysis.score_density_grid(
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
