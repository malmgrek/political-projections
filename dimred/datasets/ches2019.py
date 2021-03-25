"""Data tools for CHES2019 dataset

https://www.chesdata.eu/2019-chapel-hill-expert-survey

FIXME: Fix Unnamed columns in DataFrame

"""

from collections import OrderedDict
from io import StringIO
import logging
import os
import requests

import numpy as np
import pandas as pd

from dimred import analysis


def here(*args):
    return os.path.join(os.path.dirname(__file__), *args)


url = "https://www.chesdata.eu/s/CHES2019_experts.csv"


#
# Manually picked subset of columns and their min/max limits, based on survey
# documentation PDF
#
features_bounds = OrderedDict(
    position=[1.0, 7.0],
    eu_salience=[0.0, 10.0],
    eu_dissent=[0.0, 10.0],
    eu_blur=[0.0, 10.0],
    lrecon=[0.0, 10.0],
    lrecon_blur=[0.0, 10.0],
    lrecon_dissent=[0.0, 10.0],
    lrecon_salience=[0.0, 10.0],
    galtan=[0.0, 10.0],
    galtan_blur=[0.0, 10.0],
    galtan_dissent=[0.0, 10.0],
    galtan_salience=[0.0, 10.0],
    lrgen=[0.0, 10.0],
    immigrate_policy=[0.0, 10.0],
    immigra_salience=[0.0, 10.0],
    immigrate_dissent=[0.0, 10.0],
    multiculturalism=[0.0, 10.0],
    multicult_salience=[0.0, 10.0],
    multicult_dissent=[0.0, 10.0],
    redistribution=[0.0, 10.0],
    redist_salience=[0.0, 10.0],
    environment=[0.0, 10.0],
    enviro_salience=[0.0, 10.0],
    spendvtax=[0.0, 10.0],
    deregulation=[0.0, 10.0],
    econ_interven=[0.0, 10.0],
    civlib_laworder=[0.0, 10.0],
    sociallifestyle=[0.0, 10.0],
    religious_principles=[0.0, 10.0],
    ethnic_minorities=[0.0, 10.0],
    nationalism=[0.0, 10.0],
    urban_rural=[0.0, 10.0],
    protectionism=[0.0, 10.0],
    regions=[0.0, 10.0],
    russian_interference=[0.0, 10.0],
    anti_islam_rhetoric=[0.0, 10.0],
    people_vs_elite=[0.0, 10.0],
    antielite_salience=[0.0, 10.0],
    corrupt_salience=[0.0, 10.0],
    members_vs_leadership=[0.0, 10.0],
    eu_cohesion=[1.0, 7.0],
    eu_foreign=[1.0, 7.0],
    eu_intmark=[1.0, 7.0],
    eu_budgets=[1.0, 7.0],
    eu_asylum=[1.0, 7.0],
    eu_econ_require=[1.0, 7.0],
    eu_political_require=[1.0, 7.0 ],
    eu_googov_require=[1.0, 7.0]
)


def read_csv(res):
    return pd.read_csv(StringIO(res.text))


def download():
    """Download dataset from web

    """
    logging.info("GET {}".format(url))
    res = requests.get(url)
    res.raise_for_status()
    return read_csv(res)


def update(filepath=here("cache", "ches2019.csv")):
    """Download and save

    """
    x = download()
    logging.info("Saved to file {}".format(filepath))
    x.to_csv(filepath)
    return x


def load(filepath=here("cache", "ches2019.csv")):
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
