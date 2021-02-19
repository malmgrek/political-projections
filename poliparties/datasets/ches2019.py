"""Data tools for CHES2019 dataset

https://www.chesdata.eu/2019-chapel-hill-expert-survey

"""

from io import StringIO
import os
import requests
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from poliparties import utils


def here(*args):
    return os.path.join(os.path.dirname(__file__), *args)


#
# Manually picked subset of columns and their min/max limits, based on survey docs
#
COLUMN_INTERVALS = {
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


def download():
    res = requests.get("https://www.chesdata.eu/s/CHES2019_experts.csv")
    return pd.read_csv(StringIO(res.text))


def update(fp=here("cache", "dump.csv")):
    x = download()
    x.to_csv(fp)
    return


def load(fp=here("cache", "dump.csv")):
    return pd.read_csv(fp)


def load_training_data(
        online=True,
        nan_floor_row=0.9,
        nan_floor_col=0.75,
):
    """Load cleaned up training dataset

    Parameters
    ----------
    offline : bool
        Use cached dataset
    na_row : float
        Drop rows with NaN ratio larger than this
    na_col : float
        Drop columns with NaN ratio larger than this

    """

    x = load() if not online else download()
    x = x[COLUMN_INTERVALS]

    # Fix data types column-wise
    x = x.apply(lambda s: pd.to_numeric(s, errors="coerce"))
    # Drop columns
    x = x.dropna(axis=1, thresh=nan_floor_col*x.shape[0])
    # Drop rows
    x = x.dropna(axis=0, thresh=nan_floor_row*x.shape[1])

    X = x.values

    # Scale to reasonable interval
    scaler = utils.IntervalScaler([
        v for (k, v) in COLUMN_INTERVALS.items() if k in x.columns
    ])
    X = scaler.transform(X)

    # Impute missing values
    imputer = IterativeImputer(max_iter=100, random_state=0).fit(X)
    X = imputer.transform(X)

    return (X, x.index, x.columns)
