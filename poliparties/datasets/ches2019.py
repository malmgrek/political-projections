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


def cleanup(
        x: pd.DataFrame,
        nan_floor_row=0.9,
        nan_floor_col=0.75,
        columns=list(feature_scales)+["party_id"]
):
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
    """Group by parties and build heuristic weights for cells


    The weights are calculated with the following logic:

    - NaN's have zero weight.
    - Otherwise the weight of each grouped cell is the number of samples
      divided by estimated variance.
    - The group variance is estimated by a weighted sum of (1) the mean of cell variances
      belonging to the same feature and (2) the sample variance of the cell.
    - If there is only one sample in the cell, the mean variance is used.

    NOTE: The weights are sensible only if data is in comparable units.

    """
    mean = x.groupby(x[groupby_feature]).mean()
    var = x.groupby(x[groupby_feature]).var()
    var_0 = var.mean(axis=0)
    var_0 = 1
    counts = x[groupby_feature].value_counts().loc[mean.index]
    var_estimate = (
        var
        .fillna(var_0)
        .mul(counts, axis=0)
        .add(var_0)
        .div(counts.add(1), axis=0)
    ).add(1)
    weights = (
        1. / var_estimate.div(counts.pow(0.5), axis=0)
    ).mul(mean.notnull().mul(1))
    return (mean, weights)
