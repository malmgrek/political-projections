"""Data tools for CHES2019 dataset

https://www.chesdata.eu/2019-chapel-hill-expert-survey

FIXME: Fix Unnamed columns in DataFrame

"""

from io import StringIO
import os
import requests

import numpy as np
import pandas as pd


def here(*args):
    return os.path.join(os.path.dirname(__file__), *args)


#
# Manually picked subset of columns and their min/max limits, based on survey
# documentation PDF
#
feature_scales = {
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
    """Download dataset from web

    """
    res = requests.get("https://www.chesdata.eu/s/CHES2019_experts.csv")
    return pd.read_csv(StringIO(res.text))


def update(filepath=here("cache", "dump.csv")):
    """Download and save

    """
    x = download()
    x.to_csv(filepath)
    return


def load(filepath=here("cache", "dump.csv")):
    """Load from disk

    """
    return pd.read_csv(filepath)


def cleanup(
        x: pd.DataFrame,
        nan_floor_row=0.9,
        nan_floor_col=0.75,
        columns=list(feature_scales)+["party_id"]
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
    agg = x.groupby(x[groupby_feature]).median()
    weights = agg.notnull().mul(1)
    # X_train, w, features
    return (agg.values, weights.values, agg.columns)
