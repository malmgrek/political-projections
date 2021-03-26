"""Data tools for Yle 2019 Finnish parlamentary election questionnaire

"""

from collections import OrderedDict
from io import BytesIO, StringIO
import logging
import os
import requests
import zipfile

import numpy as np
import pandas as pd

from dimred import analysis


def here(*args):
    return os.path.join(os.path.dirname(__file__), *args)


url = "https://vaalit.beta.yle.fi/avoindata/avoin_data_eduskuntavaalit_2019.zip"


#
# Manually picked and renamed subset of columns and their min/max limits.
#
features_bounds = OrderedDict(
    climate_change=[1, 5],
    pro_cars=[1, 5],
    carnivore_tax=[1, 5],
    preserve_forests=[1, 5],
    austerity=[1, 5],
    basic_income=[1, 5],
    pro_euro=[1, 5],
    public_sohe_services=[1, 5],
    privatize_eldercare=[1, 5],
    euthanasy=[1, 5],
    gender_change_under18=[1, 5],
    wines_to_supermarkets=[1, 5],
    forbid_energydrinks_u15=[1, 5],
    decrease_snus_import=[1, 5],
    equal_parental_leave=[1, 5],
    extend_education=[1, 5],
    fshift_summer_vacay=[1, 5],
    uni_qualvquant=[1, 5],
    immigrants_dangerous=[1, 5],
    immigrants_needed=[1, 5],
    nato_membership=[1, 5],
    hatespeech_criminalize=[1, 5],
    traditional_values=[1, 5],
    hard_police=[1, 5],
    econ_inequality=[1, 5],
    individual_responsibility=[1, 5],
    voter_loyalty=[1, 5],
    principles=[1, 5],
    political_correctness=[1, 5]
)


def download():
    """Download and unzip the raw dataset

    """
    logging.info("GET {}".format(url))
    res = requests.get(url)
    zf = zipfile.ZipFile(BytesIO(res.content))
    return pd.read_csv(
        BytesIO(zf.read("Avoin_data_eduskuntavaalit_2019_valintatiedot.csv"))
    )


def cleanup(
        x: pd.DataFrame,
        nan_floor_row=0.9,
        nan_floor_col=0.75
):
    # Select only those who made it to the parliament
    # Also, party information is discarded
    x = x[x.iloc[:, 2] == 1].iloc[:, 4:33]
    # Rename columns with shortened names
    x.columns = list(features_bounds)
    # Fix data types column-wise
    x = x.apply(lambda s: pd.to_numeric(s, errors="coerce"))
    # Drop columns
    x = x.dropna(axis=1, thresh=nan_floor_col*x.shape[0])
    # Drop rows
    x = x.dropna(axis=0, thresh=nan_floor_row*x.shape[1])
    return x


def update(filepath=here("resources", "yle2019.csv")):
    """Download and cache compressed dataset

    """
    x = cleanup(download(), nan_floor_row=0.9, nan_floor_col=0.75)
    logging.info("Saved to file {}".format(filepath))
    x.to_csv(filepath)
    return x


def load(filepath=here("resources", "yle2019.csv")):
    return pd.read_csv(filepath, index_col=0)


def create_scaler(X, features, normalize: bool):
    (a, b) = np.array([features_bounds[f] for f in features]).T
    return (
        analysis.UnitScaler(X) if normalize else
        analysis.IntervalScaler(a=a, b=b)
    )


def create_training_data(
        cleaned_data,
        normalize: bool,
        impute: bool,
        corrcov: str
):
    """Create training data, features list and scaler object

    """

    # Optionally impute
    X = (
        analysis.impute_missing(cleaned_data.values, max_iter=21)
        if impute
        else cleaned_data.dropna().values
    )
    features = list(cleaned_data.columns)

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


#
# Dash app related text fields etc
#

app_data = {
    "title": "YLE 2019 Finnish electoral candidate survey",
    "description": (
        "Dimensionality reduction of the survey responses. Various common methods "
        "are supported. The original set of survey questions has been reduced to "
        "the politically 'most interesting' questions directly related to values, "
        "economy, etc. Contains only those candidates who got through in the "
        "2019 Finnish parliamentary elections."
    ),
    "information": "https://yle.fi/uutiset/3-10725384"
}
