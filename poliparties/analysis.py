"""Data analysis"""


# TODO: Use sample number weights in KDE


from typing import List

import numpy as np
import pandas as pd


def estimate_gaussian(y: np.ndarray):
    """Estimate the mean and covariance from a set of samples

    """
    mean = y.mean(axis=0)
    cov = np.NaN if len(y) == 1 else np.cov(y.T)
    return [mean, cov]


def split_by_labels(y: np.ndarray, labels: List):
    # x is typically a transformed version of training data
    index = list(set(labels))
    data = [(i, y[labels == i]) for i in set(labels)]
    return pd.Series(data, index)


def estimate_spheres(y: np.ndarray, labels: List):
    groups = split_by_labels(y, labels)
    data = [
        [mean[0], mean[1], np.sqrt(np.linalg.det(cov))]
        for (mean, cov) in map(estimate_gaussian, groups)
    ]
    return pd.DataFrame(
        data=data,
        index=groups.index,
        columns=["mean_x", "mean_y", "r_std"]
    )
