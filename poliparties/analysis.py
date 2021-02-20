"""Data analysis"""


# TODO: Use sample number weights in KDE


import numpy as np
import pandas as pd


def estimate_gaussian(y: np.ndarray):
    """Estimate the mean and covariance from a set of samples

    """
    y = np.array(y)
    mean = y.mean(axis=0)
    cov = np.NaN if len(y) < 3 else np.cov(y.T)
    return [mean, cov]


def split_by_labels(y: np.ndarray, labels: np.ndarray):
    # x is typically a transformed version of training data
    label_counts = pd.Series(labels).value_counts()
    return [y[labels == i, :] for i in label_counts.index]


def estimate_spheres(y: np.ndarray, labels: np.ndarray):
    # TODO: Unit test
    groups = split_by_labels(y, labels)
    dim = y.shape[1]
    data = [
        # sqrt: var -> std
        # nth root: area -> radius
        np.hstack((
            mean,
            [0] if np.isnan(np.sum(cov)) else
            [np.linalg.det(cov) ** (1 / dim / 2)]
        ))
        for (mean, cov) in map(estimate_gaussian, groups)
    ]
    return pd.DataFrame(
        data=data,
        index=pd.Series(labels).value_counts(),
        columns=["mean_x", "mean_y", "r_std"]
    )
