"""Data analysis"""

import numpy as np


def estimate_gaussian(x):
    """Estimate the mean and covariance from a set of samples

    """
    mean = x.mean(axis=0)
    cov = np.cov(x.T)
    return (mean, cov)


def groupby_labels(x, labels):
    # x is typically a transformed version of training data
    return [x[labels == i] for i in set(labels)]


def form_labeled_dataset():
    return
