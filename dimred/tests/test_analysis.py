"""Unit tests for the Analysis module"""


import numpy as np
from numpy.testing import assert_almost_equal

from dimred import analysis


np.random.seed(42)


def test_standard_scaler():
    X_orig = 100 * np.random.rand(14, 8)
    scaler = analysis.StandardScaler().fit(X_orig)
    X_new = scaler.transform(X_orig)
    assert_almost_equal(X_new.mean(axis=0), 0)
    assert_almost_equal(X_new.std(axis=0), 1)
    assert_almost_equal(scaler.inverse_transform(X_new), X_orig)
    assert_almost_equal(scaler.transform(scaler.inverse_transform(X_new)), X_new)
    return


def test_interval_scaler():
    X_orig = np.array([
        [1, 0, 2, 1],
        [8, 4, 1, 7],
        [5, 3, 3, 5],
        [6, 3, 0, 4],
        [4, 1, 1, 3]
    ])
    scaler = analysis.IntervalScaler([
        [0, 8],
        [0, 4],
        [0, 5],
        [1, 9]
    ])
    X_new = scaler.transform(X_orig)
    assert_almost_equal(
        np.array([
            [-0.75,-1.  ,-0.2 ,-1.  ],
            [ 1.  , 1.  ,-0.6 , 0.5 ],
            [ 0.25, 0.5 , 0.2 , 0.  ],
            [ 0.5 , 0.5 ,-1.  ,-0.25],
            [ 0.  ,-0.5 ,-0.6 ,-0.5 ]
        ]),
        X_new
    )
    assert_almost_equal(scaler.inverse_transform(X_new), X_orig)
    assert_almost_equal(scaler.transform(scaler.inverse_transform(X_new)), X_new)
    return
