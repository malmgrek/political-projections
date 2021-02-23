"""Unit tests for the Analysis module"""

import numpy as np
from dimred import analysis
from numpy.testing import assert_almost_equal, assert_array_equal

np.random.seed(42)


def test_standard_scaler():

    X_orig = 100 * np.random.rand(14, 42)
    scaler = analysis.StandardScaler().fit(X_orig)
    X_new = scaler.transform(X_orig)
    assert_almost_equal(X_new.mean(axis=0), 0)
    assert_almost_equal(X_new.std(axis=0), 1)
    assert_almost_equal(scaler.inverse_transform(X_new), X_orig)
    assert_almost_equal(scaler.transform(scaler.inverse_transform(X_new)), X_new)

    # Check that one point transforms without errors
    x = 100 * np.random.rand(42)
    x_new = scaler.transform(x)

    return


def test_interval_scaler():

    X_orig = np.array([
        [1, 0, 2, 1, -45],
        [8, 4, 1, 7, -44],
        [5, 3, 3, 5, -43],
        [6, 3, 0, 4, -42],
        [4, 1, 1, 3, -41]
    ])
    scaler = analysis.IntervalScaler([
        [0,     8],
        [0,     4],
        [0,     5],
        [1,     9],
        [-45, -40]
    ])
    X_new = scaler.transform(X_orig)
    assert_almost_equal(
        np.array([
            [-0.75,-1.  ,-0.2 ,-1.,  -1   ],
            [ 1.  , 1.  ,-0.6 , 0.5, -0.6 ],
            [ 0.25, 0.5 , 0.2 , 0.,  -0.2 ],
            [ 0.5 , 0.5 ,-1.  ,-0.25, 0.2 ],
            [ 0.  ,-0.5 ,-0.6 ,-0.5,  0.6 ]
        ]),
        X_new
    )
    assert_almost_equal(scaler.inverse_transform(X_new), X_orig)
    assert_almost_equal(scaler.transform(scaler.inverse_transform(X_new)), X_new)

    # Transform one point
    x = np.array([8, 4, 5, 9, -40])
    x_new = scaler.transform(x)
    assert_almost_equal(x_new, np.array([1, 1, 1, 1, 1]))

    return
