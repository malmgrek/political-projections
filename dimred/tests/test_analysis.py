"""Unit tests for the Analysis module"""

import numpy as np
from dimred import analysis
from numpy.testing import assert_almost_equal, assert_array_equal

np.random.seed(42)


def test_unit_scaler():

    X_orig = 100 * np.random.rand(14, 42)
    scaler = analysis.UnitScaler(X_orig)
    X_new = scaler.transform(X_orig)
    assert_almost_equal(X_new.mean(axis=0), 0)
    assert_almost_equal(X_new.std(axis=0), 1)
    assert_almost_equal(scaler.inverse_transform(X_new), X_orig)
    assert_almost_equal(scaler.transform(scaler.inverse_transform(X_new)), X_new)

    # Check that one point transforms without errors
    x = 100 * np.random.rand(42)
    x_new = scaler.transform(x)

    X = np.random.rand(66, 42)
    X[13, 7] = np.NaN
    scaler = analysis.UnitScaler(X)
    dict_scaler = analysis.AffineScaler.from_dict(scaler.to_dict())
    assert_almost_equal(scaler.w, dict_scaler.w)
    assert_almost_equal(scaler.bias, dict_scaler.bias)

    return


def test_interval_scaler():

    X_orig = np.array([
        [1, 0, 2, 1, -45],
        [8, 4, 1, 7, -44],
        [5, 3, 3, 5, -43],
        [6, 3, 0, 4, -42],
        [4, 1, 1, 3, -41]
    ])
    scaler = analysis.IntervalScaler(
        a=[0, 0, 0, 1, -45],
        b=[8, 4, 5, 9, -40]
    )
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

    # Serializing DICT
    a = np.array([1, np.pi, np.exp(1), np.NaN, np.inf])
    b = np.array([-1, 666, 42, 0, np.NaN])
    scaler = analysis.IntervalScaler(a=a, b=b)
    dict_scaler = analysis.AffineScaler.from_dict(scaler.to_dict())
    assert_almost_equal(dict_scaler.w, scaler.w)
    assert_almost_equal(dict_scaler.bias, scaler.bias)

    return
