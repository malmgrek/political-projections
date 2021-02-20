"""Helpers and utilities"""

import functools
import pdb

import numpy as np


def with_bp(x):
    """Breakpoint wrapper

    """
    pdb.set_trace()
    return x


def identity(x):
    return x


def curry1(f):
    """Curry first argument

    """
    return lambda *args, **kwargs: functools.parial(f, *args, **kwargs)


def compose2(f, g):
    """Composition of two functions

    """
    return lambda *args, **kwargs: f(g(*args, **kwargs))


def compose(*fs):
    """Compose arbitrary number of functions

    """
    return functools.partial(functools.reduce, compose2)(fs)


def pipe(arg, *fs):
    """Pipe operator as a function

    """
    return compose(fs[::-1])(arg)


# Fmap for lists with curried first argument
listmap = curry1(compose(list, map))
arraymap = curry1(compose(np.array, list, map))


#
# Data analysis utilities
#


class IntervalScaler():
    """Affine transform from a given interval to [-1, 1]

    TODO: Unit test

    """

    def __init__(self, intervals: list):
        (a, b) = np.array(intervals).T
        self.bias = -(a + b) / (b - a) / 2.
        self.w = 1. / (b - a)
        self.inverse_bias = (a + b) / 2.
        self.inverse_w = b - a

    def transform(self, X: np.ndarray):
        return X * self.w + self.bias

    def inverse_transform(self, X: np.ndarray):
        return X * self.inverse_w + self.inverse_bias
