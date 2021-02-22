"""Unit tests for data handling

Run the web tests with the additional marker `webtest`

"""

from functools import reduce

import numpy as np
import pytest

from dimred.datasets import ches2019


@pytest.mark.webtest
def test_run_ches2019():

    x = ches2019.download()
    assert set(ches2019.feature_scales).issubset(x.columns)

    (X, features) = ches2019.prepare(
        ches2019.cleanup(x, nan_floor_row=0, nan_floor_col=0)
    )

    assert X.shape == (277, 48)
    assert X[42, 42] == 6.0
    assert X[13, 27] == 5.0
    assert np.isnan(X).sum() == 1007
    assert abs(np.nanmean(X) - 4.503173) < 1e-6
    assert abs(np.nanstd(X) - 2.4980646) < 1e-6
    assert np.nanmedian(X) == 4.5

    assert features[6] == "lrecon_dissent"
    assert features[13] == "immigrate_policy"

    return
