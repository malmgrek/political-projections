"""Unit tests for data handling

Run the web tests with the additional marker `webtest`

"""

from functools import reduce

import numpy as np
import pytest

from dimred.datasets import ches2019


@pytest.mark.webtest
def test_run_ches2019():

    raw = ches2019.download()
    assert set(ches2019.features_bounds).issubset(raw.columns)

    training_data = ches2019.prepare(
        ches2019.cleanup(raw, nan_floor_row=0, nan_floor_col=0)
    )
    X = training_data.values
    features = list(training_data.columns)

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
