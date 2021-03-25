"""Unit tests for data handling

Run the web tests with the additional marker `webtest`

"""

from functools import reduce

import numpy as np
from pandas.testing import assert_frame_equal
import pytest

from dimred.datasets import ches2019, yle2019


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


@pytest.mark.webtest
def test_run_yle2019(tmp_path):

    f = tmp_path / "foo/bar.csv"
    f.parent.mkdir()
    f.touch()

    raw = yle2019.update(f)
    raw_loaded = yle2019.load(f)

    assert_frame_equal(raw, raw_loaded)

    assert raw.columns[6] == "Metsiä hakataan Suomessa liikaa."
    assert raw.columns[9] == "Euron ulkopuolella Suomi pärjäisi paremmin."
    assert raw.columns[12] == (
        "Parantumattomasti sairaalla on oltava oikeus eutanasiaan."
    )
    assert raw.columns[23] == (
        "Nato-jäsenyys vahvistaisi Suomen turvallisuuspoliittista asemaa."
    )
    assert raw.columns[31] == (
        "On oikein nähdä vaivaa sen eteen, ettei vahingossakaan loukkaa toista."
    )

    training_data = yle2019.cleanup(raw)  # , nan_floor_row=0, nan_floor_col=0)
    X = training_data.values
    features = list(training_data.columns)

    assert X.shape == (194, 28)
    assert X[13, 13] == 4.0
    assert X[13, 27] == 4.0
    assert np.isnan(X).sum() == 41
    assert abs(np.nanmean(X) - 3.090521) < 1e-6
    assert abs(np.nanstd(X) - 1.484716) < 1e-6
    assert np.nanmedian(X) == 4.0

    assert features[6] == "pro_euro"
    assert features[13] == "decrease_snus_import"

    return
