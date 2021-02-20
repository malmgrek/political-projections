import numpy as np
import pandas as pd

from poliparties.datasets import ches2019


def test_prepare():
    x = pd.DataFrame({
        "foo": [1, -1, 1, -1, 1, -1, np.nan],
        "bar": [1,  1, 1,  1, 1,  1, 0     ],
        "id":  [1,  1, 2,  2, 2,  3, 4     ]
    })
    (mean, weights) = ches2019.prepare(x, "id")
    return
