import unittest

import numpy as np
import pandas as pd

from models.elasticnet import ElasticNetModel
from models.linear_ridge import RidgeModel


class ModelInterfaceTests(unittest.TestCase):
    def test_ridge_fit_predict(self):
        x = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [0.0, 1.0, 0.0, 1.0]})
        y = pd.Series([0.1, 0.2, 0.3, 0.4])
        model = RidgeModel(alpha=1.0)

        model.fit(x, y)
        predictions = model.predict(x)

        self.assertEqual(len(predictions), 4)
        self.assertTrue(np.isfinite(predictions).all())

    def test_elasticnet_fit_predict(self):
        x = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [0.0, 1.0, 0.0, 1.0]})
        y = pd.Series([0.1, 0.2, 0.3, 0.4])
        model = ElasticNetModel(alpha=0.1, l1_ratio=0.5)

        model.fit(x, y)
        predictions = model.predict(x)

        self.assertEqual(len(predictions), 4)
        self.assertTrue(np.isfinite(predictions).all())


if __name__ == "__main__":
    unittest.main()
