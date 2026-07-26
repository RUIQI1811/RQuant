import importlib.util
import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from models.elasticnet import ElasticNetModel
from models.lightgbm_model import LightGBMModel
from models.linear_ridge import RidgeModel
from models.mlp_torch import TorchMLPModel
from training.qlib_dataset import build_qlib_dataset


class ModelInterfaceTests(unittest.TestCase):
    def test_elasticnet_defaults_are_scaled_for_ranked_features(self):
        parameters = inspect.signature(ElasticNetModel).parameters
        self.assertEqual(parameters["alpha"].default, 0.001)
        self.assertEqual(parameters["l1_ratio"].default, 0.5)

    def test_torch_mlp_default_epochs_is_ten(self):
        parameter = inspect.signature(TorchMLPModel).parameters["epochs"]
        self.assertEqual(parameter.default, 10)

    @unittest.skipUnless(importlib.util.find_spec("qlib") is not None, "pyqlib is optional")
    def test_lightgbm_fit_predict(self):
        dates = pd.bdate_range("2026-01-02", periods=12)
        frame = pd.DataFrame(
            [
                {
                    "date": date,
                    "symbol": str(symbol).zfill(6),
                    "signal": float(day + symbol * 0.1),
                    "noise": float((day + symbol) % 2),
                    "target": float(day * 0.2 - symbol * 0.01),
                }
                for day, date in enumerate(dates)
                for symbol in range(1, 5)
            ]
        )
        bundle = build_qlib_dataset(
            train=frame.loc[frame["date"].isin(dates[:10])],
            test=frame.loc[frame["date"].isin(dates[10:])],
            feature_cols=("signal", "noise"),
            target_col="target",
        )
        model = LightGBMModel(
            n_estimators=5,
            n_jobs=1,
            random_state=7,
        )

        model.fit(bundle.dataset)
        predictions = model.predict(bundle.dataset)

        self.assertEqual(predictions.name, "score")
        self.assertEqual(predictions.index.names, ["datetime", "instrument"])
        self.assertEqual(predictions.index.tolist(), bundle.test_index.tolist())
        self.assertTrue(np.isfinite(predictions).all())

    def test_torch_mlp_dependency_failure_is_explicit(self):
        with patch.dict(sys.modules, {"torch": None}):
            with self.assertRaisesRegex(ImportError, "requires torch"):
                TorchMLPModel(hidden_sizes=(8,), epochs=2)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is optional")
    def test_torch_mlp_fit_predict_save_and_load(self):
        x = pd.DataFrame(
            {
                "a": np.linspace(-1.0, 1.0, 24),
                "b": np.tile([0.0, 1.0], 12),
            }
        )
        y = pd.Series(0.4 * x["a"] - 0.2 * x["b"])
        model = TorchMLPModel(
            hidden_sizes=(8,),
            epochs=20,
            batch_size=8,
            learning_rate=0.02,
            random_state=7,
            device="cpu",
        )

        model.fit(x, y)
        predictions = model.predict(x)

        self.assertEqual(predictions.name, "score")
        self.assertTrue(np.isfinite(predictions).all())
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.pt"
            model.save(path)
            restored = TorchMLPModel.load(path, device="cpu")
            restored_predictions = restored.predict(x[["b", "a"]])
        np.testing.assert_allclose(predictions, restored_predictions, rtol=1e-6, atol=1e-6)

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

    def test_numpy_elasticnet_fallback_applies_real_l1_sparsity(self):
        x = pd.DataFrame(
            {
                "signal": np.linspace(-2.0, 2.0, 80),
                "noise": np.tile([-1.0, 1.0], 40),
            }
        )
        y = pd.Series(1.5 * x["signal"])
        with patch.dict(sys.modules, {"sklearn.linear_model": None}):
            model = ElasticNetModel(alpha=0.1, l1_ratio=0.9)

        model.fit(x, y)
        predictions = model.predict(x)

        self.assertIsNone(model.model)
        self.assertGreater(abs(model.coef_[0]), 0.5)
        self.assertAlmostEqual(model.coef_[1], 0.0, places=8)
        self.assertTrue(np.isfinite(predictions).all())


if __name__ == "__main__":
    unittest.main()
