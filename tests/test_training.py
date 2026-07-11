import json
import importlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from training.predict_score import scores_to_signals
from training.train_walk_forward import (
    WalkForwardTrainingConfig,
    infer_target_horizon,
    run_walk_forward_training,
)
from training.validation import (
    WalkForwardWindow,
    build_walk_forward_windows,
    validate_feature_label_frame,
)


class TrainingValidationTests(unittest.TestCase):
    def test_optional_backends_complete_real_walk_forward_windows(self):
        backends = []
        for model_name, module_name in (
            ("elasticnet", "sklearn"),
            ("lightgbm", "lightgbm"),
            ("mlp", "torch"),
        ):
            try:
                importlib.import_module(module_name)
            except Exception:
                continue
            backends.append(model_name)
        if not backends:
            self.skipTest("optional model backends are not importable")

        dates = pd.bdate_range("2026-01-02", periods=9)
        feature_rows = []
        label_rows = []
        for day_index, date in enumerate(dates):
            for symbol_index, symbol in enumerate(("000001", "000002")):
                feature = float(day_index + symbol_index * 0.25)
                feature_rows.append({"date": date, "symbol": symbol, "feature": feature})
                label_rows.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "forward_return_1d": 0.1 * feature - 0.01 * symbol_index,
                    }
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            features_path = root / "features.csv"
            labels_path = root / "labels.csv"
            pd.DataFrame(feature_rows).to_csv(features_path, index=False)
            pd.DataFrame(label_rows).to_csv(labels_path, index=False)

            for model_name in backends:
                with self.subTest(model=model_name):
                    output_dir = root / model_name
                    config = WalkForwardTrainingConfig(
                        feature_cols=("feature",),
                        target_col="forward_return_1d",
                        model=model_name,
                        train_size=5,
                        test_size=2,
                        purge_days=1,
                        signal_top_n=1,
                        lightgbm_estimators=5,
                        mlp_hidden_sizes=(4,),
                        mlp_epochs=2,
                        mlp_batch_size=8,
                        device="cpu",
                    )
                    outputs = run_walk_forward_training(
                        features_path=features_path,
                        labels_path=labels_path,
                        output_dir=output_dir,
                        config=config,
                    )
                    predictions = pd.read_csv(outputs["predictions_path"])
                    summary = json.loads(
                        Path(outputs["summary_path"]).read_text(encoding="utf-8")
                    )

                    self.assertTrue(np.isfinite(predictions["score"]).all())
                    self.assertGreater(summary["prediction_count"], 0)
                    artifact_name = "model.pt" if model_name == "mlp" else "model.pkl"
                    self.assertTrue(any(output_dir.glob(f"windows/*/{artifact_name}")))

    def test_walk_forward_windows_insert_purge_gap(self):
        dates = pd.date_range("2026-01-01", periods=9, freq="B")

        windows = build_walk_forward_windows(
            dates=dates,
            train_size=3,
            test_size=2,
            purge_size=2,
        )

        self.assertEqual(windows[0].train_end, dates[2])
        self.assertEqual(windows[0].purge_start, dates[3])
        self.assertEqual(windows[0].purge_end, dates[4])
        self.assertEqual(windows[0].test_start, dates[5])
        self.assertEqual(windows[0].test_end, dates[6])

    def test_target_horizon_is_inferred_but_generic_target_requires_explicit_purge(self):
        self.assertEqual(infer_target_horizon("forward_return_20d"), 20)
        self.assertEqual(infer_target_horizon("next_open_return_20d"), 21)
        self.assertEqual(infer_target_horizon("forward_return_20d_cs_rank"), 20)
        self.assertEqual(infer_target_horizon("next_open_return_20d_cs_zscore"), 21)
        self.assertIsNone(infer_target_horizon("target"))

    def test_walk_forward_training_writes_only_out_of_sample_scores_and_ranked_signals(self):
        dates = pd.bdate_range("2025-01-02", periods=12)
        feature_rows = []
        label_rows = []
        for day_index, date in enumerate(dates):
            for symbol_index, symbol in enumerate(("000001", "000002")):
                feature = float(day_index + symbol_index * 0.5)
                feature_rows.append(
                    {"date": date, "symbol": symbol, "feature": feature}
                )
                label_rows.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "forward_return_1d": 0.2 * feature + symbol_index * 0.01,
                    }
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            features_path = root / "features.csv"
            labels_path = root / "labels.csv"
            output_dir = root / "output"
            pd.DataFrame(feature_rows).to_csv(features_path, index=False)
            pd.DataFrame(label_rows).to_csv(labels_path, index=False)
            config = WalkForwardTrainingConfig(
                feature_cols=("feature",),
                target_col="forward_return_1d",
                model="ridge",
                train_size=5,
                test_size=2,
                purge_days=1,
                signal_top_n=1,
                ridge_alpha=0.01,
            )

            outputs = run_walk_forward_training(
                features_path=features_path,
                labels_path=labels_path,
                output_dir=output_dir,
                config=config,
            )

            resumed_outputs = run_walk_forward_training(
                features_path=features_path,
                labels_path=labels_path,
                output_dir=output_dir,
                config=config,
            )

            predictions = pd.read_csv(outputs["predictions_path"], dtype={"symbol": str})
            signals = pd.read_csv(outputs["signals_path"], dtype={"symbol": str})
            windows = pd.read_csv(outputs["windows_path"])
            summary = json.loads(Path(outputs["summary_path"]).read_text(encoding="utf-8"))
            resumed_summary = json.loads(
                Path(resumed_outputs["summary_path"]).read_text(encoding="utf-8")
            )

        self.assertEqual(len(windows), 3)
        self.assertEqual(len(predictions), 12)
        self.assertFalse(predictions.duplicated(["date", "symbol"]).any())
        self.assertEqual(len(signals), 6)
        self.assertEqual(signals.groupby("date").size().tolist(), [1] * 6)
        self.assertTrue(np.isfinite(predictions["score"]).all())
        self.assertEqual(summary["window_count"], 3)
        self.assertEqual(summary["purge_days"], 1)
        self.assertEqual(summary["prediction_count"], 12)
        self.assertEqual(resumed_summary["reused_window_count"], 3)

    def test_validate_feature_label_frame_rejects_duplicates(self):
        frame = pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-01"],
                "symbol": ["000001", "000001"],
                "feature": [1.0, 2.0],
                "target": [0.1, 0.2],
            }
        )

        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_feature_label_frame(frame, feature_cols=("feature",), target_col="target")

    def test_walk_forward_windows_are_ordered(self):
        windows = build_walk_forward_windows(
            dates=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]),
            train_size=2,
            test_size=1,
        )

        self.assertEqual(
            windows[0],
            WalkForwardWindow(
                train_start=pd.Timestamp("2026-01-01"),
                train_end=pd.Timestamp("2026-01-02"),
                test_start=pd.Timestamp("2026-01-03"),
                test_end=pd.Timestamp("2026-01-03"),
            ),
        )

    def test_scores_to_signals_keeps_six_digit_symbols(self):
        scores = pd.DataFrame({"date": ["2026-01-02"], "symbol": ["1"], "score": [0.8]})
        signals = scores_to_signals(scores, source="ridge")

        self.assertEqual(signals.loc[0, "symbol"], "000001")
        self.assertEqual(signals.loc[0, "source"], "model_ridge")
        self.assertIsInstance(signals.loc[0, "metadata"], dict)

    def test_scores_to_signals_selects_daily_top_n_with_equal_weights(self):
        scores = pd.DataFrame(
            {
                "date": ["2026-01-02"] * 3,
                "symbol": [1, 2, 3],
                "score": [0.1, 0.3, 0.2],
            }
        )

        signals = scores_to_signals(scores, source="ridge", top_n=2)

        self.assertEqual(signals["symbol"].tolist(), ["000002", "000003"])
        self.assertEqual(signals["weight"].tolist(), [0.5, 0.5])
        self.assertEqual(signals.loc[0, "metadata"]["rank_position"], 1)


if __name__ == "__main__":
    unittest.main()
