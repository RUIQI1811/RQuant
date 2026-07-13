import argparse
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.build_dataset import MLDatasetConfig, build_ml_dataset
from training.multifactor import (
    MultifactorFitConfig,
    add_arguments as add_multifactor_arguments,
    config_from_args,
    load_lifecycle_factors,
    run_multifactor_fit,
)
from training.train_walk_forward import WalkForwardTrainingConfig, run_walk_forward_training
from backtest.signal_portfolio import run_signal_portfolio_backtest
from domain.artifacts import WorkflowResult
from domain.execution import BacktestResult
from domain.research import MLDatasetResult, ModelFitResult, MultifactorComparisonResult


class MLDatasetTest(unittest.TestCase):
    def test_multifactor_imports_active_lifecycle_factors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "gtja191_factors.yaml"
            config_path.write_text(
                "default_status: disabled\n"
                "factors:\n"
                "  gtja_042: {status: active, useful_horizons: [10, 20]}\n"
                "  gtja_062: {status: watch, useful_horizons: [20]}\n",
                encoding="utf-8",
            )
            self.assertEqual(
                load_lifecycle_factors(config_path),
                ("gtja_042",),
            )
            self.assertEqual(
                load_lifecycle_factors(config_path, statuses=("active", "watch")),
                ("gtja_042", "gtja_062"),
            )
            parser = argparse.ArgumentParser()
            add_multifactor_arguments(parser)
            args = parser.parse_args(
                [
                    "--factors",
                    "alpha_040",
                    "--factor-config",
                    str(config_path),
                    "--lifecycle-statuses",
                    "active",
                ]
            )
            self.assertEqual(
                config_from_args(args).factors,
                ("alpha_040", "gtja_042"),
            )
            self.assertFalse(hasattr(args, "no_progress"))
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(["--factors", "alpha_040", "--no-progress"])

    def test_dataset_to_training_to_signal_backtest_is_reproducible(self):
        dates = pd.bdate_range("2025-01-02", periods=40)
        day = np.arange(len(dates), dtype=float)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "raw"
            dataset_dir = root / "dataset"
            training_dir = root / "training"
            multifactor_dir = root / "multifactor"
            backtest_dir = root / "backtest"
            data_dir.mkdir()
            for index, symbol in enumerate(("000001", "000002", "000003", "000004")):
                close = (
                    10.0
                    + index * 2.0
                    + day * (0.02 + index * 0.002)
                    + np.sin(day / (3.0 + index)) * 0.1
                )
                open_price = close * (1.0 + 0.001 * np.cos(day / (2.0 + index)))
                pd.DataFrame(
                    {
                        "date": dates,
                        "open": open_price,
                        "close": close,
                        "high": np.maximum(open_price, close) + 0.1,
                        "low": np.minimum(open_price, close) - 0.1,
                        "volume": 1_000_000 + index * 100_000 + day * 1000,
                    }
                ).to_csv(data_dir / f"{symbol}.csv", index=False)

            dataset_outputs = build_ml_dataset(
                data_dir=data_dir,
                output_dir=dataset_dir,
                config=MLDatasetConfig(
                    factors=("alpha_101", "custom_002"),
                    target_windows=(1,),
                    factor_lag_days=1,
                    feature_transform="rank",
                    target_transform="rank",
                    start_date=str(dates[2].date()),
                    end_date=str(dates[-3].date()),
                ),
            )
            features = pd.read_csv(
                dataset_outputs["features_path"],
                dtype={"symbol": str},
            )
            labels = pd.read_csv(
                dataset_outputs["labels_path"],
                dtype={"symbol": str},
            )
            manifest = json.loads(
                Path(dataset_outputs["manifest_path"]).read_text(encoding="utf-8")
            )

            training_outputs = run_walk_forward_training(
                features_path=dataset_outputs["features_path"],
                labels_path=dataset_outputs["labels_path"],
                output_dir=training_dir,
                config=WalkForwardTrainingConfig(
                    feature_cols=("alpha_101", "custom_002"),
                    target_col="next_open_return_1d_cs_rank",
                    model="ridge",
                    train_size=12,
                    test_size=4,
                    purge_days=2,
                    signal_top_n=2,
                    ridge_alpha=0.1,
                ),
            )
            backtest_outputs = run_signal_portfolio_backtest(
                signals_path=training_outputs["signals_path"],
                data_dir=data_dir,
                output_dir=backtest_dir,
                source="model_ridge",
                hold_days=3,
                initial_cash=500_000,
                max_positions=2,
                lot_size=100,
            )
            summary_exists = (backtest_dir / "portfolio_summary.json").exists()
            with self.assertLogs("training", level="INFO") as captured_logs:
                multifactor_outputs = run_multifactor_fit(
                    data_dir=data_dir,
                    output_dir=multifactor_dir,
                    config=MultifactorFitConfig(
                        factors=("alpha_101", "custom_002"),
                        models=("ridge",),
                        target_window=1,
                        train_size=12,
                        test_size=4,
                        purge_days=2,
                        signal_top_n=2,
                        start_date=str(dates[2].date()),
                        end_date=str(dates[-3].date()),
                    ),
                )
            leaderboard = pd.read_csv(multifactor_outputs["leaderboard_path"])
            multifactor_manifest = json.loads(
                multifactor_outputs["manifest_path"].read_text(encoding="utf-8")
            )
            multifactor_signals_exists = (
                multifactor_dir / "models/ridge/signals.csv"
            ).exists()

        self.assertEqual(features["symbol"].str.len().unique().tolist(), [6])
        self.assertEqual(
            features.columns.tolist(),
            ["date", "symbol", "alpha_101", "custom_002"],
        )
        self.assertIn("next_open_return_1d", labels.columns)
        self.assertIn("next_open_return_1d_cs_rank", labels.columns)
        self.assertTrue(features[["alpha_101", "custom_002"]].max().max() <= 1.0)
        self.assertTrue(features[["alpha_101", "custom_002"]].min().min() > 0.0)
        self.assertEqual(manifest["config"]["factor_lag_days"], 1)
        self.assertEqual(manifest["config"]["factors"], ["alpha_101", "custom_002"])
        self.assertEqual(
            manifest["target_columns"]["fitted"],
            ["next_open_return_1d_cs_rank"],
        )
        self.assertGreater(backtest_outputs["result"].summary["signal_count"], 0)
        self.assertIsInstance(dataset_outputs, WorkflowResult)
        self.assertIsInstance(dataset_outputs.result, MLDatasetResult)
        self.assertIsInstance(training_outputs.result, ModelFitResult)
        self.assertIsInstance(backtest_outputs.result, BacktestResult)
        self.assertIsInstance(multifactor_outputs.result, MultifactorComparisonResult)
        self.assertTrue(summary_exists)
        self.assertEqual(leaderboard["model"].tolist(), ["ridge"])
        self.assertEqual(multifactor_manifest["target_col"], "next_open_return_1d_cs_rank")
        self.assertTrue(multifactor_signals_exists)
        logs = "\n".join(captured_logs.output)
        self.assertIn("Starting multi-factor fit", logs)
        self.assertIn("Calculating factor alpha_101", logs)
        self.assertIn("Model ridge has", logs)
        self.assertIn("Multi-factor fit complete", logs)


if __name__ == "__main__":
    unittest.main()
