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
    load_factor_selection_file,
    load_lifecycle_factors,
    load_ml_run_config,
    resolve_run_args,
    run_multifactor_fit,
)
from training.train_walk_forward import WalkForwardTrainingConfig, run_walk_forward_training
from backtest.signal_portfolio import run_signal_portfolio_backtest
from domain.artifacts import WorkflowResult
from domain.execution import BacktestResult
from domain.research import MLDatasetResult, ModelFitResult, MultifactorComparisonResult


class MLDatasetTest(unittest.TestCase):
    def test_multifactor_yaml_config_loads_and_cli_overrides(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "ml.yaml"
            config_path.write_text(
                "version: 1\n"
                "inputs:\n"
                "  data: yaml/raw\n"
                "features:\n"
                "  names: [alpha_040, custom_002]\n"
                "training:\n"
                "  models: [ridge, elasticnet]\n"
                "  target_window: 10\n"
                "backtest:\n"
                "  enabled: false\n"
                "execution:\n"
                "  output: yaml/output\n",
                encoding="utf-8",
            )
            parser = argparse.ArgumentParser()
            add_multifactor_arguments(parser)
            args = parser.parse_args(
                ["--config", str(config_path), "--models", "ridge"]
            )
            args._specified_options = frozenset({"--config", "--models"})
            resolved = resolve_run_args(args)
            config = config_from_args(resolved)

            self.assertEqual(load_ml_run_config(config_path)["data"], "yaml/raw")
            self.assertEqual(resolved.data, "yaml/raw")
            self.assertEqual(resolved.output, "yaml/output")
            self.assertEqual(config.factors, ("alpha_040", "custom_002"))
            self.assertEqual(config.models, ("ridge",))
            self.assertEqual(config.target_window, 10)
            self.assertFalse(config.run_backtests)

    def test_multifactor_yaml_config_rejects_unknown_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "ml.yaml"
            config_path.write_text(
                "version: 1\ntraining:\n  modelz: [ridge]\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "unsupported fields"):
                load_ml_run_config(config_path)

    def test_dataset_uses_daily_point_in_time_listing_universe(self):
        dates = pd.bdate_range("2025-01-02", periods=6)
        histories = {
            "000001": dates[:3],
            "000002": dates,
            "000003": dates[3:],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "raw"
            output_dir = root / "dataset"
            factor_path = root / "external_factors.csv"
            data_dir.mkdir()
            for symbol_index, (symbol, symbol_dates) in enumerate(histories.items()):
                day = np.arange(len(symbol_dates), dtype=float)
                close = 10.0 + symbol_index * 2.0 + day * 0.1
                pd.DataFrame(
                    {
                        "date": symbol_dates,
                        "open": close - 0.05,
                        "close": close,
                        "high": close + 0.1,
                        "low": close - 0.1,
                        "volume": 1_000_000 + symbol_index * 100_000,
                    }
                ).to_csv(data_dir / f"{symbol}.csv", index=False)

            pd.DataFrame(
                [
                    {
                        "date": date,
                        "symbol": symbol,
                        "external_a": float(symbol_index + 1),
                    }
                    for date in dates
                    for symbol_index, symbol in enumerate(histories)
                ]
            ).to_csv(factor_path, index=False)

            outputs = build_ml_dataset(
                data_dir=data_dir,
                output_dir=output_dir,
                factor_file=factor_path,
                config=MLDatasetConfig(
                    factors=("external_a",),
                    target_windows=(1,),
                    factor_lag_days=1,
                    feature_transform="rank",
                    target_transform="rank",
                ),
            )
            features = pd.read_csv(outputs["features_path"], dtype={"symbol": str})
            labels = pd.read_csv(outputs["labels_path"], dtype={"symbol": str})
            manifest = json.loads(
                outputs["manifest_path"].read_text(encoding="utf-8")
            )

        symbols_by_date = features.groupby("date")["symbol"].apply(list).to_dict()
        for date in dates[:3]:
            self.assertEqual(
                symbols_by_date[str(date.date())],
                ["000001", "000002"],
            )
        for date in dates[3:]:
            self.assertEqual(
                symbols_by_date[str(date.date())],
                ["000002", "000003"],
            )

        first_new_row = features[
            (features["date"] == str(dates[3].date()))
            & (features["symbol"] == "000003")
        ].iloc[0]
        self.assertTrue(pd.isna(first_new_row["external_a"]))
        ranked_after_listing = features[
            features["date"] == str(dates[4].date())
        ].set_index("symbol")["external_a"]
        self.assertEqual(
            ranked_after_listing.to_dict(),
            {"000002": 0.5, "000003": 1.0},
        )

        delisting_tail = labels[
            (labels["date"] == str(dates[1].date()))
            & (labels["symbol"] == "000001")
        ].iloc[0]
        self.assertTrue(pd.isna(delisting_tail["next_open_return_1d"]))
        self.assertTrue(pd.isna(delisting_tail["next_open_return_1d_cs_rank"]))
        self.assertEqual(len(features), 12)
        self.assertEqual(len(labels), 12)
        self.assertEqual(
            manifest["point_in_time_universe"]["method"],
            "finite_close_observation_on_exact_date",
        )
        self.assertTrue(
            manifest["point_in_time_universe"]["factor_values_masked_before_lag"]
        )
        self.assertEqual(
            manifest["point_in_time_universe"]["yearly_counts"],
            [
                {
                    "year": 2025,
                    "trading_dates": 6,
                    "eligible_rows": 12,
                    "min_symbols": 2,
                    "median_symbols": 2.0,
                    "max_symbols": 2,
                }
            ],
        )

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
            selection_path = Path(temp_dir) / "deduplicated_factors.csv"
            pd.DataFrame({"factor": ["external_a", "alpha_040"]}).to_csv(
                selection_path,
                index=False,
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
                    "--factor-selection-file",
                    str(selection_path),
                ]
            )
            self.assertEqual(
                config_from_args(args).factors,
                ("alpha_040", "external_a", "gtja_042"),
            )
            self.assertEqual(
                load_factor_selection_file(selection_path),
                ("external_a", "alpha_040"),
            )
            self.assertEqual(config_from_args(args).mlp_epochs, 10)
            self.assertEqual(config_from_args(args).elasticnet_alpha, 0.001)
            self.assertEqual(config_from_args(args).elasticnet_l1_ratio, 0.5)
            self.assertTrue(config_from_args(args).run_backtests)
            skip_args = parser.parse_args(
                ["--factors", "alpha_040", "--skip-backtests"]
            )
            self.assertFalse(config_from_args(skip_args).run_backtests)
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
                vwap = (open_price + close) / 2.0
                pd.DataFrame(
                    {
                        "date": dates,
                        "open": open_price,
                        "close": close,
                        "high": np.maximum(open_price, close) + 0.1,
                        "low": np.minimum(open_price, close) - 0.1,
                        "vwap": vwap,
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
                        run_backtests=True,
                        backtest_initial_cash=500_000,
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
            gross_summary_exists = (
                multifactor_dir / "backtests/ridge/gross/portfolio_summary.json"
            ).exists()
            net_summary_exists = (
                multifactor_dir / "backtests/ridge/net/portfolio_summary.json"
            ).exists()
            profitable_models_exists = (
                multifactor_dir / "profitable_models.csv"
            ).exists()
            returns_summary = pd.read_csv(
                multifactor_outputs["returns_summary_path"]
            )
            yearly_returns = pd.read_csv(
                multifactor_outputs["yearly_returns_path"]
            )
            net_equity_curve_exists = multifactor_outputs[
                "net_equity_curve_html_path"
            ].exists()

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
        self.assertTrue(gross_summary_exists)
        self.assertTrue(net_summary_exists)
        self.assertTrue(profitable_models_exists)
        self.assertTrue(net_equity_curve_exists)
        self.assertEqual(
            returns_summary[["model", "scenario"]].values.tolist(),
            [["ridge", "gross"], ["ridge", "net"]],
        )
        self.assertIn("annualized_return", returns_summary.columns)
        self.assertIn("average_yearly_annualized_return", returns_summary.columns)
        self.assertEqual(set(yearly_returns["scenario"]), {"gross", "net"})
        self.assertIn("is_partial_year", yearly_returns.columns)
        self.assertEqual(leaderboard.loc[0, "backtest_status"], "success")
        self.assertIn("net_average_yearly_annualized_return", leaderboard.columns)
        self.assertGreaterEqual(
            leaderboard.loc[0, "gross_total_return"],
            leaderboard.loc[0, "net_total_return"],
        )
        self.assertIn("backtests", multifactor_manifest["models"]["ridge"])
        self.assertEqual(
            multifactor_manifest["profitable_models"], "profitable_models.csv"
        )
        self.assertTrue(
            multifactor_manifest["performance"]["net_equity_curve_html_path"].endswith(
                "net_equity_curve.html"
            )
        )
        logs = "\n".join(captured_logs.output)
        self.assertIn("Starting multi-factor fit", logs)
        self.assertIn("Calculating factor alpha_101", logs)
        self.assertIn("Model ridge has", logs)
        self.assertIn("Multi-factor fit complete", logs)


if __name__ == "__main__":
    unittest.main()
