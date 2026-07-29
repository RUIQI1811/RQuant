import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from factors.correlation import (
    FactorCorrelationConfig,
    calculate_external_factor_correlations,
)
from factors.external import load_external_factor_file
from factors.external import load_research_context_file, merge_context_with_raw_data
from factors.alpha101 import build_alpha101_panels
from reports.alpha101_batch import build_forward_return_frame
from reports.external_factor_batch import run_external_factor_batch
from reports.factor_tester import FactorTesterConfig
from training.build_dataset import MLDatasetConfig, build_ml_dataset
from scripts.test_factor_batch import build_parser as build_factor_batch_parser
from scripts.test_factor_batch import run_from_args as run_factor_batch_from_args


class ExternalFactorTests(unittest.TestCase):
    def test_official_factor_batch_requires_external_classification_on_request(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            factor_path = root / "external.csv"
            pd.DataFrame(
                {
                    "date": ["2026-01-02", "2026-01-02"],
                    "symbol": ["000001", "000002"],
                    "factor_a": [1.0, 2.0],
                    "factor_b": [2.0, 1.0],
                }
            ).to_csv(factor_path, index=False)
            output = root / "batch"
            args = build_factor_batch_parser().parse_args(
                [
                    "--family",
                    "external",
                    "--factor-file",
                    str(factor_path),
                    "--require-classification",
                    "--output",
                    str(output),
                ]
            )

            with self.assertRaisesRegex(ValueError, "missing research categories"):
                run_factor_batch_from_args(args)

            template = output / "factor_classification_template.yaml"
            self.assertTrue(template.exists())
            payload = yaml.safe_load(template.read_text(encoding="utf-8"))
            self.assertEqual(payload["categories"]["factor_a"], "unclassified")
            self.assertEqual(payload["factors"]["factor_b"], "active")

    def test_point_in_time_context_populates_dynamic_cap_and_industry(self):
        dates = pd.bdate_range("2026-01-02", periods=2)
        raw = {
            "000001": pd.DataFrame(
                {
                    "date": dates,
                    "open": [10.0, 10.1],
                    "high": [10.2, 10.3],
                    "low": [9.9, 10.0],
                    "close": [10.1, 10.2],
                    "volume": [1000, 1100],
                }
            )
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            context_path = Path(temp_dir) / "context.csv"
            pd.DataFrame(
                {
                    "trade_date": dates,
                    "code": ["1", "1"],
                    "market_cap": [1_000.0, 1_100.0],
                    "industry": ["old_industry", "new_industry"],
                    "market_regime": ["sideways", "bull"],
                }
            ).to_csv(context_path, index=False)
            context = load_research_context_file(
                context_path,
                date_col="trade_date",
                symbol_col="code",
            )
        panels = build_alpha101_panels(merge_context_with_raw_data(raw, context))

        self.assertEqual(panels.cap.loc[dates[1], "000001"], 1_100.0)
        self.assertEqual(panels.industry.loc[dates[0], "000001"], "old_industry")
        self.assertEqual(panels.industry.loc[dates[1], "000001"], "new_industry")
        self.assertEqual(panels.market_regime.loc[dates[1], "000001"], "bull")
        research = build_forward_return_frame(panels, (1,))
        self.assertEqual(
            research.loc[research["date"].eq(dates[1]), "market_regime"].iloc[0],
            "bull",
        )

    def test_wide_and_long_files_share_the_same_canonical_frame(self):
        rows = [
            {"date": "2026-01-02", "symbol": "1", "factor_a": 1.0, "factor_b": 3.0},
            {"date": "2026-01-02", "symbol": "2", "factor_a": 2.0, "factor_b": 4.0},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            wide_path = root / "wide.csv"
            long_path = root / "long.csv"
            pd.DataFrame(rows).to_csv(wide_path, index=False)
            (
                pd.DataFrame(rows)
                .melt(
                    id_vars=["date", "symbol"],
                    var_name="factor",
                    value_name="factor_value",
                )
                .to_csv(long_path, index=False)
            )
            wide = load_external_factor_file(wide_path)
            long = load_external_factor_file(long_path)

        self.assertEqual(wide.factors, ("factor_a", "factor_b"))
        self.assertEqual(long.factors, wide.factors)
        self.assertEqual(wide.frame["symbol"].tolist(), ["000001", "000002"])
        pd.testing.assert_frame_equal(long.frame, wide.frame)

    def test_external_correlations_lag_values_and_deduplicate_by_quality(self):
        dates = pd.bdate_range("2026-01-02", periods=8)
        rows = []
        for day, date in enumerate(dates):
            for symbol_index, symbol in enumerate(("000001", "000002", "000003", "000004")):
                value = day + symbol_index
                rows.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "factor_a": value,
                        "factor_b": value * 2.0,
                        "factor_c": (-1.0 if symbol_index % 2 else 1.0) * day,
                    }
                )
        result = calculate_external_factor_correlations(
            pd.DataFrame(rows),
            ("factor_a", "factor_b", "factor_c"),
            config=FactorCorrelationConfig(
                factor_lag_days=1,
                min_observations=3,
                min_dates=3,
                high_correlation_threshold=0.8,
            ),
            priority_scores={"factor_a": 0.5, "factor_b": 1.5},
        )

        pair = result.pairs.loc[
            (result.pairs["factor_a"] == "factor_a")
            & (result.pairs["factor_b"] == "factor_b")
        ].iloc[0]
        self.assertAlmostEqual(pair["spearman"], 1.0)
        self.assertTrue(pair["high_correlation"])
        cluster = result.deduplication.loc[
            result.deduplication["factor"].isin(["factor_a", "factor_b"])
        ]
        self.assertEqual(cluster["representative"].unique().tolist(), ["factor_b"])

    def test_external_batch_and_ml_use_the_same_unlagged_source(self):
        dates = pd.bdate_range("2025-01-02", periods=30)
        symbols = tuple(f"{index:06d}" for index in range(1, 7))
        external_rows = []
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            for symbol_index, symbol in enumerate(symbols):
                close = 10.0 + symbol_index + np.arange(len(dates)) * (
                    0.01 + symbol_index * 0.001
                )
                open_price = close * 0.999
                market = pd.DataFrame(
                    {
                        "date": dates,
                        "open": open_price,
                        "high": close + 0.1,
                        "low": open_price - 0.1,
                        "close": close,
                        "volume": 1_000_000 + symbol_index * 10_000,
                        "market_cap": (symbol_index + 1) * 1_000_000_000.0,
                        "industry": "industry_a" if symbol_index < 3 else "industry_b",
                    }
                )
                market.to_csv(raw_dir / f"{symbol}.csv", index=False)
                for day_index, date in enumerate(dates):
                    value = float(day_index * 10 + symbol_index)
                    external_rows.append(
                        {
                            "date": date,
                            "symbol": symbol,
                            "external_a": value,
                            "external_b": value * 2.0,
                        }
                    )
            factor_path = root / "external.csv"
            pd.DataFrame(external_rows).to_csv(factor_path, index=False)
            external = load_external_factor_file(factor_path)
            raw_data = {
                symbol: pd.read_csv(raw_dir / f"{symbol}.csv") for symbol in symbols
            }
            batch = run_external_factor_batch(
                external,
                raw_data,
                output_dir=root / "batch",
                tester_config=FactorTesterConfig(
                    groups=5,
                    top_n_counts=(1,),
                    forward_return_windows=(1,),
                    min_periods=2,
                    min_listing_days=0,
                    market_regime_lookback_days=5,
                    market_regime_min_periods=2,
                    profile="core",
                ),
                factor_categories={
                    "external_a": "price_behavior",
                    "external_b": "price_volume",
                },
                data_signature="synthetic-market-v1",
            )
            resumed_batch = run_external_factor_batch(
                external,
                raw_data,
                output_dir=root / "batch",
                tester_config=FactorTesterConfig(
                    groups=5,
                    top_n_counts=(1,),
                    forward_return_windows=(1,),
                    min_periods=2,
                    min_listing_days=0,
                    market_regime_lookback_days=5,
                    market_regime_min_periods=2,
                    profile="core",
                ),
                factor_categories={
                    "external_a": "price_behavior",
                    "external_b": "price_volume",
                },
                data_signature="synthetic-market-v1",
            )
            dataset = build_ml_dataset(
                data_dir=raw_dir,
                output_dir=root / "dataset",
                config=MLDatasetConfig(
                    factors=("external_a",),
                    target_windows=(1,),
                    factor_lag_days=1,
                ),
                factor_file=factor_path,
            )
            features = pd.read_csv(dataset["features_path"], dtype={"symbol": str})
            manifest = json.loads(dataset["manifest_path"].read_text(encoding="utf-8"))

            self.assertTrue((root / "batch/external_a/market_cap_ic.csv").exists())
            self.assertTrue((root / "batch/external_a/industry_ic.csv").exists())
            self.assertFalse((root / "batch/external_a/market_regime_ic.csv").exists())
            self.assertTrue((root / "batch/external_a/annual_long_only.csv").exists())
            self.assertTrue((root / "batch/external_a/horizon_effectiveness.csv").exists())
            self.assertTrue((root / "batch/long_only_profitability.csv").exists())
            self.assertTrue((root / "batch/profitable_long_only.csv").exists())
            self.assertEqual(set(batch.status["status"]), {"success"})
            self.assertTrue(resumed_batch.status["resumed"].all())
            self.assertTrue(batch.leaderboard["profile"].eq("core").all())
            batch_manifest = json.loads(
                (root / "batch/manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(batch_manifest["profile"], "core")
            self.assertEqual(
                set(batch.leaderboard["factor_category"]),
                {"price_behavior", "price_volume"},
            )
            profitability = pd.read_csv(root / "batch/long_only_profitability.csv")
            self.assertEqual(set(profitability["side"]), {"high_factor", "low_factor"})
            self.assertIn("net_sharpe", profitability.columns)
            self.assertIn("preferred_net_sharpe", batch.leaderboard.columns)
            second_day = features.loc[
                (features["date"] == str(dates[1].date()))
                & (features["symbol"] == "000001"),
                "external_a",
            ].iloc[0]
            self.assertEqual(second_day, 0.0)
            self.assertEqual(manifest["factor_sources"], {"external_a": "external"})
            self.assertEqual(manifest["config"]["factor_lag_days"], 1)


if __name__ == "__main__":
    unittest.main()
