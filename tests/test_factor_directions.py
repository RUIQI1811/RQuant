import json
import argparse
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from factors.correlation import (
    FactorCorrelationConfig,
    calculate_gtja_factor_correlations,
)
from factors.directions import load_gtja_factor_directions
from factors.gtja191 import GTJA191, GTJA191ExternalData, GTJA191_NAMES
from reports.gtja191_batch import GTJA191BatchRunner
from tests.test_gtja191 import _complete_panels
from training.build_dataset import (
    MLDatasetConfig,
    add_arguments as add_ml_dataset_arguments,
    build_ml_dataset,
)


class FactorDirectionConfigTest(unittest.TestCase):
    def test_loads_explicit_and_default_directions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "factors.yaml"
            path.write_text(
                "default_direction: 1\n"
                "factors:\n"
                "  gtja_002: {status: watch, direction: -1}\n"
                "directions:\n"
                "  gtja_001: -1\n",
                encoding="utf-8",
            )

            directions = load_gtja_factor_directions(
                path,
                ("gtja_001", "gtja_002", "gtja_003"),
            )

        self.assertEqual(
            directions,
            {"gtja_001": -1, "gtja_002": -1, "gtja_003": 1},
        )

    def test_current_gtja_snapshot_contains_104_inversions(self):
        path = Path(__file__).resolve().parents[1] / "config" / "gtja191_factors.yaml"
        directions = load_gtja_factor_directions(path, GTJA191_NAMES)

        inverted = {name for name, direction in directions.items() if direction == -1}

        self.assertEqual(len(inverted), 104)
        self.assertTrue({"gtja_070", "gtja_095", "gtja_150", "gtja_159"}.issubset(inverted))
        self.assertTrue({"gtja_005", "gtja_030", "gtja_149", "gtja_182"}.isdisjoint(inverted))


class FactorDirectionApplicationTest(unittest.TestCase):
    def test_ml_dataset_cli_exposes_direction_config(self):
        parser = argparse.ArgumentParser()
        add_ml_dataset_arguments(parser)
        args = parser.parse_args(
            [
                "--factors",
                "gtja_070",
                "--factor-config",
                "config/gtja191_factors.yaml",
                "--output",
                "dataset",
            ]
        )

        self.assertEqual(args.factor_config, "config/gtja191_factors.yaml")

    def test_gtja_batch_calculator_applies_direction_after_formula(self):
        panels = replace(_complete_panels(days=90), external=GTJA191ExternalData())
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = GTJA191BatchRunner(
                panels,
                factors=("gtja_001",),
                output_dir=temp_dir,
                factor_directions={"gtja_001": -1},
            )

            actual = runner.calculator.calculate("gtja_001")
            expected = GTJA191(panels).calculate("gtja_001") * -1

        pd.testing.assert_frame_equal(actual, expected)

    def test_gtja_batch_direction_changes_resume_fingerprint(self):
        panels = replace(_complete_panels(days=90), external=GTJA191ExternalData())
        with tempfile.TemporaryDirectory() as temp_dir:
            positive = GTJA191BatchRunner(
                panels,
                factors=("gtja_001",),
                output_dir=temp_dir,
                factor_directions={"gtja_001": 1},
            )
            inverted = GTJA191BatchRunner(
                panels,
                factors=("gtja_001",),
                output_dir=temp_dir,
                factor_directions={"gtja_001": -1},
            )

        self.assertNotEqual(positive.fingerprint, inverted.fingerprint)

    def test_gtja_correlation_applies_direction(self):
        panels = replace(_complete_panels(days=90), external=GTJA191ExternalData())
        config = FactorCorrelationConfig(
            factor_lag_days=1,
            min_observations=2,
            min_dates=2,
        )
        original = calculate_gtja_factor_correlations(
            panels,
            ("gtja_001", "gtja_002"),
            config=config,
        )
        directed = calculate_gtja_factor_correlations(
            panels,
            ("gtja_001", "gtja_002"),
            config=config,
            factor_directions={"gtja_001": -1, "gtja_002": 1},
        )

        self.assertAlmostEqual(
            directed.spearman.loc["gtja_001", "gtja_002"],
            -original.spearman.loc["gtja_001", "gtja_002"],
        )

    def test_ml_dataset_applies_direction_before_lag_and_transform(self):
        dates = pd.bdate_range("2025-01-02", periods=4)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "raw"
            data_dir.mkdir()
            factor_path = root / "external.csv"
            for symbol_index, symbol in enumerate(("000001", "000002"), start=1):
                close = 10.0 + symbol_index + np.arange(len(dates)) * 0.1
                pd.DataFrame(
                    {
                        "date": dates,
                        "open": close - 0.05,
                        "close": close,
                        "high": close + 0.1,
                        "low": close - 0.1,
                        "volume": 1_000_000,
                    }
                ).to_csv(data_dir / f"{symbol}.csv", index=False)
            pd.DataFrame(
                [
                    {"date": date, "symbol": symbol, "external_a": value}
                    for date in dates
                    for symbol, value in (("000001", 1.0), ("000002", 2.0))
                ]
            ).to_csv(factor_path, index=False)

            outputs = build_ml_dataset(
                data_dir=data_dir,
                output_dir=root / "dataset",
                factor_file=factor_path,
                factor_directions={"external_a": -1},
                config=MLDatasetConfig(
                    factors=("external_a",),
                    target_windows=(1,),
                    feature_transform="rank",
                    target_transform="rank",
                ),
            )
            features = pd.read_csv(outputs["features_path"], dtype={"symbol": str})
            manifest = json.loads(outputs["manifest_path"].read_text(encoding="utf-8"))

        ranked = features.loc[features["date"].eq(str(dates[1].date()))].set_index("symbol")
        self.assertEqual(ranked["external_a"].to_dict(), {"000001": 1.0, "000002": 0.5})
        self.assertEqual(manifest["factor_directions"], {"external_a": -1})


if __name__ == "__main__":
    unittest.main()
