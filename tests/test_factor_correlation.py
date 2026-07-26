import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factors.correlation import (
    FactorCorrelationConfig,
    calculate_factor_correlations,
    calculate_gtja_factor_correlations,
    write_factor_correlation_reports,
)
from factors.alpha101 import Alpha101Panels
from factors.gtja191 import GTJA191ExternalData, GTJA191Panels


def _panels() -> Alpha101Panels:
    dates = pd.date_range("2025-01-01", periods=3, freq="B")
    symbols = ["000001", "000002", "000003", "000004"]
    close = pd.DataFrame(10.0, index=dates, columns=symbols)
    return Alpha101Panels(
        open=close,
        close=close,
        high=close,
        low=close,
        volume=close,
        vwap=close,
        returns=close.pct_change(fill_method=None),
    )


class FactorCorrelationTest(unittest.TestCase):
    def test_gtja191_uses_same_lagged_cross_sectional_contract(self):
        base = _panels()
        panels = GTJA191Panels(
            open=base.open,
            close=base.close,
            high=base.high,
            low=base.low,
            volume=base.volume,
            amount=base.close * base.volume,
            vwap=base.vwap,
            returns=base.returns,
            external=GTJA191ExternalData(),
        )
        factor_a = pd.DataFrame(
            np.tile(np.arange(4, dtype=float), (3, 1)),
            index=base.close.index,
            columns=base.close.columns,
        )
        factor_b = factor_a * -1
        with patch(
            "factors.correlation.GTJA191.calculate",
            side_effect=lambda name: {
                "gtja_001": factor_a,
                "gtja_002": factor_b,
            }[name],
        ):
            result = calculate_gtja_factor_correlations(
                panels,
                (1, "gtja_002"),
                config=FactorCorrelationConfig(
                    factor_lag_days=1,
                    min_observations=3,
                    min_dates=2,
                    high_correlation_threshold=0.8,
                ),
                factor_statuses={"gtja_001": "watch", "gtja_002": "watch"},
            )

        self.assertAlmostEqual(result.spearman.at["gtja_001", "gtja_002"], -1.0)
        self.assertTrue(bool(result.pairs.loc[0, "high_correlation"]))
        self.assertEqual(result.status["factor_status"].tolist(), ["watch", "watch"])

    def test_daily_cross_sectional_correlations_are_averaged_after_lag(self):
        panels = _panels()
        factor_a = pd.DataFrame(
            [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
            index=panels.close.index,
            columns=panels.close.columns,
            dtype=float,
        )
        factor_b = pd.DataFrame(
            [[1, 2, 3, 4], [4, 3, 2, 1], [1, 2, 3, 4]],
            index=panels.close.index,
            columns=panels.close.columns,
            dtype=float,
        )
        values = {"alpha_001": factor_a, "alpha_002": factor_b}

        with patch(
            "factors.correlation.Alpha101.calculate",
            side_effect=lambda name: values[name],
        ):
            result = calculate_factor_correlations(
                panels,
                ("alpha_001", "alpha_002"),
                config=FactorCorrelationConfig(
                    factor_lag_days=1,
                    min_observations=3,
                    min_dates=2,
                    high_correlation_threshold=0.8,
                ),
            )

        self.assertAlmostEqual(result.spearman.at["alpha_001", "alpha_002"], 0.0)
        self.assertAlmostEqual(result.pearson.at["alpha_001", "alpha_002"], 0.0)
        self.assertEqual(result.valid_dates.at["alpha_001", "alpha_002"], 2)
        self.assertFalse(bool(result.pairs.loc[0, "high_correlation"]))
        self.assertEqual(result.successful_factors, ("alpha_001", "alpha_002"))

    def test_failure_is_isolated_and_reports_are_written(self):
        panels = _panels()
        base = pd.DataFrame(
            np.tile(np.arange(4, dtype=float), (3, 1)),
            index=panels.close.index,
            columns=panels.close.columns,
        )

        def calculate(name):
            if name == "alpha_003":
                raise ValueError("missing classification")
            return base if name == "alpha_001" else base * 2

        config = FactorCorrelationConfig(
            factor_lag_days=0,
            min_observations=3,
            min_dates=2,
            high_correlation_threshold=0.8,
        )
        with patch("factors.correlation.Alpha101.calculate", side_effect=calculate):
            result = calculate_factor_correlations(
                panels,
                ("alpha_001", "alpha_002", "alpha_003"),
                config=config,
                priority_scores={"alpha_001": 0.4, "alpha_002": 1.2},
            )

        self.assertEqual(result.failed_factors, ("alpha_003",))
        self.assertEqual(result.spearman.shape, (2, 2))
        self.assertAlmostEqual(result.spearman.at["alpha_001", "alpha_002"], 1.0)
        self.assertTrue(bool(result.pairs.loc[0, "high_correlation"]))
        self.assertEqual(result.deduplicated_factors["factor"].tolist(), ["alpha_002"])
        removed = result.deduplication.loc[result.deduplication["factor"].eq("alpha_001")].iloc[0]
        self.assertFalse(bool(removed["kept"]))
        self.assertEqual(removed["representative"], "alpha_002")

        with tempfile.TemporaryDirectory() as temp_dir:
            output = write_factor_correlation_reports(
                result,
                temp_dir,
                config=config,
                eligible_factors=("alpha_002",),
                eligibility_settings={
                    "column": "preferred_profitable_after_cost",
                    "window": 20,
                },
            )
            for filename in (
                "spearman_matrix.csv",
                "pearson_matrix.csv",
                "valid_date_count_matrix.csv",
                "correlation_pairs.csv",
                "deduplication.csv",
                "deduplicated_factors.csv",
                "factor_status.csv",
                "ml_candidate_factors.csv",
                "spearman_heatmap.html",
                "manifest.json",
            ):
                self.assertTrue((output / filename).exists(), filename)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["failed_factors"], ["alpha_003"])
            self.assertEqual(manifest["method"], "mean_daily_cross_sectional_correlation")
            self.assertEqual(manifest["ml_candidate_factor_count"], 1)
            candidates = pd.read_csv(output / "ml_candidate_factors.csv")
            self.assertEqual(candidates["factor"].tolist(), ["alpha_002"])

    def test_minimum_valid_dates_masks_unstable_pair(self):
        panels = _panels()
        base = pd.DataFrame(
            np.tile(np.arange(4, dtype=float), (3, 1)),
            index=panels.close.index,
            columns=panels.close.columns,
        )
        with patch("factors.correlation.Alpha101.calculate", return_value=base):
            result = calculate_factor_correlations(
                panels,
                ("alpha_001", "alpha_002"),
                config=FactorCorrelationConfig(
                    factor_lag_days=1,
                    min_observations=3,
                    min_dates=3,
                ),
            )
        self.assertTrue(pd.isna(result.spearman.at["alpha_001", "alpha_002"]))
        self.assertEqual(result.valid_dates.at["alpha_001", "alpha_002"], 2)


if __name__ == "__main__":
    unittest.main()
