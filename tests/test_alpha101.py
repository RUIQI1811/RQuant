import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.factor_tester import build_long_factor_frame_from_raw
from factors.alpha101 import (
    ALPHA101_NAMES,
    Alpha101,
    Alpha101DataError,
    Alpha101Panels,
    alpha101_to_long,
    build_alpha101_panels,
    decay_linear,
    normalize_alpha_name,
    rank,
    ts_argmax,
)


def _complete_panels(days: int = 320, symbols: int = 6) -> Alpha101Panels:
    dates = pd.date_range("2024-01-01", periods=days, freq="B")
    columns = [f"{number:06d}" for number in range(1, symbols + 1)]
    rng = np.random.default_rng(20260629)
    close = pd.DataFrame(
        20 + rng.normal(0, 0.2, (days, symbols)).cumsum(axis=0),
        index=dates,
        columns=columns,
    )
    open_ = close * (1 + rng.normal(0, 0.01, close.shape))
    high = pd.DataFrame(
        np.maximum(open_, close) * (1 + rng.random(close.shape) * 0.02),
        index=dates,
        columns=columns,
    )
    low = pd.DataFrame(
        np.minimum(open_, close) * (1 - rng.random(close.shape) * 0.02),
        index=dates,
        columns=columns,
    )
    volume = pd.DataFrame(
        rng.lognormal(12, 0.4, close.shape),
        index=dates,
        columns=columns,
    )
    groups = pd.DataFrame(
        [["bank", "bank", "tech", "tech", "energy", "energy"]] * days,
        index=dates,
        columns=columns,
    )
    return Alpha101Panels(
        open=open_,
        close=close,
        high=high,
        low=low,
        volume=volume,
        vwap=(high + low + close) / 3,
        returns=close.pct_change(fill_method=None),
        cap=close * volume,
        sector=groups,
        industry=groups,
        subindustry=groups,
    )


class Alpha101OperatorsTest(unittest.TestCase):
    def test_cross_sectional_rank_is_percentile_rank(self):
        frame = pd.DataFrame([[1.0, 2.0, 3.0]], columns=list("abc"))
        actual = rank(frame)
        self.assertAlmostEqual(actual.loc[0, "a"], 1 / 3)
        self.assertAlmostEqual(actual.loc[0, "c"], 1.0)

    def test_decay_linear_weights_recent_values_more(self):
        frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        self.assertAlmostEqual(decay_linear(frame, 3).iloc[-1, 0], 14 / 6)

    def test_ts_argmax_uses_oldest_day_as_one(self):
        frame = pd.DataFrame({"a": [3.0, 1.0, 2.0]})
        self.assertEqual(ts_argmax(frame, 3).iloc[-1, 0], 1.0)


class Alpha101CalculatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panels = _complete_panels()
        cls.calculator = Alpha101(cls.panels)

    def test_registry_contains_all_101_names(self):
        self.assertEqual(len(ALPHA101_NAMES), 101)
        self.assertEqual(ALPHA101_NAMES[0], "alpha_001")
        self.assertEqual(ALPHA101_NAMES[-1], "alpha_101")
        self.assertTrue(all(callable(getattr(self.calculator, name)) for name in ALPHA101_NAMES))

    def test_all_101_factors_execute_on_complete_data(self):
        for name in ALPHA101_NAMES:
            with self.subTest(name=name):
                result = self.calculator.calculate(name)
                self.assertEqual(result.shape, self.panels.close.shape)
                self.assertEqual(result.index.tolist(), self.panels.close.index.tolist())
                self.assertEqual(result.columns.tolist(), self.panels.close.columns.tolist())

    def test_alpha_101_matches_formula(self):
        actual = self.calculator.calculate("alpha101")
        expected = (self.panels.close - self.panels.open) / (
            self.panels.high - self.panels.low + 0.001
        )
        pd.testing.assert_frame_equal(actual, expected)

    def test_normalize_alpha_name(self):
        self.assertEqual(normalize_alpha_name(1), "alpha_001")
        self.assertEqual(normalize_alpha_name("alpha101"), "alpha_101")
        with self.assertRaises(KeyError):
            normalize_alpha_name(102)

    def test_missing_cap_is_explicit_for_alpha_056(self):
        panels = Alpha101Panels(**{**self.panels.__dict__, "cap": None})
        with self.assertRaisesRegex(Alpha101DataError, "market-cap"):
            Alpha101(panels).calculate("alpha_056")


class Alpha101RawAdapterTest(unittest.TestCase):
    def test_raw_adapter_uses_typical_price_and_metadata(self):
        raw = {}
        dates = pd.date_range("2025-01-01", periods=20, freq="B")
        for number in (1, 2):
            raw[f"{number:06d}"] = pd.DataFrame(
                {
                    "date": dates,
                    "open": np.arange(20) + 10 + number,
                    "close": np.arange(20) + 10.5 + number,
                    "high": np.arange(20) + 11 + number,
                    "low": np.arange(20) + 9 + number,
                    "volume": np.arange(20) + 1000 + number,
                }
            )
        metadata = pd.DataFrame({"symbol": [1, 2], "industry": ["a", "b"]})

        panels = build_alpha101_panels(raw, metadata=metadata)
        expected_vwap = (panels.high + panels.low + panels.close) / 3
        pd.testing.assert_frame_equal(panels.vwap, expected_vwap)
        self.assertEqual(panels.industry.iloc[0].tolist(), ["a", "b"])

        long = alpha101_to_long(raw, "alpha_101", metadata=metadata)
        self.assertEqual(
            list(long.columns),
            [
                "date",
                "symbol",
                "factor_value",
                "close",
                "volume",
                "daily_return",
                "listing_age_days",
                "industry",
                "turnover_value",
            ],
        )
        self.assertEqual(len(long), 40)
        self.assertEqual(long["listing_age_days"].max(), 20)
        self.assertTrue((long["turnover_value"] == long["close"] * long["volume"]).all())

        via_factor_tester = build_long_factor_frame_from_raw(
            raw,
            factor_name="alpha_101",
            metadata=metadata,
        )
        pd.testing.assert_frame_equal(long, via_factor_tester)


class TopLevelAlpha101ImportTests(unittest.TestCase):
    def test_alpha101_registry_imports_from_top_level_package(self):
        import factors.alpha101 as alpha101

        self.assertTrue(hasattr(alpha101, "Alpha101Panels"))


if __name__ == "__main__":
    unittest.main()
