import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reports.factor_tester import build_long_factor_frame_from_raw
from factors.alpha101 import (
    ALPHA101_NAMES,
    Alpha101,
    Alpha101DataError,
    Alpha101Panels,
    alpha101_to_long,
    build_alpha101_panels,
    decay_linear,
    indneutralize,
    normalize_alpha_name,
    rank,
    ts_argmax,
)
from factors.catalog import load_factor_catalog
from factors.operators import correlation, delta, safe_div, ts_rank


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
        turnover_value=close * volume * 100.0,
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

    def test_adv_uses_traded_value_not_share_volume(self):
        actual = self.panels.adv(20)
        expected = self.panels.turnover_value.rolling(20, min_periods=20).sum() / 20
        pd.testing.assert_frame_equal(actual, expected)
        self.assertFalse(
            actual.equals(self.panels.volume.rolling(20, min_periods=20).mean())
        )

    def test_missing_traded_value_is_explicit_for_adv_factor(self):
        panels = Alpha101Panels(**{**self.panels.__dict__, "turnover_value": None})
        with self.assertRaisesRegex(Alpha101DataError, "traded value"):
            Alpha101(panels).calculate("alpha_007")

        partial = self.panels.turnover_value.copy()
        partial.iloc[0, 0] = np.nan
        panels = Alpha101Panels(**{**self.panels.__dict__, "turnover_value": partial})
        with self.assertRaisesRegex(Alpha101DataError, "missing 1 observed market rows"):
            Alpha101(panels).calculate("alpha_007")

    def test_boolean_formulas_keep_zero_one_source_values(self):
        positive = self.calculator.calculate("alpha_061").stack().dropna()
        negative = self.calculator.calculate("alpha_062").stack().dropna()
        self.assertTrue(set(positive.unique()).issubset({0.0, 1.0}))
        self.assertTrue(set(negative.unique()).issubset({-1.0, 0.0}))
        self.assertNotIn(-1.0, positive.unique())
        self.assertNotIn(1.0, negative.unique())

    def test_conditional_factors_mask_incomplete_warmup_windows(self):
        leading_missing = {
            "alpha_009": 5,
            "alpha_010": 4,
            "alpha_023": 19,
            "alpha_024": 199,
            "alpha_049": 20,
            "alpha_051": 20,
        }
        for factor, count in leading_missing.items():
            with self.subTest(factor=factor):
                values = self.calculator.calculate(factor).iloc[:, 0]
                self.assertTrue(values.iloc[:count].isna().all())
                self.assertTrue(values.iloc[count:].notna().any())

    def test_vwap_factor_rejects_missing_true_vwap(self):
        panels = Alpha101Panels(
            **{
                **self.panels.__dict__,
                "vwap": self.panels.close * np.nan,
            }
        )
        with self.assertRaisesRegex(Alpha101DataError, "VWAP-dependent"):
            Alpha101(panels).calculate("alpha_011")

        partial = self.panels.vwap.copy()
        partial.iloc[0, 0] = np.nan
        panels = Alpha101Panels(**{**self.panels.__dict__, "vwap": partial})
        with self.assertRaisesRegex(Alpha101DataError, "missing 1 observed market rows"):
            Alpha101(panels).calculate("alpha_011")

    def test_algebraic_simplifications_preserve_alpha101_outputs(self):
        calculator = self.calculator
        expected_059 = -ts_rank(
            decay_linear(correlation(indneutralize(self.panels.vwap, self.panels.industry, "industry"), self.panels.volume, 4.25197), 16.2289),
            8.19648,
        )
        expected_066 = -(
            rank(decay_linear(delta(self.panels.vwap, 3.51013), 7.23052))
            + ts_rank(
                decay_linear(
                    safe_div(
                        self.panels.low - self.panels.vwap,
                        self.panels.open - (self.panels.high + self.panels.low) / 2,
                    ),
                    11.4157,
                ),
                6.72611,
            )
        )
        pd.testing.assert_frame_equal(calculator.calculate("alpha_059"), expected_059)
        pd.testing.assert_frame_equal(calculator.calculate("alpha_066"), expected_066)


class Alpha101EconomicLifecycleTest(unittest.TestCase):
    def test_default_catalog_keeps_unspecified_factors_on_watch(self):
        path = Path(__file__).resolve().parents[1] / "config" / "factors.yaml"
        catalog = load_factor_catalog(path)

        self.assertEqual(catalog.default_status, "watch")
        self.assertEqual(catalog.status_for("alpha_040"), "watch")
        self.assertEqual(catalog.status_for("alpha_059"), "watch")
        self.assertEqual(catalog.status_for("alpha_077"), "watch")
        self.assertEqual(
            catalog.select(("alpha_040", "alpha_059", "alpha_077", "alpha_101")),
            ("alpha_040", "alpha_059", "alpha_077", "alpha_101"),
        )


class Alpha101RawAdapterTest(unittest.TestCase):
    def test_raw_adapter_builds_qfq_vwap_and_dated_classification(self):
        dates = pd.date_range("2025-01-01", periods=3, freq="B")
        volume = np.array([1000.0, 1100.0, 1200.0])
        unadjusted_vwap = np.array([10.0, 10.2, 11.0])
        raw = {
            "000001": pd.DataFrame(
                {
                    "date": dates,
                    "open": [4.9, 10.1, 10.9],
                    "close": [5.1, 10.3, 11.1],
                    "high": [5.2, 10.4, 11.2],
                    "low": [4.8, 10.0, 10.8],
                    "volume": volume,
                    "amount": unadjusted_vwap * volume / 10.0,
                    "adj_factor": [1.0, 2.0, 2.0],
                }
            )
        }
        metadata = pd.DataFrame(
            {
                "date": dates,
                "symbol": ["000001"] * 3,
                "industry": ["old", "old", "new"],
            }
        )

        panels = build_alpha101_panels(raw, metadata=metadata)

        expected_vwap = pd.DataFrame(
            {"000001": [5.0, 10.2, 11.0]},
            index=dates,
        )
        pd.testing.assert_frame_equal(panels.vwap, expected_vwap, check_freq=False)
        self.assertEqual(panels.industry["000001"].tolist(), ["old", "old", "new"])
        self.assertEqual(
            panels.turnover_value["000001"].tolist(),
            (raw["000001"]["amount"] * 1000.0).tolist(),
        )

    def test_static_metadata_is_not_backfilled_across_history(self):
        dates = pd.date_range("2025-01-01", periods=2, freq="B")
        raw = {
            "000001": pd.DataFrame(
                {
                    "date": dates,
                    "open": [10.0, 10.1],
                    "close": [10.1, 10.2],
                    "high": [10.2, 10.3],
                    "low": [9.9, 10.0],
                    "volume": [1000.0, 1100.0],
                    "vwap": [10.05, 10.15],
                }
            )
        }
        static = pd.DataFrame({"symbol": ["000001"], "industry": ["current"]})

        panels = build_alpha101_panels(raw, metadata=static)

        self.assertIsNone(panels.industry)
        with self.assertRaisesRegex(Alpha101DataError, "industry"):
            Alpha101(panels).calculate("alpha_063")

    def test_industry_is_not_aliased_to_sector_or_subindustry(self):
        dates = pd.date_range("2025-01-01", periods=2, freq="B")
        raw = {
            "000001": pd.DataFrame(
                {
                    "date": dates,
                    "open": [10.0, 10.1],
                    "close": [10.1, 10.2],
                    "high": [10.2, 10.3],
                    "low": [9.9, 10.0],
                    "volume": [1000.0, 1100.0],
                    "vwap": [10.05, 10.15],
                    "industry": ["bank", "bank"],
                }
            )
        }

        panels = build_alpha101_panels(raw)

        self.assertEqual(panels.industry["000001"].tolist(), ["bank", "bank"])
        self.assertIsNone(panels.sector)
        self.assertIsNone(panels.subindustry)
        with self.assertRaisesRegex(Alpha101DataError, "sector"):
            Alpha101(panels).calculate("alpha_058")

    def test_explicit_vwap_on_an_incompatible_price_basis_is_rejected(self):
        dates = pd.date_range("2025-01-01", periods=2, freq="B")
        raw = {
            "000001": pd.DataFrame(
                {
                    "date": dates,
                    "open": [10.0, 10.1],
                    "close": [10.1, 10.2],
                    "high": [10.2, 10.3],
                    "low": [9.9, 10.0],
                    "volume": [1000.0, 1100.0],
                    "vwap": [20.0, 20.1],
                }
            )
        }

        panels = build_alpha101_panels(raw)

        self.assertTrue(panels.vwap.isna().all().all())
        with self.assertRaisesRegex(Alpha101DataError, "VWAP-dependent"):
            Alpha101(panels).calculate("alpha_011")

    def test_missing_vwap_is_not_filled_with_typical_price(self):
        dates = pd.date_range("2025-01-01", periods=3, freq="B")
        raw = {
            "000001": pd.DataFrame(
                {
                    "date": dates,
                    "open": [10.0, 10.1, 10.2],
                    "close": [10.1, 10.2, 10.3],
                    "high": [10.2, 10.3, 10.4],
                    "low": [9.9, 10.0, 10.1],
                    "volume": [1000.0, 1100.0, 1200.0],
                }
            )
        }
        panels = build_alpha101_panels(raw)
        self.assertTrue(panels.vwap.isna().all().all())
        with self.assertRaisesRegex(Alpha101DataError, "VWAP-dependent"):
            Alpha101(panels).calculate("alpha_011")

    def test_long_adapter_preserves_non_proxy_market_fields(self):
        raw = {}
        dates = pd.date_range("2025-01-01", periods=20, freq="B")
        for number in (1, 2):
            volume = np.arange(20) + 1000 + number
            vwap = np.arange(20) + 10.25 + number
            raw[f"{number:06d}"] = pd.DataFrame(
                {
                    "date": dates,
                    "open": np.arange(20) + 10 + number,
                    "close": np.arange(20) + 10.5 + number,
                    "high": np.arange(20) + 11 + number,
                    "low": np.arange(20) + 9 + number,
                    "volume": volume,
                    "vwap": vwap,
                    "amount": vwap * volume / 10.0,
                    "adj_factor": np.ones(20),
                }
            )
        metadata = pd.DataFrame({"symbol": [1, 2], "industry": ["a", "b"]})

        panels = build_alpha101_panels(raw, metadata=metadata)
        self.assertIsNone(panels.industry)

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
                "turnover_value",
            ],
        )
        self.assertEqual(len(long), 40)
        self.assertEqual(long["listing_age_days"].max(), 20)
        expected_turnover = {
            (date, f"{number:06d}"): amount * 1000.0
            for number in (1, 2)
            for date, amount in zip(dates, raw[f"{number:06d}"]["amount"])
        }
        actual_turnover = long.set_index(["date", "symbol"])["turnover_value"]
        for key, expected in expected_turnover.items():
            self.assertAlmostEqual(actual_turnover.loc[key], expected)

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
