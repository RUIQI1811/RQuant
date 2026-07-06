import sys
import subprocess
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.factors.gtja191 import (  # noqa: E402
    GTJA191,
    GTJA191DataError,
    GTJA191_NAMES,
    GTJA191ExternalData,
    GTJA191Panels,
    build_gtja191_panels,
    count,
    highday,
    lowday,
    normalize_gtja_name,
    gtja191_to_long,
    regbeta,
    regresi,
    sma_cn,
    sumif,
    wma,
)
from pipeline.factors.alpha101 import (  # noqa: E402
    correlation,
    delay,
    delta,
    rank,
    stddev,
)
from pipeline.factor_tester import build_long_factor_frame_from_raw  # noqa: E402


def _complete_panels(days: int = 320, symbols: int = 6) -> GTJA191Panels:
    dates = pd.date_range("2024-01-01", periods=days, freq="B")
    columns = [f"{number:06d}" for number in range(1, symbols + 1)]
    rng = np.random.default_rng(20260706)
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
    returns = close.pct_change(fill_method=None)
    external = GTJA191ExternalData(
        benchmark_open=pd.Series(
            3000 + rng.normal(0, 5, days).cumsum(), index=dates
        ),
        benchmark_close=pd.Series(
            3000 + rng.normal(0, 5, days).cumsum(), index=dates
        ),
        mkt=pd.Series(rng.normal(0, 0.01, days), index=dates),
        smb=pd.Series(rng.normal(0, 0.01, days), index=dates),
        hml=pd.Series(rng.normal(0, 0.01, days), index=dates),
    )
    return GTJA191Panels(
        open=open_,
        close=close,
        high=high,
        low=low,
        volume=volume,
        amount=close * volume,
        vwap=(high + low + close) / 3.0,
        returns=returns,
        external=external,
    )


def _raw_symbol_frames(days: int = 40, symbols: int = 2) -> dict[str, pd.DataFrame]:
    dates = pd.date_range("2025-01-01", periods=days, freq="B")
    output = {}
    for number in range(1, symbols + 1):
        close = np.arange(days, dtype=float) + 10 + number
        output[f"{number:06d}"] = pd.DataFrame(
            {
                "date": dates,
                "open": close - 0.2,
                "close": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "volume": np.arange(days, dtype=float) + 1000,
                "amount": (np.arange(days, dtype=float) + 1000) * close / 1000,
            }
        )
    return output


class GTJA191OperatorsTest(unittest.TestCase):
    def test_normalize_gtja_name_accepts_supported_aliases(self):
        self.assertEqual(normalize_gtja_name(1), "gtja_001")
        self.assertEqual(normalize_gtja_name("gtja1"), "gtja_001")
        self.assertEqual(normalize_gtja_name("gtja_191"), "gtja_191")

    def test_normalize_gtja_name_rejects_other_families_and_out_of_range(self):
        with self.assertRaises(KeyError):
            normalize_gtja_name("alpha_001")
        with self.assertRaises(KeyError):
            normalize_gtja_name(192)

    def test_sma_cn_uses_report_recursion(self):
        frame = pd.DataFrame({"a": [1.0, 4.0, 7.0]})
        expected = pd.DataFrame({"a": [1.0, 2.0, 11.0 / 3.0]})
        pd.testing.assert_frame_equal(sma_cn(frame, 3, 1), expected)

    def test_wma_uses_point_nine_distance_weights(self):
        frame = pd.DataFrame({"a": [1.0, 2.0, 4.0]})
        expected = (4.0 + 0.9 * 2.0 + 0.9**2) / (1.0 + 0.9 + 0.9**2)
        self.assertAlmostEqual(wma(frame, 3).iloc[-1, 0], expected)
        self.assertTrue(wma(frame, 3).iloc[:2, 0].isna().all())

    def test_highday_and_lowday_return_distance_from_current(self):
        frame = pd.DataFrame({"a": [3.0, 1.0, 2.0, 3.0]})
        self.assertEqual(highday(frame, 4).iloc[-1, 0], 0.0)
        self.assertEqual(lowday(frame, 4).iloc[-1, 0], 2.0)

    def test_regbeta_and_regresi_use_intercept(self):
        independent = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        dependent = 2.0 * independent + 3.0
        self.assertAlmostEqual(
            regbeta(dependent, independent, 4).iloc[-1, 0],
            2.0,
        )
        self.assertAlmostEqual(
            regresi(dependent, independent, 4).iloc[-1, 0],
            0.0,
        )

    def test_count_and_sumif_require_full_windows(self):
        values = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        condition = values > 2.0
        actual_count = count(condition, 3)
        actual_sum = sumif(values, 3, condition)
        self.assertTrue(actual_count.iloc[:2, 0].isna().all())
        self.assertEqual(actual_count.iloc[-1, 0], 2.0)
        self.assertEqual(actual_sum.iloc[-1, 0], 7.0)


class GTJA191PanelsTest(unittest.TestCase):
    def test_panels_keep_aligned_market_data_and_default_external_inputs(self):
        frame = pd.DataFrame(
            {"000001": [10.0, 11.0], "000002": [20.0, 21.0]},
            index=pd.date_range("2026-01-01", periods=2),
        )
        panels = GTJA191Panels(
            open=frame,
            close=frame,
            high=frame,
            low=frame,
            volume=frame,
            amount=frame,
            vwap=frame,
            returns=frame.pct_change(fill_method=None),
        )
        self.assertEqual(panels.close.columns.tolist(), ["000001", "000002"])
        self.assertEqual(panels.external, GTJA191ExternalData())


class GTJA191FirstEightyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panels = _complete_panels()
        cls.calculator = GTJA191(cls.panels)

    def test_registry_contains_all_191_names(self):
        self.assertEqual(len(GTJA191_NAMES), 191)
        self.assertEqual(GTJA191_NAMES[0], "gtja_001")
        self.assertEqual(GTJA191_NAMES[-1], "gtja_191")

    def test_gtja_001_matches_formula(self):
        expected = -correlation(
            rank(delta(np.log(self.panels.volume), 1)),
            rank((self.panels.close - self.panels.open) / self.panels.open),
            6,
        )
        pd.testing.assert_frame_equal(self.calculator.calculate(1), expected)

    def test_gtja_015_matches_open_gap(self):
        expected = self.panels.open / delay(self.panels.close, 1) - 1.0
        pd.testing.assert_frame_equal(self.calculator.calculate(15), expected)

    def test_gtja_070_is_amount_volatility(self):
        pd.testing.assert_frame_equal(
            self.calculator.calculate(70),
            stddev(self.panels.amount, 6),
        )

    def test_gtja_080_is_five_day_volume_change_percent(self):
        expected = (
            (self.panels.volume - delay(self.panels.volume, 5))
            / delay(self.panels.volume, 5)
            * 100.0
        )
        pd.testing.assert_frame_equal(self.calculator.calculate(80), expected)

    def test_first_eighty_return_aligned_panels(self):
        for number in range(1, 81):
            with self.subTest(number=number):
                result = self.calculator.calculate(number)
                self.assertEqual(result.shape, self.panels.close.shape)


class GTJA191MiddleEightyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panels = _complete_panels(days=400)
        cls.calculator = GTJA191(cls.panels)

    def test_gtja_088_is_twenty_day_price_change_percent(self):
        previous = delay(self.panels.close, 20)
        expected = (self.panels.close - previous) / previous * 100.0
        pd.testing.assert_frame_equal(self.calculator.calculate(88), expected)

    def test_gtja_100_is_twenty_day_volume_std(self):
        pd.testing.assert_frame_equal(
            self.calculator.calculate(100),
            stddev(self.panels.volume, 20),
        )

    def test_gtja_126_is_typical_price(self):
        expected = (self.panels.close + self.panels.high + self.panels.low) / 3.0
        pd.testing.assert_frame_equal(self.calculator.calculate(126), expected)

    def test_gtja_143_carries_prior_value_on_non_positive_day(self):
        actual = self.calculator.calculate(143)
        mask = self.panels.close <= delay(self.panels.close, 1)
        pd.testing.assert_frame_equal(
            actual.where(mask),
            delay(actual, 1).where(mask),
        )

    def test_gtja_149_requires_benchmark_close(self):
        panels = GTJA191Panels(
            **{**self.panels.__dict__, "external": GTJA191ExternalData()}
        )
        with self.assertRaisesRegex(GTJA191DataError, "benchmark_close"):
            GTJA191(panels).calculate(149)

    def test_middle_eighty_return_aligned_panels(self):
        for number in range(81, 161):
            with self.subTest(number=number):
                result = self.calculator.calculate(number)
                self.assertEqual(result.shape, self.panels.close.shape)


class GTJA191CompleteLibraryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panels = _complete_panels(days=420)
        cls.calculator = GTJA191(cls.panels)

    def test_gtja_168_is_negative_relative_volume(self):
        expected = -self.panels.volume / self.panels.volume.rolling(20).mean()
        pd.testing.assert_frame_equal(self.calculator.calculate(168), expected)

    def test_gtja_191_matches_formula(self):
        expected = correlation(self.panels.volume.rolling(20).mean(), self.panels.low, 5)
        expected = expected + (self.panels.high + self.panels.low) / 2.0
        expected = expected - self.panels.close
        pd.testing.assert_frame_equal(self.calculator.calculate(191), expected)

    def test_all_191_factors_are_callable_and_aligned(self):
        for name in GTJA191_NAMES:
            with self.subTest(name=name):
                result = self.calculator.calculate(name)
                self.assertEqual(result.index.tolist(), self.panels.close.index.tolist())
                self.assertEqual(result.columns.tolist(), self.panels.close.columns.tolist())


class GTJA191RawAdapterTest(unittest.TestCase):
    def test_single_factor_script_lists_gtja_family(self):
        result = subprocess.run(
            [sys.executable, "scripts/test_factor.py", "--list-factors"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
        names = result.stdout.splitlines()
        self.assertIn("gtja_001", names)
        self.assertIn("gtja_191", names)

    def test_public_factor_package_exports_gtja_registry(self):
        from pipeline.factors import GTJA191_NAMES as exported_names

        self.assertEqual(exported_names, GTJA191_NAMES)

    def test_raw_adapter_builds_amount_and_typical_vwap(self):
        panels = build_gtja191_panels(_raw_symbol_frames())
        expected_vwap = (panels.high + panels.low + panels.close) / 3.0
        pd.testing.assert_frame_equal(panels.vwap, expected_vwap)
        self.assertEqual(panels.close.columns.tolist(), ["000001", "000002"])

    def test_long_adapter_routes_through_factor_tester(self):
        raw = _raw_symbol_frames()
        direct = gtja191_to_long(raw, "gtja_015")
        routed = build_long_factor_frame_from_raw(raw, factor_name="gtja_015")
        self.assertEqual(direct["symbol"].str.len().unique().tolist(), [6])
        self.assertIn("turnover_value", direct.columns)
        pd.testing.assert_frame_equal(direct, routed)


if __name__ == "__main__":
    unittest.main()
