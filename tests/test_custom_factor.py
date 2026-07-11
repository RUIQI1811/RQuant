import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factors.alpha101 import Alpha101Panels
from factors.custom import (
    CUSTOM_FACTOR_NAMES,
    CustomFactorDataError,
    CustomFactors,
    build_custom_factor_panels,
    custom_factor_to_long,
    normalize_custom_factor_name,
)
from factors.operators import covariance, delay, rank, safe_div
from reports.factor_tester import (
    FactorTester,
    FactorTesterConfig,
    build_long_factor_frame_from_raw,
)


def _panels(days: int = 12, symbols: int = 6) -> Alpha101Panels:
    dates = pd.date_range("2026-01-01", periods=days, freq="B")
    columns = [f"{number:06d}" for number in range(1, symbols + 1)]
    day = np.arange(days, dtype=float)[:, None]
    stock = np.arange(symbols, dtype=float)[None, :]
    close = pd.DataFrame(
        10.0 + day * (0.1 + stock * 0.01) + np.sin(day + stock) * 0.05,
        index=dates,
        columns=columns,
    )
    volume = pd.DataFrame(
        1000.0 + day * (5.0 + stock) + stock * 100.0,
        index=dates,
        columns=columns,
    )
    return Alpha101Panels(
        open=close * 0.99,
        close=close,
        high=close * 1.01,
        low=close * 0.98,
        volume=volume,
        vwap=close,
        returns=close.pct_change(fill_method=None),
        turnover_value=close * volume,
    )


def _raw_data(panels: Alpha101Panels) -> dict[str, pd.DataFrame]:
    raw = {}
    for symbol in panels.close.columns:
        raw[symbol] = pd.DataFrame(
            {
                "date": panels.close.index,
                "open": panels.open[symbol].to_numpy(),
                "close": panels.close[symbol].to_numpy(),
                "high": panels.high[symbol].to_numpy(),
                "low": panels.low[symbol].to_numpy(),
                "volume": panels.volume[symbol].to_numpy(),
                "turnover_value": panels.turnover_value[symbol].to_numpy(),
            }
        )
    return raw


class CustomFactorTest(unittest.TestCase):
    def test_registry_contains_custom_factor(self):
        self.assertEqual(CUSTOM_FACTOR_NAMES, ("custom_001", "custom_002"))
        self.assertEqual(normalize_custom_factor_name(1), "custom_001")
        self.assertEqual(normalize_custom_factor_name(2), "custom_002")
        self.assertEqual(normalize_custom_factor_name("custom001"), "custom_001")
        self.assertEqual(
            normalize_custom_factor_name("custom-return-turnover-cov-5d"),
            "custom_001",
        )
        with self.assertRaises(KeyError):
            normalize_custom_factor_name("custom_003")

    def test_factor_matches_requested_formula(self):
        panels = _panels()
        actual = CustomFactors(panels).calculate("custom_001")
        daily_return = panels.close / delay(panels.close, 1) - 1.0
        expected = -rank(
            covariance(rank(daily_return), rank(panels.turnover_value), 5)
        )
        pd.testing.assert_frame_equal(actual, expected)

    def test_custom_002_matches_vwap_close_gap_formula(self):
        panels = _panels()
        actual = CustomFactors(panels).calculate("custom_002")
        expected = rank(safe_div(panels.vwap - panels.close, panels.vwap))
        pd.testing.assert_frame_equal(actual, expected)

    def test_missing_turnover_is_explicit(self):
        panels = _panels()
        missing = Alpha101Panels(**{**panels.__dict__, "turnover_value": None})
        with self.assertRaisesRegex(CustomFactorDataError, "requires turnover_value"):
            CustomFactors(missing).calculate("custom_001")

        fallback = CustomFactors(missing).calculate_many(on_error="nan")
        self.assertTrue(fallback["custom_001"].isna().all().all())
        self.assertFalse(fallback["custom_002"].isna().all().all())

    def test_panel_builder_has_family_specific_error_contract(self):
        with self.assertRaises(CustomFactorDataError):
            build_custom_factor_panels({})

    def test_raw_adapter_routes_custom_factor_and_factor_tester_lags_one_day(self):
        panels = _panels()
        raw = _raw_data(panels)
        direct = custom_factor_to_long(raw, "custom_001")
        routed = build_long_factor_frame_from_raw(
            raw,
            factor_name="custom_001",
        )
        pd.testing.assert_frame_equal(direct, routed)
        legacy = custom_factor_to_long(raw, "custom_return_turnover_cov_5d")
        pd.testing.assert_frame_equal(direct, legacy)
        custom_002 = build_long_factor_frame_from_raw(
            raw,
            factor_name="custom_002",
        )
        self.assertEqual(len(custom_002), len(direct))
        self.assertTrue(custom_002["factor_value"].notna().any())

        tester = FactorTester(
            routed,
            factor_name="custom_001",
            config=FactorTesterConfig(
                forward_return_windows=(1,),
                groups=5,
                min_listing_days=0,
                commission_rate=0.0,
                slippage_rate=0.0,
                stamp_tax_rate=0.0,
            ),
        )
        prepared = tester.prepare_data().sort_values(["symbol", "date"])
        expected_lagged = prepared.groupby("symbol")["factor_raw"].shift(1)
        pd.testing.assert_series_equal(
            prepared["factor_lagged"],
            expected_lagged,
            check_names=False,
        )

    def test_single_factor_cli_lists_custom_factor(self):
        result = subprocess.run(
            [sys.executable, "scripts/test_factor.py", "--list-factors"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("custom_001", result.stdout.splitlines())
        self.assertIn("custom_002", result.stdout.splitlines())
        self.assertNotIn("custom_return_turnover_cov_5d", result.stdout.splitlines())


if __name__ == "__main__":
    unittest.main()
