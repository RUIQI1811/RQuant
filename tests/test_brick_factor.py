import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strategies.selector import BrickChartSelector, BrickComputeParams
from reports.factor_tester import (
    FactorTester,
    FactorTesterConfig,
    build_long_factor_frame_from_raw,
)
from factors.brick import brick_factor_to_long


def _raw_frame(periods: int = 180) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=periods, freq="B")
    phase = np.linspace(0.0, 18.0 * np.pi, periods)
    close = 20.0 + np.sin(phase) * 2.0 + np.arange(periods) * 0.01
    return pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.998,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": np.arange(periods, dtype=float) + 1000.0,
        }
    )


def _brick_config() -> dict:
    return {
        "n": 4,
        "m1": 4,
        "m2": 6,
        "m3": 6,
        "t": 4.0,
        "shift1": 90.0,
        "shift2": 100.0,
        "sma_w1": 1,
        "sma_w2": 1,
        "sma_w3": 1,
        "daily_return_threshold": 1.0,
        "brick_growth_ratio": 0.0,
        "min_prior_green_bars": 1,
        "zxdq_ratio": None,
        "require_zxdq_gt_zxdkx": False,
        "require_weekly_ma_bull": False,
        "zxdkx_m1": 2,
        "zxdkx_m2": 3,
        "zxdkx_m3": 4,
        "zxdkx_m4": 5,
        "wma_short": 1,
        "wma_mid": 2,
        "wma_long": 3,
    }


class BrickFactorTest(unittest.TestCase):
    def test_dense_growth_matches_existing_brick_formula(self):
        raw = _raw_frame(80)
        actual = brick_factor_to_long(
            {"000001.SZ": raw},
            "brick_growth",
            config=_brick_config(),
        )

        params = BrickComputeParams(
            n=4, m1=4, m2=6, m3=6, t=4.0, shift1=90.0, shift2=100.0
        )
        brick = params.compute_arr(raw)
        previous = np.r_[np.nan, brick[:-1]]
        expected = np.divide(
            brick,
            np.abs(previous),
            out=brick.astype(float, copy=True),
            where=np.abs(previous) > 0,
        )

        np.testing.assert_allclose(actual["factor_value"], expected, equal_nan=True)
        self.assertTrue(actual["symbol"].eq("000001").all())
        self.assertTrue(actual["factor_value"].equals(actual["brick_growth"]))

    def test_strategy_factor_matches_selector_gate(self):
        raw = _raw_frame()
        config = _brick_config()
        actual = brick_factor_to_long({"000001": raw}, "brick", config=config)

        indexed = raw.set_index("date", drop=False)
        selector = BrickChartSelector(
            daily_return_threshold=1.0,
            brick_growth_ratio=0.0,
            min_prior_green_bars=1,
            zxdq_ratio=None,
            require_zxdq_gt_zxdkx=False,
            require_weekly_ma_bull=False,
            zxdkx_m1=2,
            zxdkx_m2=3,
            zxdkx_m3=4,
            zxdkx_m4=5,
            wma_short=1,
            wma_mid=2,
            wma_long=3,
        )
        expected = selector.prepare_df(indexed)
        expected_factor = expected["brick_growth"].where(expected["_vec_pick"])

        np.testing.assert_allclose(actual["factor_value"], expected_factor, equal_nan=True)
        self.assertGreater(int(actual["brick_signal"].sum()), 0)
        self.assertTrue(actual.loc[~actual["brick_signal"], "factor_value"].isna().all())

    def test_dense_factor_is_causal_when_future_rows_are_appended(self):
        raw = _raw_frame(81)
        before = brick_factor_to_long(
            {"000001": raw.iloc[:80]}, "brick_growth", config=_brick_config()
        )
        after = brick_factor_to_long(
            {"000001": raw}, "brick_growth", config=_brick_config()
        )

        np.testing.assert_allclose(
            before["factor_value"],
            after.iloc[:80]["factor_value"],
            equal_nan=True,
        )

    def test_raw_adapter_and_factor_tester_apply_one_day_lag(self):
        raw = _raw_frame(80)
        long = build_long_factor_frame_from_raw(
            {"1": raw},
            factor_name="brick_growth",
            factor_config=_brick_config(),
        )
        tester = FactorTester(
            long,
            factor_name="brick_growth",
            config=FactorTesterConfig(
                forward_return_windows=(1,),
                groups=5,
                min_listing_days=0,
                commission_rate=0.0,
                slippage_rate=0.0,
                stamp_tax_rate=0.0,
            ),
        )

        prepared = tester.prepare_data().sort_values("date").reset_index(drop=True)
        self.assertTrue(pd.isna(prepared.loc[0, "factor_processed"]))
        self.assertEqual(prepared.loc[2, "factor_processed"], prepared.loc[1, "factor_raw"])


class TopLevelBrickFactorImportTests(unittest.TestCase):
    def test_brick_factor_imports_from_top_level_package(self):
        import factors.brick as brick

        self.assertTrue(hasattr(brick, "brick_factor_to_long"))


if __name__ == "__main__":
    unittest.main()
