import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.signal_returns import build_enabled_selectors
from pipeline.strategies.mbdsr import MBDSRSelector, add_mbdsr_features, calc_rci


def _qualifying_frame(*, add_next_confirmation: bool = False) -> pd.DataFrame:
    close = np.linspace(100.0, 120.0, 80)
    pullback_start = close[-11]
    close[-10:-1] = pullback_start - np.arange(1, 10) * 0.05
    close[-1] = close[-2] + 0.075

    frame = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=len(close), freq="D"),
            "open": close - 0.01,
            "high": close + 0.50,
            "low": close - 0.50,
            "close": close,
            "volume": 1000.0,
        }
    )
    if add_next_confirmation:
        prior_high = float(frame["high"].iloc[-1])
        confirmation_close = prior_high + 0.10
        frame.loc[len(frame)] = {
            "date": frame["date"].iloc[-1] + pd.Timedelta(days=1),
            "open": confirmation_close - 0.05,
            "high": confirmation_close + 0.20,
            "low": confirmation_close - 0.20,
            "close": confirmation_close,
            "volume": 1000.0,
        }
    return frame.set_index("date", drop=False)


class MBDSRTest(unittest.TestCase):
    def test_calc_rci_has_expected_extremes_and_warmup(self):
        rising = calc_rci(pd.Series(range(1, 13)), 9)
        falling = calc_rci(pd.Series(range(12, 0, -1)), 9)

        self.assertTrue(rising.iloc[:8].isna().all())
        self.assertAlmostEqual(rising.iloc[-1], 100.0)
        self.assertAlmostEqual(falling.iloc[-1], -100.0)

    def test_calc_rci_rejects_invalid_window(self):
        with self.assertRaisesRegex(ValueError, "at least 2"):
            calc_rci(pd.Series([1.0, 2.0]), 1)

    def test_calc_rci_uses_average_price_ranks_for_ties(self):
        prices = pd.Series([10.0, 11.0, 11.0, 9.0, 12.0])
        price_ranks = prices.rank(method="average").to_numpy()
        time_ranks = np.arange(1.0, 6.0)
        expected = (1.0 - 6.0 * np.square(time_ranks - price_ranks).sum() / (5 * 24)) * 100.0

        actual = calc_rci(prices, 5).iloc[-1]

        self.assertAlmostEqual(actual, expected)

    def test_feature_builder_emits_all_indicators_and_exact_signal(self):
        result = add_mbdsr_features(_qualifying_frame())
        expected_columns = {
            "MA20",
            "MA60",
            "ATR14",
            "OBV",
            "VOL20",
            "RCI9",
            "RCI26",
            "RCI52",
            "trend_filter",
            "rci_pullback",
            "support_touch",
            "volume_filter",
            "obv_filter",
            "atr_filter",
            "candle_confirm",
            "mBDSR_buy_signal",
            "mBDSR_buy_next_confirm",
        }
        self.assertTrue(expected_columns.issubset(result.columns))

        condition_columns = [
            "trend_filter",
            "rci_pullback",
            "support_touch",
            "volume_filter",
            "obv_filter",
            "atr_filter",
            "candle_confirm",
        ]
        expected_signal = result[condition_columns].all(axis=1)
        pd.testing.assert_series_equal(
            result["mBDSR_buy_signal"],
            expected_signal,
            check_names=False,
        )
        self.assertTrue(result["mBDSR_buy_signal"].iloc[-1])
        self.assertLess(result["RCI9"].iloc[-2], -80.0)
        self.assertGreater(result["RCI9"].iloc[-1], result["RCI9"].iloc[-2])

    def test_next_day_confirmation_requires_break_of_signal_high(self):
        result = add_mbdsr_features(_qualifying_frame(add_next_confirmation=True))

        self.assertTrue(result["mBDSR_buy_signal"].iloc[-2])
        self.assertTrue(result["mBDSR_buy_next_confirm"].iloc[-1])

        selector = MBDSRSelector(use_next_confirm=True, extra_bars_buffer=0)
        prepared = selector.prepare_df(_qualifying_frame(add_next_confirmation=True))
        self.assertTrue(prepared["_vec_pick"].iloc[-1])
        self.assertEqual(selector.signal_column, "mBDSR_buy_next_confirm")

    def test_missing_required_ohlcv_column_is_rejected(self):
        with self.assertRaisesRegex(KeyError, "volume"):
            add_mbdsr_features(_qualifying_frame().drop(columns="volume"))

    def test_enabled_selector_uses_configured_confirmation_mode(self):
        selectors = build_enabled_selectors(
            {
                "b1": {"enabled": False},
                "brick": {"enabled": False},
                "mbdsr": {"enabled": True, "use_next_confirm": True},
            }
        )

        self.assertEqual([name for name, _ in selectors], ["mbdsr_confirm"])
        self.assertIsInstance(selectors[0][1], MBDSRSelector)


class TopLevelMbdsrImportTests(unittest.TestCase):
    def test_mbdsr_imports_from_top_level_package(self):
        import strategies.mbdsr as mbdsr

        self.assertTrue(hasattr(mbdsr, "calc_obv"))


if __name__ == "__main__":
    unittest.main()
