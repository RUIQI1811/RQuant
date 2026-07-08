import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.signal_returns import build_enabled_selectors
from pipeline.select_stock import run_bdsr_macd_obv
from pipeline.strategies.bdsr_macd_obv import (
    BDSRMACDOBVSelector,
    add_bdsr_macd_obv_features,
)


TEST_PARAMS = {
    "bdsr_fast_window": 3,
    "bdsr_slow_window": 7,
    "macd_fast_period": 3,
    "macd_slow_period": 6,
    "macd_signal_period": 3,
    "obv_ma_window": 5,
    "obv_trend_lookback": 2,
}


def _qualifying_frame() -> pd.DataFrame:
    close = [
        20.631122,
        20.461702,
        20.362634,
        21.008921,
        21.266762,
        21.147061,
        21.447214,
        21.580499,
        21.838742,
        23.319604,
        23.185958,
        22.888124,
        23.957666,
        23.543675,
        24.827598,
        24.771659,
        24.903347,
        25.496887,
    ]
    volume = [
        2542.0,
        1645.0,
        1169.0,
        1274.0,
        542.0,
        942.0,
        758.0,
        1170.0,
        2856.0,
        2630.0,
        2389.0,
        2985.0,
        1271.0,
        1426.0,
        594.0,
        1654.0,
        1789.0,
        1337.0,
    ]
    dates = pd.date_range("2026-01-01", periods=len(close), freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "open": [value - 0.1 for value in close],
            "high": [value + 0.2 for value in close],
            "low": [value - 0.2 for value in close],
            "close": close,
            "volume": volume,
        }
    ).set_index("date", drop=False)


class BDSRMACDOBVTest(unittest.TestCase):
    def test_feature_builder_emits_exact_three_condition_signal(self):
        result = add_bdsr_macd_obv_features(_qualifying_frame(), **TEST_PARAMS)
        expected_columns = {
            "BDSR_FAST",
            "BDSR_SLOW",
            "MACD_DIF",
            "MACD_DEA",
            "MACD_HIST",
            "OBV",
            "OBV_MA",
            "bdsr_golden_cross",
            "macd_above_zero_golden_cross",
            "obv_uptrend",
            "bdsr_macd_obv_buy_signal",
        }
        self.assertTrue(expected_columns.issubset(result.columns))

        conditions = result[
            [
                "bdsr_golden_cross",
                "macd_above_zero_golden_cross",
                "obv_uptrend",
            ]
        ].all(axis=1)
        pd.testing.assert_series_equal(
            result["bdsr_macd_obv_buy_signal"],
            conditions,
            check_names=False,
        )
        self.assertTrue(result["bdsr_macd_obv_buy_signal"].iloc[-1])
        self.assertFalse(result["bdsr_macd_obv_buy_signal"].iloc[:-1].any())

    def test_crosses_are_causal_and_macd_cross_is_above_zero(self):
        result = add_bdsr_macd_obv_features(_qualifying_frame(), **TEST_PARAMS)
        previous = result.iloc[-2]
        signal = result.iloc[-1]

        self.assertLessEqual(previous["BDSR_FAST"], previous["BDSR_SLOW"])
        self.assertGreater(signal["BDSR_FAST"], signal["BDSR_SLOW"])
        self.assertLessEqual(previous["MACD_DIF"], previous["MACD_DEA"])
        self.assertGreater(signal["MACD_DIF"], signal["MACD_DEA"])
        self.assertGreater(signal["MACD_DIF"], 0.0)
        self.assertGreater(signal["MACD_DEA"], 0.0)

    def test_obv_uptrend_requires_above_rising_average(self):
        result = add_bdsr_macd_obv_features(_qualifying_frame(), **TEST_PARAMS)
        signal = result.iloc[-1]

        self.assertGreater(signal["OBV"], signal["OBV_MA"])
        self.assertGreater(result["OBV_MA"].iloc[-1], result["OBV_MA"].iloc[-3])
        self.assertTrue(signal["obv_uptrend"])

    def test_selector_and_enabled_registry_use_public_strategy_name(self):
        selectors = build_enabled_selectors(
            {
                "b1": {"enabled": False},
                "brick": {"enabled": False},
                "mbdsr": {"enabled": False},
                "bdsr_macd_obv": {"enabled": True, **TEST_PARAMS},
            }
        )

        self.assertEqual([name for name, _ in selectors], ["bdsr_macd_obv"])
        self.assertIsInstance(selectors[0][1], BDSRMACDOBVSelector)
        prepared = selectors[0][1].prepare_df(_qualifying_frame())
        self.assertTrue(prepared["_vec_pick"].iloc[-1])

    def test_preselect_candidate_preserves_auditable_indicator_values(self):
        frame = _qualifying_frame()
        frame["turnover_n"] = 10_000_000.0
        pick_date = frame.index[-1]

        candidates = run_bdsr_macd_obv(
            {"000001": frame},
            pick_date,
            ["000001"],
            {**TEST_PARAMS, "extra_bars_buffer": 0},
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].code, "000001")
        self.assertEqual(candidates[0].strategy, "bdsr_macd_obv")
        self.assertEqual(
            candidates[0].extra["signal_column"],
            "bdsr_macd_obv_buy_signal",
        )
        self.assertGreater(candidates[0].extra["macd_dif"], 0.0)
        self.assertGreater(candidates[0].extra["obv"], candidates[0].extra["obv_ma"])

    def test_missing_volume_is_rejected(self):
        with self.assertRaisesRegex(KeyError, "volume"):
            add_bdsr_macd_obv_features(
                _qualifying_frame().drop(columns="volume"),
                **TEST_PARAMS,
            )

    def test_invalid_fast_slow_relationship_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "bdsr_slow_window"):
            BDSRMACDOBVSelector(bdsr_fast_window=9, bdsr_slow_window=9)


class TopLevelBdsrMacdObvImportTests(unittest.TestCase):
    def test_bdsr_macd_obv_imports_from_top_level_package(self):
        import strategies.bdsr_macd_obv as strategy

        self.assertTrue(hasattr(strategy, "BDSRMACDOBVSelector"))


if __name__ == "__main__":
    unittest.main()
