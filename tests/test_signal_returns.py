import math
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.signal_returns import (
    build_signal_return_rows,
    filter_selectors_by_strategy,
    filter_picks_by_date,
    format_percent,
    summary_to_rows,
    summarize_signal_returns,
)


def _frame(closes):
    dates = pd.date_range("2026-01-01", periods=len(closes), freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1000] * len(closes),
        }
    ).set_index("date", drop=False)


class SignalReturnsTest(unittest.TestCase):
    def test_filter_selectors_by_strategy_keeps_requested_strategy_names(self):
        selectors = [("b1", object()), ("brick", object())]

        filtered = filter_selectors_by_strategy(selectors, ("b1",))

        self.assertEqual([name for name, _ in filtered], ["b1"])

    def test_format_percent_uses_two_decimal_places_and_blank_missing_values(self):
        self.assertEqual(format_percent(0.12345), "12.35%")
        self.assertEqual(format_percent(-0.001), "-0.10%")
        self.assertEqual(format_percent(0), "0.00%")
        self.assertEqual(format_percent(None), "")
        self.assertEqual(format_percent(float("nan")), "")

    def test_build_signal_return_rows_uses_future_close_returns(self):
        prepared = {
            "000001": _frame([10 + i for i in range(31)]),
            "000002": _frame([20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 1]),
        }
        picks_by_date = {
            pd.Timestamp("2026-01-01"): ["000001", "000002"],
            pd.Timestamp("2026-01-31"): ["000001"],
        }

        rows = build_signal_return_rows(
            prepared,
            picks_by_date,
            horizons=(1, 5, 10, 30),
            strategy="b1",
        )

        self.assertEqual(len(rows), 3)
        first = rows[0]
        self.assertEqual(first["date"], "2026-01-01")
        self.assertEqual(first["code"], "000001")
        self.assertEqual(first["strategy"], "b1")
        self.assertEqual(first["buy_mode"], "signal_close")
        self.assertEqual(first["entry_date"], "2026-01-01")
        self.assertAlmostEqual(first["entry_price"], 10.0)
        self.assertAlmostEqual(first["close"], 10.0)
        self.assertAlmostEqual(first["return_1d"], 0.1)
        self.assertAlmostEqual(first["return_5d"], 0.5)
        self.assertAlmostEqual(first["return_10d"], 1.0)
        self.assertAlmostEqual(first["return_30d"], 3.0)

        last = rows[-1]
        self.assertTrue(math.isnan(last["return_1d"]))
        self.assertTrue(math.isnan(last["return_5d"]))
        self.assertTrue(math.isnan(last["return_10d"]))

    def test_build_signal_return_rows_can_use_next_open_entry(self):
        dates = pd.date_range("2026-01-01", periods=7, freq="D")
        prepared = {
            "000001": pd.DataFrame(
                {
                    "date": dates,
                    "open": [10, 20, 30, 40, 50, 60, 70],
                    "high": [10, 22, 33, 44, 55, 66, 77],
                    "low": [10, 18, 27, 36, 45, 54, 63],
                    "close": [10, 22, 33, 44, 55, 66, 77],
                    "volume": [1000] * 7,
                }
            ).set_index("date", drop=False)
        }
        picks_by_date = {pd.Timestamp("2026-01-01"): ["000001"]}

        rows = build_signal_return_rows(
            prepared,
            picks_by_date,
            horizons=(1, 5),
            strategy="b1",
            buy_mode="next_open",
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["buy_mode"], "next_open")
        self.assertEqual(rows[0]["entry_date"], "2026-01-02")
        self.assertAlmostEqual(rows[0]["entry_price"], 20.0)
        self.assertAlmostEqual(rows[0]["return_1d"], 0.5)
        self.assertAlmostEqual(rows[0]["return_5d"], 2.5)

    def test_summarize_signal_returns_reports_mean_median_win_rate_and_count(self):
        rows = [
            {"return_1d": 0.10, "return_5d": 0.50, "return_10d": 1.00},
            {"return_1d": -0.10, "return_5d": -0.50, "return_10d": -0.95},
            {"return_1d": float("nan"), "return_5d": float("nan"), "return_10d": float("nan")},
        ]

        summary = summarize_signal_returns(rows, horizons=(1, 5, 10))

        self.assertEqual(summary["return_1d"]["count"], 2)
        self.assertAlmostEqual(summary["return_1d"]["mean_return"], 0.0)
        self.assertAlmostEqual(summary["return_1d"]["median_return"], 0.0)
        self.assertAlmostEqual(summary["return_1d"]["win_rate"], 0.5)
        self.assertAlmostEqual(summary["return_10d"]["mean_return"], 0.025)

    def test_summary_to_rows_flattens_metrics_for_csv(self):
        summary = {
            "return_1d": {
                "count": 2,
                "mean_return": 0.01,
                "median_return": 0.02,
                "win_rate": 0.5,
            },
            "return_5d": {
                "count": 0,
                "mean_return": None,
                "median_return": None,
                "win_rate": None,
            },
        }

        rows = summary_to_rows(summary)

        self.assertEqual(
            rows,
            [
                {
                    "horizon": "return_1d",
                    "count": 2,
                    "mean_return": 0.01,
                    "median_return": 0.02,
                    "win_rate": 0.5,
                },
                {
                    "horizon": "return_5d",
                    "count": 0,
                    "mean_return": None,
                    "median_return": None,
                    "win_rate": None,
                },
            ],
        )

    def test_filter_picks_by_date_limits_signal_dates_without_trimming_future_bars(self):
        prepared = {
            "000001": _frame([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]),
        }
        picks_by_date = {
            pd.Timestamp("2026-01-03"): ["000001"],
            pd.Timestamp("2026-01-04"): ["000001"],
        }

        filtered = filter_picks_by_date(
            picks_by_date,
            start_date="2026-01-01",
            end_date="2026-01-03",
        )
        rows = build_signal_return_rows(
            prepared,
            filtered,
            horizons=(10,),
            strategy="b1",
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["date"], "2026-01-03")
        self.assertAlmostEqual(rows[0]["return_10d"], 10 / 12)


if __name__ == "__main__":
    unittest.main()
