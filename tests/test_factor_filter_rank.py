import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factors.alpha101 import Alpha101Panels
from factors.filter_rank import (
    FilterRankConfig,
    build_filter_rank_frame,
    filter_then_rank,
    write_filter_rank_reports,
)


class FilterThenRankTest(unittest.TestCase):
    def test_selects_an_inclusive_post_filter_rank_interval(self):
        frame = pd.DataFrame(
            {
                "date": ["2026-06-23"] * 6,
                "symbol": [1, 2, 3, 4, 5, 6],
                "filter_value": [1, 2, 3, 4, 5, 6],
                "rank_value": [0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            }
        )
        config = FilterRankConfig(
            filter_top_quantile=1.0,
            top_n=1,
            rank_start=2,
            rank_end=4,
            min_universe=2,
        )

        result = filter_then_rank(frame, config=config)

        self.assertEqual(result.selections["symbol"].tolist(), ["000002", "000003", "000004"])
        self.assertEqual(result.selections["rank_position"].tolist(), [2, 3, 4])
        self.assertEqual(result.daily_summary.loc[0, "rank_start"], 2)
        self.assertEqual(result.daily_summary.loc[0, "rank_end"], 4)

    def test_rank_end_must_not_precede_rank_start(self):
        with self.assertRaisesRegex(ValueError, "rank_end"):
            FilterRankConfig(rank_start=500, rank_end=200)

    def test_alpha77_filters_before_alpha40_ranking(self):
        frame = pd.DataFrame(
            {
                "date": ["2026-06-23"] * 6,
                "symbol": [1, 2, 3, 4, 5, 6],
                "filter_value": [1, 2, 3, 4, 5, 6],
                "rank_value": [1.0, 0.9, 0.8, 0.7, 0.95, 0.1],
                "eligible": [True] * 6,
            }
        )
        config = FilterRankConfig(
            filter_top_quantile=0.5,
            top_n=2,
            min_universe=2,
        )

        result = filter_then_rank(frame, config=config)

        self.assertEqual(result.selections["symbol"].tolist(), ["000005", "000004"])
        self.assertEqual(result.selections["rank_position"].tolist(), [1, 2])
        self.assertEqual(result.signals["score"].to_list(), [1.0, 2 / 3])
        self.assertEqual(result.signals["weight"].to_list(), [0.5, 0.5])
        self.assertEqual(result.daily_summary.loc[0, "filtered_count"], 3)
        metadata = result.signals.item(0, "metadata")
        self.assertEqual(metadata["filter_factor"], "alpha_077")
        self.assertEqual(metadata["rank_factor"], "alpha_040")
        self.assertEqual(metadata["factor_lag_days"], 1)

    def test_small_universe_is_skipped_instead_of_forcing_signals(self):
        frame = pd.DataFrame(
            {
                "date": ["2026-06-23"] * 2,
                "symbol": ["000001", "000002"],
                "filter_value": [0.8, 0.9],
                "rank_value": [0.2, 0.3],
            }
        )
        result = filter_then_rank(
            frame,
            config=FilterRankConfig(min_universe=3),
        )
        self.assertTrue(result.signals.is_empty())
        self.assertEqual(result.daily_summary.loc[0, "status"], "skipped")
        self.assertIn("below min_universe", result.daily_summary.loc[0, "message"])

    def test_factor_values_and_eligibility_are_lagged_before_selection(self):
        dates = pd.date_range("2026-01-01", periods=3, freq="B")
        symbols = ["000001", "000002"]
        close = pd.DataFrame([[10, 20], [11, 21], [12, 22]], index=dates, columns=symbols)
        panels = Alpha101Panels(
            open=close,
            close=close,
            high=close,
            low=close,
            volume=close * 100,
            vwap=close,
            returns=close.pct_change(fill_method=None),
            turnover_value=close * 1000,
        )
        filter_values = pd.DataFrame([[1, 2], [3, 4], [5, 6]], index=dates, columns=symbols)
        rank_values = filter_values * 10

        with patch(
            "factors.filter_rank.Alpha101.calculate",
            side_effect=lambda name: filter_values if name == "alpha_077" else rank_values,
        ):
            frame, status = build_filter_rank_frame(
                panels,
                config=FilterRankConfig(
                    factor_lag_days=1,
                    min_listing_days=0,
                    min_universe=2,
                ),
                dates=[dates[-1]],
            )

        actual = frame.sort_values("symbol")
        self.assertEqual(actual["filter_value"].tolist(), [3.0, 4.0])
        self.assertEqual(actual["rank_value"].tolist(), [30.0, 40.0])
        self.assertEqual(actual["reference_close"].tolist(), [11.0, 21.0])
        self.assertEqual(status.set_index("filter").at["is_st", "status"], "missing_input")

    def test_reports_preserve_unified_schema_and_json_metadata(self):
        frame = pd.DataFrame(
            {
                "date": ["2026-06-23"] * 3,
                "symbol": ["000001", "000002", "000003"],
                "filter_value": [0.7, 0.8, 0.9],
                "rank_value": [0.7, 0.8, 0.9],
            }
        )
        config = FilterRankConfig(filter_top_quantile=1.0, top_n=2, min_universe=2)
        result = filter_then_rank(frame, config=config)
        with tempfile.TemporaryDirectory() as temp_dir:
            output = write_filter_rank_reports(result, temp_dir, config=config)
            payload = json.loads((output / "signals.json").read_text(encoding="utf-8"))
            self.assertEqual(len(payload["signals"]), 2)
            self.assertIsInstance(payload["signals"][0]["metadata"], dict)
            csv_signals = pd.read_csv(output / "signals.csv", dtype={"symbol": str})
            self.assertEqual(csv_signals["symbol"].tolist(), ["000003", "000002"])
            self.assertTrue((output / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
