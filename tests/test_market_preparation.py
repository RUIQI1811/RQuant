import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market.preparation import MarketDataPreparer, _prepare_worker


class MarketDataPreparerTest(unittest.TestCase):
    def test_brick_only_parallel_path_propagates_symbol_context(self):
        class BrokenSelector:
            @staticmethod
            def prepare_df_brick_only(frame):
                raise ValueError("broken brick calculation")

        preparer = MarketDataPreparer(n_jobs=1)
        frames = {"000001": pd.DataFrame({"close": [10.0]})}

        with self.assertRaisesRegex(
            RuntimeError,
            "brick-only feature preparation failed for symbol 000001",
        ) as raised:
            preparer.apply_brick_features_only(frames, BrokenSelector())

        self.assertIsInstance(raised.exception.__cause__, ValueError)

    def test_brick_only_parallel_path_waits_for_in_place_updates(self):
        class UpdatingSelector:
            @staticmethod
            def prepare_df_brick_only(frame):
                frame["brick_ready"] = True
                return frame

        preparer = MarketDataPreparer(n_jobs=2)
        frames = {
            "000001": pd.DataFrame({"close": [10.0]}),
            "000002": pd.DataFrame({"close": [11.0]}),
        }

        result = preparer.apply_brick_features_only(frames, UpdatingSelector())

        self.assertIs(result, frames)
        self.assertTrue(all(frame["brick_ready"].all() for frame in result.values()))

    def test_prepare_accepts_date_as_both_index_and_column(self):
        dates = pd.date_range("2026-01-01", periods=3, freq="D")
        raw = pd.DataFrame(
            {
                "date": dates,
                "open": [10.0, 11.0, 12.0],
                "high": [10.5, 11.5, 12.5],
                "low": [9.5, 10.5, 11.5],
                "close": [10.0, 11.0, 12.0],
                "volume": [1000, 1200, 1400],
            }
        ).set_index("date", drop=False)

        code, frame = _prepare_worker(("000001", raw, None, None, 250, 20, None))

        self.assertEqual(code, "000001")
        self.assertIsNotNone(frame)
        self.assertIn("date", frame.columns)
        self.assertEqual(frame.index.name, "date")
        self.assertEqual(frame["date"].tolist(), list(dates))
        self.assertIn("turnover_n", frame.columns)


if __name__ == "__main__":
    unittest.main()
