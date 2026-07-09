import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market import fetch_kline


def _kline_frame(dates: list[str], closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "open": closes,
            "close": closes,
            "high": closes,
            "low": closes,
            "volume": [1000.0] * len(dates),
        }
    )


class FetchKlineRangeOverwriteTest(unittest.TestCase):
    def test_requested_range_overwrites_only_that_part(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            _kline_frame(
                ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
                [10.0, 11.0, 12.0, 13.0],
            ).to_csv(
                output / "000001.csv", index=False
            )
            replacement = _kline_frame(["2026-01-02", "2026-01-03"], [11.0, 12.0])

            with patch.object(
                fetch_kline, "_get_kline_tushare", return_value=replacement
            ) as api:
                outcome = fetch_kline.fetch_one(
                    "000001", "20260102", "20260103", output
                )

            self.assertEqual(outcome, "overwritten")
            api.assert_called_once_with("000001", "20260102", "20260103")
            saved = pd.read_csv(output / "000001.csv")
            self.assertEqual(
                saved["date"].tolist(),
                ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
            )

    def test_existing_end_date_is_requested_and_overwritten(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            _kline_frame(["2026-01-01", "2026-01-02"], [10.0, 11.0]).to_csv(
                output / "000001.csv", index=False
            )
            requested_day = _kline_frame(["2026-01-02"], [11.0])

            with patch.object(
                fetch_kline, "_get_kline_tushare", return_value=requested_day
            ) as api:
                outcome = fetch_kline.fetch_one(
                    "000001", "20260102", "20260102", output
                )

            self.assertEqual(outcome, "overwritten")
            api.assert_called_once_with("000001", "20260102", "20260102")
            saved = pd.read_csv(output / "000001.csv")
            self.assertEqual(saved["date"].tolist(), ["2026-01-01", "2026-01-02"])

    def test_empty_update_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            csv_path = output / "000001.csv"
            _kline_frame(["2026-01-01", "2026-01-02"], [10.0, 11.0]).to_csv(
                csv_path, index=False
            )
            original = csv_path.read_bytes()

            with patch.object(
                fetch_kline, "_get_kline_tushare", return_value=pd.DataFrame()
            ):
                outcome = fetch_kline.fetch_one(
                    "000001", "20260103", "20260103", output
                )

            self.assertEqual(outcome, "no_new_data")
            self.assertEqual(csv_path.read_bytes(), original)

    def test_changed_qfq_overlap_triggers_full_refresh(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            _kline_frame(["2026-01-01", "2026-01-02"], [10.0, 11.0]).to_csv(
                output / "000001.csv", index=False
            )
            incremental = _kline_frame(
                ["2026-01-02", "2026-01-03"], [10.0, 11.0]
            )
            refreshed = _kline_frame(
                ["2026-01-01", "2026-01-02", "2026-01-03"],
                [9.0, 10.0, 11.0],
            )

            with patch.object(
                fetch_kline,
                "_get_kline_tushare",
                side_effect=[incremental, refreshed],
            ) as api:
                outcome = fetch_kline.fetch_one(
                    "000001", "20260102", "20260103", output
                )

            self.assertEqual(outcome, "refreshed")
            self.assertEqual(
                [call.args for call in api.call_args_list],
                [
                    ("000001", "20260102", "20260103"),
                    ("000001", "20260101", "20260103"),
                ],
            )
            saved = pd.read_csv(output / "000001.csv")
            self.assertEqual(saved["close"].tolist(), [9.0, 10.0, 11.0])


class TopLevelFetchImportTests(unittest.TestCase):
    def test_fetch_kline_imports_from_top_level_package(self):
        import market.fetch_kline as fetch_kline

        self.assertTrue(hasattr(fetch_kline, "main"))


if __name__ == "__main__":
    unittest.main()
