import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from factors.external import load_research_context_file
from market.fetch_context import fetch_daily_basic_context


class _FakeTushareSession:
    def __init__(self, *, fail_date: str | None = None) -> None:
        self.fail_date = fail_date
        self.daily_calls: list[str] = []

    def trade_cal(self, **_kwargs):
        return pd.DataFrame(
            {
                "cal_date": ["20260102", "20260103", "20260105"],
                "is_open": [1, 0, 1],
            }
        )

    def daily_basic(self, *, trade_date: str, fields: str):
        self.daily_calls.append(trade_date)
        if trade_date == self.fail_date:
            raise RuntimeError("provider unavailable")
        self.last_fields = fields
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "600000.SH"],
                "trade_date": [trade_date, trade_date],
                "total_mv": [100.0, 250.0],
                "circ_mv": [80.0, 200.0],
                "turnover_rate": [1.2, 0.8],
                "volume_ratio": [0.9, 1.1],
                "pb": [2.0, 4.0],
            }
        )


class FetchResearchContextTest(unittest.TestCase):
    def test_fetches_by_open_date_converts_units_and_resumes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "context"
            first_session = _FakeTushareSession()
            result = fetch_daily_basic_context(
                start="2026-01-02",
                end="2026-01-05",
                output_dir=destination,
                max_requests_per_minute=0,
                session=first_session,
            )

            self.assertTrue(result["ok"])
            self.assertEqual(result["worker_count"], 8)
            self.assertEqual(first_session.daily_calls, ["20260102", "20260105"])
            self.assertTrue((destination / "2026/20260102.csv").exists())
            context = load_research_context_file(destination)
            first = context.loc[
                context["symbol"].eq("000001") & context["date"].eq("2026-01-02")
            ].iloc[0]
            self.assertEqual(first["market_cap"], 1_000_000.0)
            self.assertEqual(first["circulating_market_cap"], 800_000.0)
            self.assertEqual(first["book_to_market"], 0.5)

            resumed_session = _FakeTushareSession()
            resumed = fetch_daily_basic_context(
                start="20260102",
                end="20260105",
                output_dir=destination,
                resume=True,
                max_requests_per_minute=0,
                session=resumed_session,
            )

            self.assertTrue(resumed["ok"])
            self.assertEqual(resumed["fetched_date_count"], 0)
            self.assertEqual(resumed["reused_date_count"], 2)
            self.assertEqual(resumed_session.daily_calls, [])
            manifest = json.loads(
                (destination / "_context_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "complete")
            self.assertNotIn("token", json.dumps(manifest).lower())

    def test_partial_provider_failure_is_auditable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "context"
            result = fetch_daily_basic_context(
                start="20260102",
                end="20260105",
                output_dir=destination,
                max_requests_per_minute=0,
                session=_FakeTushareSession(fail_date="20260105"),
            )

            self.assertFalse(result["ok"])
            self.assertEqual(result["failed_dates"], ["20260105"])
            manifest = json.loads(
                (destination / "_context_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "partial")
            self.assertIn("20260105", manifest["failed_dates"])
            with self.assertRaisesRegex(ValueError, "partial"):
                load_research_context_file(destination)

    def test_resume_rejects_a_different_requested_range(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "context"
            fetch_daily_basic_context(
                start="20260102",
                end="20260105",
                output_dir=destination,
                max_requests_per_minute=0,
                session=_FakeTushareSession(),
            )

            with self.assertRaisesRegex(ValueError, "does not match"):
                fetch_daily_basic_context(
                    start="20260102",
                    end="20260106",
                    output_dir=destination,
                    resume=True,
                    max_requests_per_minute=0,
                    session=_FakeTushareSession(),
                )


if __name__ == "__main__":
    unittest.main()
