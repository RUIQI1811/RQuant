import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from market.fetch_benchmark import fetch_benchmark_index


class _FakeIndexSession:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[dict[str, str]] = []

    def index_daily(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("index provider unavailable")
        return pd.DataFrame(
            {
                "ts_code": ["000300.SH", "000300.SH"],
                "trade_date": ["20260105", "20260102"],
                "open": [4000.0, 3900.0],
                "close": [4010.0, 3920.0],
                "high": [4020.0, 3930.0],
                "low": [3990.0, 3890.0],
                "pre_close": [3995.0, 3880.0],
                "pct_chg": [0.4, 1.0],
                "vol": [100.0, 90.0],
                "amount": [1000.0, 900.0],
            }
        )


class FetchBenchmarkTest(unittest.TestCase):
    def test_fetches_validates_and_resumes_complete_index_range(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "benchmark.csv"
            session = _FakeIndexSession()
            result = fetch_benchmark_index(
                start="20260102",
                end="20260105",
                output_file=output,
                session=session,
            )

            self.assertTrue(result["ok"])
            self.assertFalse(result["reused"])
            self.assertEqual(result["row_count"], 2)
            frame = pd.read_csv(output)
            self.assertEqual(frame["date"].tolist(), ["2026-01-02", "2026-01-05"])
            self.assertEqual(frame["index_code"].unique().tolist(), ["000300.SH"])
            manifest_path = Path(result["manifest_path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertNotIn("token", json.dumps(manifest).lower())

            resumed_session = _FakeIndexSession()
            resumed = fetch_benchmark_index(
                start="20260102",
                end="20260105",
                output_file=output,
                resume=True,
                session=resumed_session,
            )
            self.assertTrue(resumed["ok"])
            self.assertTrue(resumed["reused"])
            self.assertEqual(resumed_session.calls, [])

    def test_provider_failure_writes_partial_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "benchmark.csv"
            result = fetch_benchmark_index(
                start="20260102",
                end="20260105",
                output_file=output,
                session=_FakeIndexSession(fail=True),
            )

            self.assertFalse(result["ok"])
            self.assertFalse(output.exists())
            manifest = json.loads(
                Path(result["manifest_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "partial")
            self.assertIn("provider unavailable", manifest["error"])

    def test_resume_rejects_different_index_code(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "benchmark.csv"
            fetch_benchmark_index(
                start="20260102",
                end="20260105",
                output_file=output,
                session=_FakeIndexSession(),
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                fetch_benchmark_index(
                    start="20260102",
                    end="20260105",
                    index_code="000001.SH",
                    output_file=output,
                    resume=True,
                    session=_FakeIndexSession(),
                )


if __name__ == "__main__":
    unittest.main()
