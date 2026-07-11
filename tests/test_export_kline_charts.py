import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from dashboard import export_kline_charts


class FakeFigure:
    def __init__(self, content=b"jpeg-bytes"):
        self.content = content

    def write_image(self, path, **kwargs):
        Path(path).write_bytes(self.content)


class ChartExportTest(unittest.TestCase):
    def _project(self, root: Path, codes=("000001",)):
        candidates = root / "candidates.json"
        raw = root / "raw"
        output = root / "kline"
        raw.mkdir()
        candidates.write_text(
            json.dumps(
                {
                    "pick_date": "2026-01-03",
                    "candidates": [{"code": code} for code in codes],
                }
            ),
            encoding="utf-8",
        )
        for code in codes:
            pd.DataFrame(
                {
                    "date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
                    "open": [10, 11, 12, 13],
                    "high": [11, 12, 13, 14],
                    "low": [9, 10, 11, 12],
                    "close": [10, 11, 12, 13],
                    "volume": [100, 110, 120, 130],
                }
            ).to_csv(raw / f"{code}.csv", index=False)
        return candidates, raw, output

    def test_export_strictly_truncates_at_candidate_pick_date_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidates, raw, output = self._project(root)
            observed = {}

            def make_chart(frame, code, **kwargs):
                observed[code] = frame.copy()
                return FakeFigure()

            with (
                patch.object(export_kline_charts, "make_daily_chart", side_effect=make_chart),
                redirect_stdout(StringIO()),
            ):
                result = export_kline_charts.run_export(
                    candidates_path=candidates,
                    raw_dir=raw,
                    output_dir=output,
                )

            manifest = json.loads(
                Path(result["manifest_path"]).read_text(encoding="utf-8")
            )
            chart = output / "2026-01-03/000001_day.jpg"

            self.assertTrue(result["ok"])
            self.assertEqual(observed["000001"]["date"].max(), pd.Timestamp("2026-01-03"))
            self.assertNotIn(pd.Timestamp("2026-01-04"), observed["000001"]["date"].tolist())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["results"]["000001"]["chart_end_date"], "2026-01-03")
            self.assertTrue(chart.is_file())
            self.assertFalse(any(chart.parent.glob(".*.tmp")))

    def test_resume_reuses_only_signature_and_hash_matching_chart(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidates, raw, output = self._project(root)
            with (
                patch.object(
                    export_kline_charts,
                    "make_daily_chart",
                    return_value=FakeFigure(),
                ),
                redirect_stdout(StringIO()),
            ):
                export_kline_charts.run_export(
                    candidates_path=candidates,
                    raw_dir=raw,
                    output_dir=output,
                )

            with (
                patch.object(export_kline_charts, "make_daily_chart") as renderer,
                redirect_stdout(StringIO()),
            ):
                resumed = export_kline_charts.run_export(
                    candidates_path=candidates,
                    raw_dir=raw,
                    output_dir=output,
                    resume=True,
                )

            renderer.assert_not_called()
            self.assertEqual(resumed["reused_count"], 1)
            self.assertEqual(resumed["rendered_count"], 0)

    def test_missing_pick_date_bar_is_partial_and_never_exports_that_symbol(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidates, raw, output = self._project(root, codes=("000001", "000002"))
            frame = pd.read_csv(raw / "000002.csv")
            frame = frame[frame["date"] != "2026-01-03"]
            frame.to_csv(raw / "000002.csv", index=False)
            with (
                patch.object(
                    export_kline_charts,
                    "make_daily_chart",
                    return_value=FakeFigure(),
                ) as renderer,
                redirect_stdout(StringIO()),
            ):
                result = export_kline_charts.run_export(
                    candidates_path=candidates,
                    raw_dir=raw,
                    output_dir=output,
                )

            self.assertFalse(result["ok"])
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["failed_codes"], ["000002"])
            self.assertEqual(renderer.call_count, 1)
            self.assertFalse((output / "2026-01-03/000002_day.jpg").exists())

    def test_main_returns_two_for_partial_export(self):
        with (
            patch.object(
                export_kline_charts,
                "run_export",
                return_value={"ok": False},
            ),
            redirect_stdout(StringIO()),
            redirect_stderr(StringIO()),
        ):
            exit_code = export_kline_charts.main([])

        self.assertEqual(exit_code, 2)

    def test_invalid_candidate_codes_fail_before_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidates = root / "candidates.json"
            candidates.write_text(
                json.dumps(
                    {
                        "pick_date": "2026-01-03",
                        "candidates": [{"code": "bad"}],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "six-digit symbol"):
                export_kline_charts.run_export(candidates_path=candidates)


if __name__ == "__main__":
    unittest.main()
