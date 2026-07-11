import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from agent.base_reviewer import BaseReviewer, ReviewRunIncomplete


class FakeReviewer(BaseReviewer):
    def __init__(self, config, *, failing_codes=()):
        super().__init__(config)
        self.failing_codes = set(failing_codes)
        self.calls = []

    def review_stock(self, code, day_chart, prompt):
        self.calls.append(code)
        if code in self.failing_codes:
            raise RuntimeError("provider unavailable")
        return {
            "code": code,
            "total_score": 4.5 if code == "000001" else 3.0,
            "verdict": "观察",
            "signal_type": "brick",
            "comment": "test",
        }


class BaseReviewerTest(unittest.TestCase):
    def _project(self, root: Path, codes=("000001", "000002")) -> dict:
        prompt = root / "prompt.md"
        candidates = root / "candidates.json"
        charts = root / "charts"
        output = root / "review"
        pick_date = "2026-07-11"
        prompt.write_text("review prompt v1", encoding="utf-8")
        candidates.write_text(
            json.dumps(
                {
                    "pick_date": pick_date,
                    "candidates": [{"code": code} for code in codes],
                }
            ),
            encoding="utf-8",
        )
        date_charts = charts / pick_date
        date_charts.mkdir(parents=True)
        chart_results = {}
        for code in codes:
            chart = date_charts / f"{code}_day.jpg"
            chart.write_bytes(f"chart-{code}".encode())
            chart_results[code] = {
                "chart_end_date": pick_date,
                "output_path": str(chart),
                "output_sha256": hashlib.sha256(chart.read_bytes()).hexdigest(),
            }
        (date_charts / "export_manifest.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "pick_date": pick_date,
                    "candidate_count": len(codes),
                    "success_count": len(codes),
                    "failed_count": 0,
                    "results": chart_results,
                }
            ),
            encoding="utf-8",
        )
        return {
            "prompt_path": prompt,
            "candidates": candidates,
            "kline_dir": charts,
            "output_dir": output,
            "model": "fake-model",
            "retry_models": [],
            "skip_existing": True,
            "suggest_min_score": 4.0,
            "request_delay": 0,
        }

    def test_successful_run_is_atomic_auditable_and_resumable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._project(root)
            first = FakeReviewer(config)
            with redirect_stdout(StringIO()):
                first_manifest = first.run()

            second = FakeReviewer(config)
            with redirect_stdout(StringIO()):
                second_manifest = second.run()

            out_dir = Path(config["output_dir"]) / "2026-07-11"
            suggestion = json.loads(
                (out_dir / "suggestion.json").read_text(encoding="utf-8")
            )
            stored = json.loads((out_dir / "000001.json").read_text(encoding="utf-8"))

            self.assertEqual(first.calls, ["000001", "000002"])
            self.assertEqual(second.calls, [])
            self.assertEqual(first_manifest["status"], "complete")
            self.assertEqual(second_manifest["reused_count"], 2)
            self.assertEqual(suggestion["status"], "complete")
            self.assertEqual(suggestion["review_candidates"][0]["code"], "000001")
            self.assertEqual(suggestion["recommendations"], suggestion["review_candidates"])
            self.assertIn("signature", stored["_review_meta"])
            self.assertFalse(any(out_dir.glob(".*.tmp")))

    def test_corrupt_or_stale_result_is_recomputed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._project(root, codes=("000001",))
            out_dir = Path(config["output_dir"]) / "2026-07-11"
            out_dir.mkdir(parents=True)
            result_path = out_dir / "000001.json"
            result_path.write_text("{broken", encoding="utf-8")

            reviewer = FakeReviewer(config)
            with redirect_stdout(StringIO()):
                reviewer.run()
            self.assertEqual(reviewer.calls, ["000001"])

            Path(config["prompt_path"]).write_text("review prompt v2", encoding="utf-8")
            changed_prompt = FakeReviewer(config)
            with redirect_stdout(StringIO()):
                changed_prompt.run()
            self.assertEqual(changed_prompt.calls, ["000001"])

    def test_partial_failure_persists_manifest_and_exits_incomplete(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._project(root)
            reviewer = FakeReviewer(config, failing_codes=("000002",))

            with redirect_stdout(StringIO()):
                with self.assertRaises(ReviewRunIncomplete) as raised:
                    reviewer.run()

            out_dir = Path(config["output_dir"]) / "2026-07-11"
            manifest = json.loads(
                (out_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            suggestion = json.loads(
                (out_dir / "suggestion.json").read_text(encoding="utf-8")
            )

            self.assertEqual(raised.exception.manifest["status"], "partial")
            self.assertEqual(manifest["failed_codes"], ["000002"])
            self.assertEqual(manifest["success_count"], 1)
            self.assertEqual(suggestion["status"], "partial")
            self.assertEqual(suggestion["failed_codes"], ["000002"])

    def test_failed_force_run_archives_old_result_and_resume_retries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._project(root, codes=("000001",))
            initial = FakeReviewer(config)
            with redirect_stdout(StringIO()):
                initial.run()

            force_config = {**config, "skip_existing": False}
            forced = FakeReviewer(force_config, failing_codes=("000001",))
            with redirect_stdout(StringIO()):
                with self.assertRaises(ReviewRunIncomplete):
                    forced.run()

            out_dir = Path(config["output_dir"]) / "2026-07-11"
            self.assertFalse((out_dir / "000001.json").exists())
            self.assertEqual(len(list((out_dir / ".stale").glob("000001_*.json"))), 1)

            resumed = FakeReviewer(config)
            with redirect_stdout(StringIO()):
                manifest = resumed.run()
            self.assertEqual(resumed.calls, ["000001"])
            self.assertEqual(manifest["reused_count"], 0)
            self.assertEqual(manifest["processed_count"], 1)

    def test_missing_chart_is_failed_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._project(root, codes=("000001",))
            chart = Path(config["kline_dir"]) / "2026-07-11/000001_day.jpg"
            chart.unlink()
            reviewer = FakeReviewer(config)

            with redirect_stdout(StringIO()):
                with self.assertRaises(ReviewRunIncomplete) as raised:
                    reviewer.run()

            self.assertEqual(raised.exception.manifest["status"], "failed")
            self.assertEqual(raised.exception.manifest["failed_codes"], ["000001"])
            self.assertEqual(reviewer.calls, [])

    def test_review_refuses_legacy_charts_without_point_in_time_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._project(root, codes=("000001",))
            manifest = Path(config["kline_dir"]) / "2026-07-11/export_manifest.json"
            manifest.unlink()
            reviewer = FakeReviewer(config)

            with self.assertRaisesRegex(
                ValueError,
                "cannot verify point-in-time charts",
            ):
                reviewer.run()

            self.assertEqual(reviewer.calls, [])

    def test_candidate_codes_are_normalized_and_duplicates_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._project(root, codes=("000001",))
            Path(config["candidates"]).write_text(
                json.dumps(
                    {
                        "pick_date": "2026-07-11",
                        "candidates": [{"code": 1}, {"code": "000001"}],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate candidate code: 000001"):
                FakeReviewer(config).run()


if __name__ == "__main__":
    unittest.main()
