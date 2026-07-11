import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import run_all


class RunAllTest(unittest.TestCase):
    def test_pipeline_uses_unified_cli_and_respects_step_range(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with (
                patch.object(run_all, "_run") as runner,
                patch.object(run_all, "_print_review_candidates") as printer,
                redirect_stdout(StringIO()),
            ):
                run_all.run_pipeline(
                    start_from=1,
                    stop_after=3,
                    skip_fetch=True,
                    root=root,
                    python="/chosen/python",
                )

        self.assertEqual(runner.call_count, 2)
        commands = [call.args[0].command for call in runner.call_args_list]
        self.assertEqual(commands[0][-1], "preselect")
        self.assertEqual(commands[1][1], str(root / "dashboard/export_kline_charts.py"))
        self.assertEqual(commands[1][-1], "--resume")
        self.assertTrue(all(command[0] == "/chosen/python" for command in commands))
        printer.assert_not_called()

    def test_nonzero_subprocess_stops_with_exact_exit_code(self):
        step = run_all.PipelineStep(2, "test", ("python", "task.py"))
        with (
            patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 7)),
            redirect_stdout(StringIO()),
        ):
            with self.assertRaises(run_all.PipelineStepError) as raised:
                run_all._run(step)

        self.assertEqual(raised.exception.return_code, 7)
        self.assertEqual(raised.exception.step, step)

    def test_missing_final_artifact_is_a_pipeline_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(run_all.PipelineResultError, "missing latest candidates"):
                run_all._print_review_candidates(root=Path(temp_dir))

    def test_review_output_preserves_symbol_and_never_calls_it_a_buy_recommendation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidates_dir = root / "data/candidates"
            review_dir = root / "data/review/2026-07-11"
            candidates_dir.mkdir(parents=True)
            review_dir.mkdir(parents=True)
            (candidates_dir / "candidates_latest.json").write_text(
                json.dumps({"pick_date": "2026-07-11"}),
                encoding="utf-8",
            )
            (review_dir / "suggestion.json").write_text(
                json.dumps(
                    {
                        "recommendations": [
                            {
                                "rank": 1,
                                "code": 1,
                                "total_score": 4.5,
                                "signal_type": "brick",
                                "verdict": "观察",
                                "comment": "test",
                            }
                        ],
                        "min_score_threshold": 4.0,
                        "total_reviewed": 1,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            output = StringIO()
            with redirect_stdout(output):
                run_all._print_review_candidates(root=root)

        rendered = output.getvalue()
        self.assertIn("000001", rendered)
        self.assertIn("研究辅助", rendered)
        self.assertNotIn("推荐购买", rendered)

    def test_main_returns_nonzero_when_final_result_is_invalid(self):
        with (
            patch.object(
                run_all,
                "run_pipeline",
                side_effect=run_all.PipelineResultError("bad result"),
            ),
            redirect_stderr(StringIO()),
        ):
            exit_code = run_all.main(["--start-from", "5"])

        self.assertEqual(exit_code, 1)

    def test_partial_review_summary_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidates_dir = root / "data/candidates"
            review_dir = root / "data/review/2026-07-11"
            candidates_dir.mkdir(parents=True)
            review_dir.mkdir(parents=True)
            (candidates_dir / "candidates_latest.json").write_text(
                json.dumps({"pick_date": "2026-07-11"}),
                encoding="utf-8",
            )
            (review_dir / "suggestion.json").write_text(
                json.dumps(
                    {
                        "status": "partial",
                        "date": "2026-07-11",
                        "review_candidates": [],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                run_all.PipelineResultError,
                "review summary is partial",
            ):
                run_all._print_review_candidates(root=root)

    def test_skip_review_omits_external_review_and_stale_result_display(self):
        with (
            patch.object(run_all, "_run") as runner,
            patch.object(run_all, "_print_review_candidates") as printer,
            redirect_stdout(StringIO()),
        ):
            run_all.run_pipeline(start_from=4, stop_after=5, skip_review=True)

        runner.assert_not_called()
        printer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
