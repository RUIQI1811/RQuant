import json
import tempfile
import unittest
from pathlib import Path

from reports.research_report import (
    ReportConsistencyError,
    ReportInputError,
    build_research_summary,
    run_research_report,
)


class ResearchReportTest(unittest.TestCase):
    def test_build_research_summary_reads_outputs_and_allows_missing_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            signal_dir = root / "signal"
            portfolio_dir = root / "portfolio"
            signal_dir.mkdir()
            portfolio_dir.mkdir()
            candidates_path = root / "candidates.json"

            (signal_dir / "signal_summary.json").write_text(
                json.dumps(
                    {
                        "run_date": "2026-06-20",
                        "start_date": "2026-01-01",
                        "end_date": "2026-06-01",
                        "horizons": [1, 5],
                        "buy_mode": "signal_close",
                        "total_signals": 12,
                        "metrics": {
                            "return_1d": {
                                "count": 10,
                                "mean_return": 0.01,
                                "median_return": 0.0,
                                "win_rate": 0.6,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            (portfolio_dir / "portfolio_summary.json").write_text(
                json.dumps(
                    {
                        "strategy": "brick",
                        "buy_mode": "signal_close",
                        "hold_days": 1,
                        "initial_cash": 100000,
                        "final_cash": 110000,
                        "total_return": 0.1,
                        "trade_count": 3,
                        "max_drawdown": 0.05,
                        "sharpe_ratio": 1.2,
                    }
                ),
                encoding="utf-8",
            )
            candidates_path.write_text(
                json.dumps(
                    {
                        "run_date": "2026-06-02",
                        "pick_date": "2026-06-01",
                        "candidates": [
                            {"code": "000001", "strategy": "brick"},
                            {"code": "000002", "strategy": "b1"},
                            {"code": "000003", "strategy": "brick"},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            summary = build_research_summary(
                signal_dir=signal_dir,
                portfolio_dir=portfolio_dir,
                candidates_path=candidates_path,
                review_path=root / "missing_review.json",
            )

            self.assertEqual(summary["candidates"]["count"], 3)
            self.assertEqual(summary["candidates"]["by_strategy"], {"brick": 2, "b1": 1})
            self.assertEqual(summary["signal_returns"]["total_signals"], 12)
            self.assertEqual(summary["portfolio"]["total_return"], 0.1)
            self.assertFalse(summary["review"]["exists"])
            self.assertEqual(summary["review"]["recommendation_count"], 0)
            self.assertEqual(summary["validation"]["status"], "warning")
            self.assertTrue(summary["source_fingerprints"]["signal_summary"])

    def test_run_research_report_writes_json_and_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            signal_dir = root / "signal"
            portfolio_dir = root / "portfolio"
            output_dir = root / "reports"
            signal_dir.mkdir()
            portfolio_dir.mkdir()
            candidates_path = root / "candidates.json"
            review_path = root / "suggestion.json"

            (signal_dir / "signal_summary.json").write_text(
                json.dumps({"total_signals": 1, "metrics": {}}),
                encoding="utf-8",
            )
            (portfolio_dir / "portfolio_summary.json").write_text(
                json.dumps({"total_return": 0.25, "max_drawdown": 0.02}),
                encoding="utf-8",
            )
            candidates_path.write_text(
                json.dumps({"pick_date": "2026-06-01", "candidates": []}),
                encoding="utf-8",
            )
            review_path.write_text(
                json.dumps(
                    {
                        "total_reviewed": 1,
                        "recommendations": [
                            {
                                "rank": 1,
                                "code": "000001",
                                "total_score": 4.5,
                                "verdict": "PASS",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = run_research_report(
                signal_dir=signal_dir,
                portfolio_dir=portfolio_dir,
                candidates_path=candidates_path,
                review_path=review_path,
                output_dir=output_dir,
            )

            self.assertTrue(result["json_path"].exists())
            self.assertTrue(result["html_path"].exists())
            written = json.loads(result["json_path"].read_text(encoding="utf-8"))
            self.assertEqual(written["review"]["recommendation_count"], 1)
            self.assertIn("RQuant Research Report", result["html_path"].read_text(encoding="utf-8"))
            self.assertFalse(any(output_dir.glob(".*.tmp")))

    def test_required_missing_or_invalid_inputs_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            signal_dir = root / "signal"
            portfolio_dir = root / "portfolio"
            signal_dir.mkdir()
            portfolio_dir.mkdir()
            candidates = root / "candidates.json"
            candidates.write_text(
                json.dumps({"pick_date": "2026-07-11", "candidates": []}),
                encoding="utf-8",
            )
            (portfolio_dir / "portfolio_summary.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(ReportInputError, "missing required signal summary"):
                build_research_summary(
                    signal_dir=signal_dir,
                    portfolio_dir=portfolio_dir,
                    candidates_path=candidates,
                )

            (signal_dir / "signal_summary.json").write_text("{broken", encoding="utf-8")
            with self.assertRaisesRegex(ReportInputError, "cannot read signal summary"):
                build_research_summary(
                    signal_dir=signal_dir,
                    portfolio_dir=portfolio_dir,
                    candidates_path=candidates,
                )

    def test_inconsistent_artifacts_are_blocked_unless_explicitly_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            signal_dir = root / "signal"
            portfolio_dir = root / "portfolio"
            output_dir = root / "report"
            signal_dir.mkdir()
            portfolio_dir.mkdir()
            candidates = root / "candidates.json"
            review = root / "suggestion.json"
            (signal_dir / "signal_summary.json").write_text(
                json.dumps(
                    {
                        "buy_mode": "signal_close",
                        "start_date": "2026-01-01",
                        "end_date": "2026-06-01",
                    }
                ),
                encoding="utf-8",
            )
            (portfolio_dir / "portfolio_summary.json").write_text(
                json.dumps(
                    {
                        "buy_mode": "next_open",
                        "start_date": "2026-01-01",
                        "end_date": "2026-06-02",
                    }
                ),
                encoding="utf-8",
            )
            candidates.write_text(
                json.dumps({"pick_date": "2026-06-01", "candidates": []}),
                encoding="utf-8",
            )
            review.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "date": "2026-06-02",
                        "review_candidates": [],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ReportConsistencyError) as raised:
                run_research_report(
                    signal_dir=signal_dir,
                    portfolio_dir=portfolio_dir,
                    candidates_path=candidates,
                    review_path=review,
                    output_dir=output_dir,
                )
            self.assertFalse(output_dir.exists())
            self.assertEqual(len(raised.exception.errors), 3)

            result = run_research_report(
                signal_dir=signal_dir,
                portfolio_dir=portfolio_dir,
                candidates_path=candidates,
                review_path=review,
                output_dir=output_dir,
                allow_inconsistent=True,
            )
            self.assertEqual(result["summary"]["validation"]["status"], "error")
            self.assertIn("Artifact Validation", result["html_path"].read_text(encoding="utf-8"))

    def test_partial_review_is_never_presented_as_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            signal_dir = root / "signal"
            portfolio_dir = root / "portfolio"
            signal_dir.mkdir()
            portfolio_dir.mkdir()
            (signal_dir / "signal_summary.json").write_text(
                json.dumps({"buy_mode": "next_open"}),
                encoding="utf-8",
            )
            (portfolio_dir / "portfolio_summary.json").write_text(
                json.dumps({"buy_mode": "next_open"}),
                encoding="utf-8",
            )
            candidates = root / "candidates.json"
            candidates.write_text(
                json.dumps({"pick_date": "2026-07-11", "candidates": []}),
                encoding="utf-8",
            )
            review = root / "suggestion.json"
            review.write_text(
                json.dumps(
                    {
                        "status": "partial",
                        "date": "2026-07-11",
                        "failed_count": 1,
                        "failed_codes": ["000001"],
                        "review_candidates": [],
                    }
                ),
                encoding="utf-8",
            )

            summary = build_research_summary(
                signal_dir=signal_dir,
                portfolio_dir=portfolio_dir,
                candidates_path=candidates,
                review_path=review,
            )

            self.assertEqual(summary["review"]["status"], "partial")
            self.assertIn("review status is partial", summary["validation"]["errors"])


class TopLevelResearchReportImportTests(unittest.TestCase):
    def test_research_report_imports_from_reports_package(self):
        import reports.research_report as research_report

        self.assertTrue(hasattr(research_report, "run_research_report"))


if __name__ == "__main__":
    unittest.main()
