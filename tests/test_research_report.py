import json
import tempfile
import unittest
from pathlib import Path

from pipeline.research_report import build_research_summary, run_research_report


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


class TopLevelResearchReportImportTests(unittest.TestCase):
    def test_research_report_imports_from_reports_package(self):
        import reports.research_report as research_report

        self.assertTrue(hasattr(research_report, "run_research_report"))


if __name__ == "__main__":
    unittest.main()
