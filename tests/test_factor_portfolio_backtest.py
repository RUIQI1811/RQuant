import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backtest.factor_portfolio as factor_portfolio_backtest
from backtest.factor_portfolio import signal_frame_to_picks


class FactorPortfolioBacktestTest(unittest.TestCase):
    def test_runner_reports_preparation_and_enables_portfolio_progress(self):
        raw_data = {"000001": pd.DataFrame()}
        panels = SimpleNamespace(
            close=pd.DataFrame(index=pd.to_datetime(["2026-01-02", "2026-01-03"]))
        )
        factor_frame = pd.DataFrame({"date": pd.to_datetime(["2026-01-02"])})
        filter_status = pd.DataFrame()
        signal_result = SimpleNamespace(
            signals=pd.DataFrame(
                {
                    "date": ["2026-01-02"],
                    "symbol": ["000001"],
                    "signal_type": ["buy"],
                }
            )
        )
        portfolio_result = SimpleNamespace(summary={})
        stderr = StringIO()

        with tempfile.TemporaryDirectory() as output_dir:
            with (
                patch.object(factor_portfolio_backtest, "load_raw_data", return_value=raw_data),
                patch.object(factor_portfolio_backtest, "build_alpha101_panels", return_value=panels),
                patch.object(
                    factor_portfolio_backtest,
                    "build_filter_rank_frame",
                    return_value=(factor_frame, filter_status),
                ),
                patch.object(factor_portfolio_backtest, "filter_then_rank", return_value=signal_result),
                patch.object(
                    factor_portfolio_backtest,
                    "write_filter_rank_reports",
                    return_value=Path(output_dir) / "factor_signals",
                ),
                patch.object(
                    factor_portfolio_backtest,
                    "signal_frame_to_picks",
                    return_value={pd.Timestamp("2026-01-02"): ["000001"]},
                ),
                patch.object(factor_portfolio_backtest, "clean_market_data", return_value=raw_data),
                patch.object(
                    factor_portfolio_backtest,
                    "run_staggered_cohort_portfolio_from_prepared",
                    return_value=portfolio_result,
                ) as portfolio_runner,
                patch.object(
                    factor_portfolio_backtest,
                    "write_portfolio_backtest_outputs",
                    return_value={"result": portfolio_result},
                ),
            ):
                with redirect_stderr(stderr):
                    outputs = factor_portfolio_backtest.run_filter_rank_portfolio_backtest(
                        output_dir=output_dir,
                        start_date="2026-01-02",
                        end_date="2026-01-03",
                        show_progress=True,
                    )

        self.assertIs(outputs["result"], portfolio_result)
        self.assertIn("正在准备数据", stderr.getvalue())
        self.assertTrue(
            portfolio_runner.call_args.kwargs["show_progress"],
        )

    def test_signal_scores_preserve_factor_rank_priority(self):
        signals = pd.DataFrame(
            {
                "date": ["2026-01-02", "2026-01-02", "2026-01-02", "2026-01-03"],
                "symbol": [1, 2, 1, 3],
                "signal_type": ["buy", "buy", "buy", "sell"],
                "score": [0.8, 1.0, 0.7, 1.0],
            }
        )

        picks = signal_frame_to_picks(signals)

        self.assertEqual(picks, {pd.Timestamp("2026-01-02"): ["000002", "000001"]})

    def test_signal_date_range_is_applied_before_backtest(self):
        signals = pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
                "symbol": [1, 2, 3],
                "signal_type": ["buy", "buy", "buy"],
            }
        )

        picks = signal_frame_to_picks(
            signals,
            start_date="2026-01-02",
            end_date="2026-01-02",
        )

        self.assertEqual(picks, {pd.Timestamp("2026-01-02"): ["000002"]})


class TopLevelFactorPortfolioImportTests(unittest.TestCase):
    def test_factor_portfolio_imports_from_backtest_package(self):
        import backtest.factor_portfolio as factor_portfolio

        self.assertTrue(hasattr(factor_portfolio, "run_filter_rank_portfolio_backtest"))


if __name__ == "__main__":
    unittest.main()
