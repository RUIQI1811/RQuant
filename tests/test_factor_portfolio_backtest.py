import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backtest.factor_portfolio as factor_portfolio_backtest
from backtest.factor_portfolio import signal_frame_to_picks
from factors.ensemble import RankEnsembleConfig


class FactorPortfolioBacktestTest(unittest.TestCase):
    def test_rank_ensemble_end_to_end_writes_signals_and_portfolio_outputs(self):
        dates = pd.bdate_range("2025-01-02", periods=90)
        day = np.arange(len(dates), dtype=float)
        config = RankEnsembleConfig(
            factors=("alpha_013", "alpha_040"),
            weights=(0.4, 0.6),
            top_n=1,
            factor_lag_days=1,
            min_universe=2,
            min_listing_days=10,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "raw"
            output_dir = root / "output"
            data_dir.mkdir()
            for index, symbol in enumerate(("000001", "000002", "000003", "000004")):
                close = (
                    10.0
                    + index * 3.0
                    + day * (0.015 + index * 0.002)
                    + np.sin(day / (3.0 + index)) * (0.15 + index * 0.03)
                )
                open_price = close * (1.0 + 0.001 * np.cos(day / (2.0 + index)))
                high = np.maximum(open_price, close) + 0.12
                low = np.minimum(open_price, close) - 0.12
                volume = (
                    1_000_000
                    + index * 80_000
                    + np.cos(day / (4.0 + index)) * 100_000
                    + day * (1000 + index * 200)
                )
                pd.DataFrame(
                    {
                        "date": dates,
                        "open": open_price,
                        "close": close,
                        "high": high,
                        "low": low,
                        "volume": volume,
                    }
                ).to_csv(data_dir / f"{symbol}.csv", index=False)

            outputs = factor_portfolio_backtest.run_rank_ensemble_portfolio_backtest(
                data_dir=data_dir,
                metadata_path=None,
                output_dir=output_dir,
                start_date=str(dates[35].date()),
                end_date=str(dates[-2].date()),
                selection_config=config,
                initial_cash=1_000_000,
                hold_days=5,
                commission_wan=0.8,
                lot_size=100,
            )

            self.assertGreater(outputs["result"].summary["signal_count"], 0)
            self.assertEqual(
                outputs["result"].summary["signal_execution_timing"],
                "signal date + 1 trading day at open",
            )
            self.assertTrue((output_dir / "factor_signals" / "signals.csv").exists())
            self.assertTrue((output_dir / "factor_signals" / "selections.csv").exists())
            self.assertTrue((output_dir / "portfolio_summary.json").exists())
            self.assertTrue((output_dir / "equity_curve.csv").exists())

    def test_rank_ensemble_runner_routes_signals_into_fixed_sleeve_portfolio(self):
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
                    "score": [0.9],
                }
            )
        )
        portfolio_result = SimpleNamespace(summary={})
        config = RankEnsembleConfig(
            factors=("alpha_013", "alpha_040"),
            weights=(0.25, 0.75),
            top_n=2,
            min_universe=2,
        )

        with tempfile.TemporaryDirectory() as output_dir:
            with (
                patch.object(factor_portfolio_backtest, "load_raw_data", return_value=raw_data),
                patch.object(factor_portfolio_backtest, "build_alpha101_panels", return_value=panels),
                patch.object(
                    factor_portfolio_backtest,
                    "build_alpha101_rank_ensemble_frame",
                    return_value=(factor_frame, filter_status),
                ),
                patch.object(
                    factor_portfolio_backtest,
                    "rank_factor_ensemble",
                    return_value=signal_result,
                ),
                patch.object(
                    factor_portfolio_backtest,
                    "write_rank_ensemble_reports",
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
                outputs = factor_portfolio_backtest.run_rank_ensemble_portfolio_backtest(
                    output_dir=output_dir,
                    start_date="2026-01-02",
                    end_date="2026-01-03",
                    selection_config=config,
                    hold_days=20,
                )

        self.assertIs(outputs["result"], portfolio_result)
        settings = portfolio_runner.call_args.kwargs["settings"]
        self.assertEqual(settings.max_positions, 2)
        self.assertAlmostEqual(settings.position_pct, 1 / 40)
        self.assertEqual(portfolio_result.summary["factors"], ["alpha_013", "alpha_040"])
        self.assertEqual(
            portfolio_result.summary["factor_weights"],
            {"alpha_013": 0.25, "alpha_040": 0.75},
        )

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

    def test_signal_pick_limit_is_applied_after_score_ranking(self):
        signals = pd.DataFrame(
            {
                "date": ["2026-01-02"] * 3,
                "symbol": [1, 2, 3],
                "signal_type": ["buy"] * 3,
                "score": [0.8, 1.0, 0.9],
            }
        )

        picks = signal_frame_to_picks(signals, max_positions=2)

        self.assertEqual(picks, {pd.Timestamp("2026-01-02"): ["000002", "000003"]})

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
