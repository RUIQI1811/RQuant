import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.signal_portfolio import run_signal_portfolio_backtest


class UnifiedSignalPortfolioBacktestTest(unittest.TestCase):
    def test_unified_signal_csv_runs_through_next_open_constrained_portfolio(self):
        dates = pd.bdate_range("2026-01-02", periods=10)
        day = np.arange(len(dates), dtype=float)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "raw"
            output_dir = root / "output"
            signals_path = root / "signals.csv"
            data_dir.mkdir()
            for index, symbol in enumerate(("000001", "000002")):
                close = 10.0 + index * 4.0 + day * (0.05 + index * 0.01)
                pd.DataFrame(
                    {
                        "date": dates,
                        "open": close * 1.001,
                        "close": close,
                        "high": close * 1.01,
                        "low": close * 0.99,
                        "volume": 1_000_000 + day * 1000,
                    }
                ).to_csv(data_dir / f"{symbol}.csv", index=False)
            pd.DataFrame(
                {
                    "date": [dates[0], dates[0], dates[2]],
                    "symbol": ["000001", "000002", "000002"],
                    "signal_type": ["buy", "buy", "buy"],
                    "source": ["model_ridge", "other", "model_ridge"],
                    "score": [0.9, 1.0, 0.8],
                    "weight": [1.0, 1.0, 1.0],
                    "metadata": ["{}", "{}", "{}"],
                }
            ).to_csv(signals_path, index=False)

            outputs = run_signal_portfolio_backtest(
                signals_path=signals_path,
                data_dir=data_dir,
                output_dir=output_dir,
                source="model_ridge",
                start_date=str(dates[0].date()),
                end_date=str(dates[-1].date()),
                initial_cash=200_000,
                hold_days=2,
                commission_wan=0.8,
                stamp_tax_rate=0.0005,
                transfer_fee_rate=0.00001,
                max_positions=1,
                lot_size=100,
            )

            summary = json.loads((output_dir / "portfolio_summary.json").read_text(encoding="utf-8"))
            trades = pd.read_csv(output_dir / "portfolio_trades.csv", dtype={"code": str})
            yearly_returns = pd.read_csv(output_dir / "yearly_returns.csv")

        self.assertEqual(summary["signal_source_filter"], "model_ridge")
        self.assertEqual(summary["signal_count"], 2)
        self.assertEqual(summary["signal_execution_timing"], "signal date + 1 trading day at open")
        self.assertEqual(summary["max_positions_per_cohort"], 1)
        self.assertEqual(summary["stamp_tax_rate"], 0.0005)
        self.assertEqual(summary["transfer_fee_rate"], 0.00001)
        self.assertIn("overall_annualized_return", summary)
        self.assertIn("average_yearly_annualized_return", summary)
        self.assertEqual(yearly_returns["year"].tolist(), [2026])
        self.assertGreater(len(trades), 0)
        self.assertEqual(Path(outputs["equity_curve_html_path"]).name, "equity_curve.html")
        self.assertEqual(Path(outputs["yearly_returns_path"]).name, "yearly_returns.csv")


if __name__ == "__main__":
    unittest.main()
