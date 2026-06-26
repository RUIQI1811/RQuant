import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.portfolio_backtest import (
    FeeModel,
    PortfolioSettings,
    calculate_trade_return,
    build_equity_curve_rows,
    calculate_risk_metrics,
    run_portfolio_from_prepared,
)


def _frame(opens, closes):
    dates = pd.date_range("2026-01-01", periods=len(closes), freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "open": opens,
            "high": closes,
            "low": opens,
            "close": closes,
            "volume": [1000] * len(closes),
        }
    ).set_index("date", drop=False)


class PortfolioBacktestTest(unittest.TestCase):
    def test_calculate_trade_return_deducts_buy_sell_stamp_and_transfer_costs(self):
        fee_model = FeeModel(
            commission_rate=0.00008,
            stamp_tax_rate=0.0005,
            transfer_fee_rate=0.00001,
        )

        trade_return = calculate_trade_return(
            entry_price=100.0,
            exit_price=110.0,
            fee_model=fee_model,
        )

        buy_cost_rate = 0.00008 + 0.00001
        sell_cost_rate = 0.00008 + 0.0005 + 0.00001
        expected = (110.0 / 100.0) / (1 + buy_cost_rate) * (1 - sell_cost_rate) - 1
        self.assertAlmostEqual(trade_return, expected)

    def test_run_portfolio_from_prepared_compounds_daily_equal_weight_baskets(self):
        prepared = {
            "000001": _frame([10, 11, 12], [10, 12, 14]),
            "000002": _frame([20, 19, 18], [20, 18, 16]),
        }
        picks_by_date = {
            pd.Timestamp("2026-01-01"): ["000001", "000002"],
            pd.Timestamp("2026-01-02"): ["000001"],
        }
        settings = PortfolioSettings(
            initial_cash=100000.0,
            strategy="brick",
            buy_mode="signal_close",
            hold_days=1,
            fee_model=FeeModel(
                commission_rate=0.0,
                stamp_tax_rate=0.0,
                transfer_fee_rate=0.0,
            ),
        )

        result = run_portfolio_from_prepared(
            prepared=prepared,
            picks_by_date=picks_by_date,
            settings=settings,
        )

        self.assertEqual(len(result.trades), 2)
        self.assertAlmostEqual(result.trades[0]["basket_return"], 0.05)
        self.assertAlmostEqual(result.trades[0]["end_cash"], 105000.0)
        self.assertAlmostEqual(result.trades[1]["basket_return"], 14 / 12 - 1)
        self.assertAlmostEqual(result.final_cash, 122500.0)

    def test_build_equity_curve_rows_starts_with_initial_cash_and_tracks_returns(self):
        trades = [
            {"signal_date": "2026-01-01", "end_cash": 105000.0},
            {"signal_date": "2026-01-02", "end_cash": 122500.0},
        ]

        rows = build_equity_curve_rows(
            initial_cash=100000.0,
            trades=trades,
            start_date="2026-01-01",
        )

        self.assertEqual(rows[0]["date"], "2026-01-01")
        self.assertAlmostEqual(rows[0]["cash"], 100000.0)
        self.assertAlmostEqual(rows[0]["total_return"], 0.0)
        self.assertEqual(rows[1]["date"], "2026-01-01")
        self.assertAlmostEqual(rows[1]["cash"], 105000.0)
        self.assertAlmostEqual(rows[1]["total_return"], 0.05)
        self.assertEqual(rows[2]["date"], "2026-01-02")
        self.assertAlmostEqual(rows[2]["cash"], 122500.0)
        self.assertAlmostEqual(rows[2]["total_return"], 0.225)

    def test_calculate_risk_metrics_reports_drawdown_volatility_and_sharpe(self):
        equity_rows = [
            {"date": "2026-01-01", "cash": 100000.0, "total_return": 0.0},
            {"date": "2026-01-02", "cash": 120000.0, "total_return": 0.2},
            {"date": "2026-01-03", "cash": 90000.0, "total_return": -0.1},
            {"date": "2026-01-04", "cash": 99000.0, "total_return": -0.01},
        ]
        trades = [
            {"basket_return": 0.20},
            {"basket_return": -0.25},
            {"basket_return": 0.10},
        ]

        metrics = calculate_risk_metrics(equity_rows=equity_rows, trades=trades)

        self.assertAlmostEqual(metrics["max_drawdown"], 0.25)
        self.assertGreater(metrics["annualized_volatility"], 0)
        self.assertAlmostEqual(
            metrics["sharpe_ratio"],
            metrics["annualized_return_mean"] / metrics["annualized_volatility"],
        )


if __name__ == "__main__":
    unittest.main()
