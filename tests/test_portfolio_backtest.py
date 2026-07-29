import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest import portfolio as portfolio_backtest
from backtest.performance import annualized_return, yearly_return_rows
from backtest.portfolio import (
    FeeModel,
    PortfolioSettings,
    calculate_trade_return,
    build_equity_curve_rows,
    calculate_risk_metrics,
    run_realistic_portfolio_from_prepared,
    run_staggered_cohort_portfolio_from_prepared,
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


def _tradeable_frame(opens, closes, *, limit_up_dates=None):
    df = _frame(opens, closes)
    df["is_tradeable"] = True
    df["is_limit_up"] = False
    df["is_limit_down"] = False
    df["turnover_n"] = 1000000.0
    for date in limit_up_dates or ():
        df.loc[pd.Timestamp(date), "is_limit_up"] = True
    return df


class PortfolioBacktestTest(unittest.TestCase):
    def test_trading_bars_held_uses_calendar_positions(self):
        dates = list(pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-07"]))
        date_positions = {date: position for position, date in enumerate(dates)}

        self.assertEqual(
            portfolio_backtest._trading_bars_held(
                date_positions,
                pd.Timestamp("2026-01-02"),
                pd.Timestamp("2026-01-07"),
            ),
            2,
        )

    def test_latest_close_on_or_before_uses_index_without_requiring_exact_date(self):
        frame = pd.DataFrame(
            {"close": [10.0, 12.0]},
            index=pd.to_datetime(["2026-01-02", "2026-01-05"]),
        )

        self.assertEqual(
            portfolio_backtest._latest_close_on_or_before(
                frame,
                pd.Timestamp("2026-01-04"),
            ),
            10.0,
        )
        self.assertIsNone(
            portfolio_backtest._latest_close_on_or_before(
                frame,
                pd.Timestamp("2026-01-01"),
            )
        )

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

        metrics = calculate_risk_metrics(
            equity_rows=equity_rows,
            trades=trades,
            initial_cash=100000.0,
        )

        self.assertAlmostEqual(metrics["max_drawdown"], 0.25)
        self.assertGreater(metrics["annualized_volatility"], 0)
        self.assertAlmostEqual(
            metrics["sharpe_ratio"],
            metrics["annualized_return_mean"] / metrics["annualized_volatility"],
        )
        self.assertAlmostEqual(
            metrics["overall_annualized_return"],
            annualized_return(-0.01, 4),
        )
        self.assertEqual(metrics["year_count"], 1)

    def test_yearly_returns_keep_prior_year_end_as_next_year_baseline(self):
        rows = yearly_return_rows(
            [
                {"date": "2025-06-30", "total_value": 100.0},
                {"date": "2025-12-31", "total_value": 110.0},
                {"date": "2026-01-02", "total_value": 121.0},
                {"date": "2026-12-31", "total_value": 133.1},
            ],
            initial_cash=100.0,
        )

        self.assertEqual([row["year"] for row in rows], [2025, 2026])
        self.assertAlmostEqual(rows[0]["total_return"], 0.10)
        self.assertTrue(rows[0]["is_partial_year"])
        self.assertAlmostEqual(rows[1]["start_equity"], 110.0)
        self.assertAlmostEqual(rows[1]["total_return"], 0.21)
        self.assertFalse(rows[1]["is_partial_year"])
        self.assertAlmostEqual(
            rows[1]["annualized_return"],
            annualized_return(0.21, 2),
        )

    def test_realistic_portfolio_buys_next_day_and_sells_after_hold_days(self):
        prepared = {
            "000001": _tradeable_frame([10, 11, 12, 13], [10, 11, 12, 13]),
        }
        picks_by_date = {pd.Timestamp("2026-01-01"): ["000001"]}
        settings = PortfolioSettings(
            initial_cash=100000.0,
            strategy="brick",
            buy_mode="next_open",
            hold_days=1,
            fee_model=FeeModel(commission_rate=0.0, stamp_tax_rate=0.0, transfer_fee_rate=0.0),
            max_positions=1,
            position_pct=1.0,
        )

        result = run_realistic_portfolio_from_prepared(
            prepared=prepared,
            picks_by_date=picks_by_date,
            settings=settings,
            start_date="2026-01-01",
            end_date="2026-01-04",
        )

        filled = [order for order in result.orders if order["status"] == "filled"]
        self.assertEqual([order["side"] for order in filled], ["buy", "sell"])
        self.assertEqual(filled[0]["date"], "2026-01-02")
        self.assertEqual(filled[1]["date"], "2026-01-03")
        self.assertEqual(len(result.trades), 1)
        self.assertAlmostEqual(result.trades[0]["return"], 12 / 11 - 1)

    def test_realistic_portfolio_blocks_limit_up_buys(self):
        prepared = {
            "000001": _tradeable_frame(
                [10, 11, 12],
                [10, 11, 12],
                limit_up_dates=("2026-01-02",),
            ),
        }
        picks_by_date = {pd.Timestamp("2026-01-01"): ["000001"]}
        settings = PortfolioSettings(
            initial_cash=100000.0,
            strategy="brick",
            buy_mode="next_open",
            hold_days=1,
            fee_model=FeeModel(commission_rate=0.0, stamp_tax_rate=0.0, transfer_fee_rate=0.0),
        )

        result = run_realistic_portfolio_from_prepared(
            prepared=prepared,
            picks_by_date=picks_by_date,
            settings=settings,
            start_date="2026-01-01",
            end_date="2026-01-03",
        )

        self.assertEqual(result.trades, [])
        self.assertEqual(result.orders[0]["status"], "blocked")
        self.assertEqual(result.orders[0]["reason"], "limit_up")

    def test_realistic_portfolio_preserves_ranked_signal_priority(self):
        prepared = {
            "000001": _tradeable_frame([10, 10, 10], [10, 10, 10]),
            "000002": _tradeable_frame([20, 20, 20], [20, 20, 20]),
        }
        settings = PortfolioSettings(
            initial_cash=100000.0,
            strategy="factor_rank",
            buy_mode="next_open",
            hold_days=2,
            fee_model=FeeModel(commission_rate=0.0, stamp_tax_rate=0.0, transfer_fee_rate=0.0),
            max_positions=1,
            position_pct=1.0,
        )

        result = run_realistic_portfolio_from_prepared(
            prepared=prepared,
            picks_by_date={pd.Timestamp("2026-01-01"): ["000002", "000001"]},
            settings=settings,
            start_date="2026-01-01",
            end_date="2026-01-03",
        )

        filled_buy = next(
            order for order in result.orders if order["side"] == "buy" and order["status"] == "filled"
        )
        self.assertEqual(filled_buy["code"], "000002")

    def test_staggered_portfolio_divides_capital_into_fixed_daily_cohorts(self):
        prepared = {
            "000001": _tradeable_frame([10] * 6, [10] * 6),
            "000002": _tradeable_frame([10] * 6, [10] * 6),
        }
        picks = {
            pd.Timestamp("2026-01-01"): ["000001", "000002"],
            pd.Timestamp("2026-01-02"): ["000001", "000002"],
            pd.Timestamp("2026-01-03"): ["000001", "000002"],
            pd.Timestamp("2026-01-04"): ["000001", "000002"],
        }
        settings = PortfolioSettings(
            initial_cash=4000.0,
            strategy="factor_rank",
            buy_mode="next_open",
            hold_days=2,
            fee_model=FeeModel(commission_rate=0.0, stamp_tax_rate=0.0, transfer_fee_rate=0.0),
            lot_size=100,
        )

        with patch.object(
            portfolio_backtest,
            "tqdm",
            side_effect=lambda dates, **kwargs: dates,
        ) as progress:
            result = run_staggered_cohort_portfolio_from_prepared(
                prepared=prepared,
                picks_by_date=picks,
                settings=settings,
                cohort_count=2,
                start_date="2026-01-01",
                end_date="2026-01-06",
                show_progress=True,
            )

        progress.assert_called_once()
        self.assertEqual(progress.call_args.kwargs["desc"], "组合回测")
        self.assertEqual(progress.call_args.kwargs["unit"], "交易日")
        self.assertFalse(progress.call_args.kwargs["disable"])
        self.assertEqual(result.summary["portfolio_mode"], "strict_staggered_cohorts")
        self.assertEqual(result.summary["cohort_count"], 2)
        self.assertEqual(result.summary["initial_cash_per_cohort"], 2000.0)
        self.assertLessEqual(
            max(row["active_cohort_count"] for row in result.equity_curve),
            2,
        )
        filled_buys = [
            order
            for order in result.orders
            if order["side"] == "buy" and order["status"] == "filled"
        ]
        self.assertEqual(
            sorted({(order["date"], order["cohort_id"]) for order in filled_buys})[:2],
            [("2026-01-02", 2), ("2026-01-03", 1)],
        )
        self.assertTrue(result.trades)

    def test_staggered_portfolio_does_not_reuse_a_blocked_cohort_slot(self):
        prepared = {
            "000001": _tradeable_frame([10] * 5, [10] * 5),
        }
        prepared["000001"].loc[pd.Timestamp("2026-01-04"), "is_limit_down"] = True
        settings = PortfolioSettings(
            initial_cash=2000.0,
            strategy="factor_rank",
            buy_mode="next_open",
            hold_days=2,
            fee_model=FeeModel(commission_rate=0.0, stamp_tax_rate=0.0, transfer_fee_rate=0.0),
            lot_size=100,
        )

        result = run_staggered_cohort_portfolio_from_prepared(
            prepared=prepared,
            picks_by_date={
                pd.Timestamp("2026-01-01"): ["000001"],
                pd.Timestamp("2026-01-03"): ["000001"],
            },
            settings=settings,
            cohort_count=2,
            start_date="2026-01-01",
            end_date="2026-01-05",
        )

        blocked_exit = next(
            order
            for order in result.orders
            if order["side"] == "sell" and order["status"] == "blocked"
        )
        skipped_entry = next(
            order
            for order in result.orders
            if order["side"] == "buy" and order["reason"] == "cohort_exit_blocked"
        )
        self.assertEqual(blocked_exit["cohort_id"], skipped_entry["cohort_id"])
        self.assertLessEqual(
            max(row["active_cohort_count"] for row in result.equity_curve),
            2,
        )


class TopLevelPortfolioBacktestImportTests(unittest.TestCase):
    def test_portfolio_backtest_imports_from_backtest_package(self):
        import backtest.portfolio as portfolio

        self.assertTrue(hasattr(portfolio, "run_portfolio_backtest"))


if __name__ == "__main__":
    unittest.main()
