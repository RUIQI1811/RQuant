"""
Run a portfolio-level strategy backtest.

Edit the settings below, then run:
    python backtest_portfolio.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline.portfolio_backtest import run_portfolio_backtest


# Example: 100000, brick, signal_close, commission 0.8 wan, hold 1 day.
INITIAL_CASH = 100000.0
STRATEGY = "brick"  # "b1" or "brick"
BUY_MODE = "signal_close"  # "signal_close" or "next_open"
COMMISSION_WAN = 0.8  # 0.8 means 0.008%
HOLD_DAYS = 1

BACKTEST_START = "2020-10-01"
BACKTEST_END = "2026-06-05"
BACKTEST_OUTPUT = "data/portfolio_backtest_manual"


def main() -> None:
    result = run_portfolio_backtest(
        initial_cash=INITIAL_CASH,
        strategy=STRATEGY,
        buy_mode=BUY_MODE,
        hold_days=HOLD_DAYS,
        commission_wan=COMMISSION_WAN,
        start_date=BACKTEST_START,
        end_date=BACKTEST_END,
        output_dir=BACKTEST_OUTPUT,
    )
    summary = result["result"].summary
    print("\nPortfolio backtest summary")
    print(f"date range: {BACKTEST_START} to {BACKTEST_END}")
    print(f"strategy: {summary['strategy']}")
    print(f"buy mode: {summary['buy_mode']}")
    print(f"hold days: {summary['hold_days']}")
    print(f"initial cash: {summary['initial_cash']:.2f}")
    print(f"final cash: {summary['final_cash']:.2f}")
    print(f"total return: {summary['total_return']:.2%}")
    print(f"max drawdown: {summary['max_drawdown']:.2%}")
    if summary["annualized_volatility"] is None:
        print("annualized volatility: n/a")
    else:
        print(f"annualized volatility: {summary['annualized_volatility']:.2%}")
    if summary["sharpe_ratio"] is None:
        print("sharpe ratio: n/a")
    else:
        print(f"sharpe ratio: {summary['sharpe_ratio']:.2f}")
    print(f"trade count: {summary['trade_count']}")
    print(f"trades: {result['trades_path']}")
    print(f"equity curve csv: {result['equity_curve_path']}")
    print(f"equity curve html: {result['equity_curve_html_path']}")
    print(f"summary: {result['summary_path']}")


if __name__ == "__main__":
    main()
