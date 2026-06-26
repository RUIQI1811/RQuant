"""
Run selector signal return backtests.

Edit the settings below, then run:
    python backtest_signal_returns.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline.signal_returns import run_signal_returns


# Edit these values when you want to run a backtest.
BACKTEST_START = "2024-10-20"
BACKTEST_END = "2026-06-05"
BACKTEST_OUTPUT = "data/backtest_manual"
BACKTEST_HORIZONS = (1, 5, 10, 30)
BACKTEST_STRATEGIES = ("brick",)  # Use ("b1",), ("brick",), or ("b1", "brick").
BACKTEST_BUY_MODE = "signal_close"  # Use "signal_close" or "next_open".


def main() -> None:
    result = run_signal_returns(
        start_date=BACKTEST_START,
        end_date=BACKTEST_END,
        output_dir=BACKTEST_OUTPUT,
        horizons=BACKTEST_HORIZONS,
        strategies=BACKTEST_STRATEGIES,
        buy_mode=BACKTEST_BUY_MODE,
    )
    summary = result["summary"]
    print("\nSignal return summary")
    print(f"date range: {summary['start_date']} to {summary['end_date']}")
    print(f"buy mode: {summary['buy_mode']}")
    print(f"signals: {summary['total_signals']}")
    for key, metrics in summary["metrics"].items():
        mean = metrics["mean_return"]
        median = metrics["median_return"]
        win_rate = metrics["win_rate"]
        if mean is None or median is None or win_rate is None:
            print(f"{key}: count=0")
        else:
            print(
                f"{key}: count={metrics['count']} "
                f"mean={mean:.4%} median={median:.4%} win_rate={win_rate:.2%}"
            )
    print(f"csv: {result['csv_path']}")
    print(f"summary: {result['summary_path']}")
    print(f"summary_csv: {result['summary_csv_path']}")


if __name__ == "__main__":
    main()
