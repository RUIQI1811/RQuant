"""Portfolio backtest orchestration for filter-then-rank factor signals."""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from backtest.portfolio import (
    FeeModel,
    PortfolioSettings,
    run_staggered_cohort_portfolio_from_prepared,
    write_portfolio_backtest_outputs,
)
from factors.alpha101 import build_alpha101_panels
from factors.filter_rank import (
    FilterRankConfig,
    build_filter_rank_frame,
    filter_then_rank,
    write_filter_rank_reports,
)
from market.data import clean_market_data
from strategies.preselect import load_raw_data


DEFAULT_FACTOR_PORTFOLIO_OUTPUT = Path("data") / "portfolio_backtest_alpha077_alpha040"


def signal_frame_to_picks(
    signals: pd.DataFrame,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict[pd.Timestamp, list[str]]:
    """Convert unified buy signals to ordered per-date portfolio candidates."""

    required = {"date", "symbol", "signal_type"}
    missing = required.difference(signals.columns)
    if missing:
        raise ValueError(f"missing signal columns: {', '.join(sorted(missing))}")
    frame = signals.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame = frame.dropna(subset=["date", "symbol"])
    frame = frame.loc[frame["signal_type"].astype(str).str.lower().eq("buy")].copy()
    if start_date:
        frame = frame.loc[frame["date"] >= pd.Timestamp(start_date)]
    if end_date:
        frame = frame.loc[frame["date"] <= pd.Timestamp(end_date)]
    if "score" in frame.columns:
        frame["_score"] = pd.to_numeric(frame["score"], errors="coerce")
        frame = frame.sort_values(
            ["date", "_score"],
            ascending=[True, False],
            na_position="last",
            kind="mergesort",
        )
    else:
        frame = frame.sort_values("date", kind="mergesort")
    frame = frame.drop_duplicates(["date", "symbol"], keep="first")
    return {
        pd.Timestamp(date): daily["symbol"].tolist()
        for date, daily in frame.groupby("date", sort=True)
    }


def run_filter_rank_portfolio_backtest(
    *,
    data_dir: str | Path = "data/raw",
    metadata_path: str | Path | None = "pipeline/stocklist.csv",
    output_dir: str | Path = DEFAULT_FACTOR_PORTFOLIO_OUTPUT,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    selection_config: FilterRankConfig | None = None,
    initial_cash: float = 10000000.0,
    hold_days: int = 20,
    commission_wan: float = 0.8,
    lot_size: int = 100,
    show_progress: bool = False,
) -> dict:
    """Generate factor signals and run the realistic next-open portfolio path."""

    config = selection_config or FilterRankConfig()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if show_progress:
        print(
            "正在准备数据（读取行情、计算因子与生成信号）...",
            file=sys.stderr,
            flush=True,
        )

    raw_data = load_raw_data(data_dir, end_date=None)
    metadata_file = Path(metadata_path) if metadata_path else None
    metadata = pd.read_csv(metadata_file) if metadata_file and metadata_file.exists() else None
    panels = build_alpha101_panels(raw_data, metadata=metadata)
    selection_dates = panels.close.index
    if start_date:
        selection_dates = selection_dates[selection_dates >= pd.Timestamp(start_date)]
    if end_date:
        selection_dates = selection_dates[selection_dates <= pd.Timestamp(end_date)]
    if selection_dates.empty:
        raise ValueError("no trading dates remain after applying start_date/end_date")

    factor_frame, filter_status = build_filter_rank_frame(
        panels,
        config=config,
        dates=selection_dates,
    )
    signal_result = filter_then_rank(
        factor_frame,
        config=config,
        filter_status=filter_status,
    )
    picks = signal_frame_to_picks(
        signal_result.signals,
        start_date=start_date,
        end_date=end_date,
    )
    signal_output_dir = write_filter_rank_reports(
        signal_result,
        destination / "factor_signals",
        config=config,
    )
    del panels
    del factor_frame
    gc.collect()

    prepared = clean_market_data(raw_data)
    del raw_data
    gc.collect()
    settings = PortfolioSettings(
        initial_cash=initial_cash,
        strategy=config.source,
        buy_mode="next_open",
        hold_days=hold_days,
        fee_model=FeeModel.from_commission_wan(commission_wan),
        max_positions=config.selected_rank_count,
        position_pct=1.0 / (hold_days * config.selected_rank_count),
        lot_size=lot_size,
    )
    portfolio_result = run_staggered_cohort_portfolio_from_prepared(
        prepared=prepared,
        picks_by_date=picks,
        settings=settings,
        cohort_count=hold_days,
        start_date=start_date,
        end_date=end_date,
        show_progress=show_progress,
    )
    portfolio_result.summary.update(
        {
            "start_date": start_date or str(selection_dates.min().date()),
            "end_date": end_date or str(selection_dates.max().date()),
            "signal_count": len(signal_result.signals),
            "signal_date_count": len(picks),
            "filter_factor": config.filter_factor,
            "rank_factor": config.rank_factor,
            "filter_top_quantile": config.filter_top_quantile,
            "factor_lag_days": config.factor_lag_days,
            "signal_execution_timing": "signal date + 1 trading day at open",
            "cohort_policy": (
                f"capital split into {hold_days} fixed sleeves; one scheduled sleeve per day"
            ),
            "rank_start": config.rank_start,
            "rank_end": config.resolved_rank_end,
            "names_per_signal": config.selected_rank_count,
            "factor_signal_dir": str(signal_output_dir),
        }
    )
    outputs = write_portfolio_backtest_outputs(portfolio_result, destination)
    outputs["signal_output_dir"] = signal_output_dir
    outputs["signal_result"] = signal_result
    return outputs
