"""Run the constrained portfolio engine from a unified signal CSV."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from domain.artifacts import WorkflowResult
from domain.execution import BacktestResult
from domain.signals import SignalBook
from backtest.portfolio import (
    FeeModel,
    PortfolioSettings,
    run_staggered_cohort_portfolio_from_prepared,
    write_portfolio_backtest_outputs,
)
from market.data import clean_market_data
from signals.schema import SIGNAL_COLUMNS, frame_to_signals
from strategies.preselect import load_raw_data


DEFAULT_SIGNAL_PORTFOLIO_OUTPUT = Path("data") / "portfolio_backtest_signals"


def signal_frame_to_picks(
    signals: pd.DataFrame,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_positions: int | None = None,
) -> SignalBook:
    """Convert unified buy signals to score-ordered per-date candidates."""

    required = {"date", "symbol", "signal_type"}
    missing = required.difference(signals.columns)
    if missing:
        raise ValueError(f"missing signal columns: {', '.join(sorted(missing))}")
    if max_positions is not None and max_positions <= 0:
        raise ValueError("max_positions must be positive")
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
            ["date", "_score", "symbol"],
            ascending=[True, False, True],
            na_position="last",
            kind="mergesort",
        )
    else:
        frame = frame.sort_values(["date", "symbol"], kind="mergesort")
    frame = frame.drop_duplicates(["date", "symbol"], keep="first")
    if max_positions is not None:
        frame = frame.groupby("date", sort=False, group_keys=False).head(max_positions)
    return SignalBook(frame_to_signals(frame))


def run_signal_portfolio_backtest(
    *,
    signals_path: str | Path,
    data_dir: str | Path = "data/raw",
    output_dir: str | Path = DEFAULT_SIGNAL_PORTFOLIO_OUTPUT,
    source: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    initial_cash: float = 10000000.0,
    hold_days: int = 20,
    commission_wan: float = 0.8,
    max_positions: int = 10,
    lot_size: int = 100,
    show_progress: bool = False,
) -> WorkflowResult[BacktestResult]:
    """Backtest one unified signal source with strict fixed staggered sleeves."""

    signal_file = Path(signals_path)
    if not signal_file.exists():
        raise FileNotFoundError(f"signals file not found: {signal_file}")
    if hold_days <= 0:
        raise ValueError("hold_days must be positive")
    if initial_cash <= 0:
        raise ValueError("initial_cash must be positive")
    if max_positions <= 0:
        raise ValueError("max_positions must be positive")
    if lot_size <= 0:
        raise ValueError("lot_size must be positive")

    signals = pd.read_csv(signal_file, dtype={"symbol": str})
    missing = set(SIGNAL_COLUMNS).difference(signals.columns)
    if missing:
        raise ValueError(
            "unified signal file is missing columns: " + ", ".join(sorted(missing))
        )
    signals["date"] = pd.to_datetime(signals["date"], errors="coerce")
    if signals["date"].isna().any():
        raise ValueError("signal dates must be valid")
    signals["symbol"] = signals["symbol"].astype(str).str.zfill(6)
    sources = sorted(signals["source"].dropna().astype(str).unique())
    if source is None and len(sources) > 1:
        raise ValueError(
            "signal file contains multiple sources; pass --source explicitly: "
            + ", ".join(sources)
        )
    resolved_source = str(source) if source is not None else (sources[0] if sources else "")
    if source is not None:
        signals = signals.loc[signals["source"].astype(str).eq(str(source))].copy()
    if signals.empty:
        raise ValueError(f"no signals remain for source {resolved_source!r}")

    raw_data = load_raw_data(str(data_dir), end_date=None)
    prepared = clean_market_data(raw_data)
    if not prepared:
        raise ValueError("no usable market data found")
    resolved_start = start_date or pd.Timestamp(signals["date"].min()).strftime("%Y-%m-%d")
    if end_date:
        resolved_end = end_date
    else:
        latest_market_date = max(frame.index.max() for frame in prepared.values())
        resolved_end = pd.Timestamp(latest_market_date).strftime("%Y-%m-%d")
    active_signals = signals.loc[
        signals["date"].between(pd.Timestamp(resolved_start), pd.Timestamp(resolved_end))
    ].copy()
    _validate_equal_weight_contract(active_signals)
    picks = signal_frame_to_picks(
        active_signals,
        start_date=resolved_start,
        end_date=resolved_end,
        max_positions=max_positions,
    )
    if not picks:
        raise ValueError("no buy signals remain after date and source filtering")

    settings = PortfolioSettings(
        initial_cash=initial_cash,
        strategy=resolved_source or "unified_signal",
        buy_mode="next_open",
        hold_days=hold_days,
        fee_model=FeeModel.from_commission_wan(commission_wan),
        max_positions=max_positions,
        position_pct=1.0 / (hold_days * max_positions),
        lot_size=lot_size,
    )
    result = run_staggered_cohort_portfolio_from_prepared(
        prepared=prepared,
        picks_by_date=picks,
        settings=settings,
        cohort_count=hold_days,
        start_date=resolved_start,
        end_date=resolved_end,
        show_progress=show_progress,
    )
    result.summary.update(
        {
            "start_date": resolved_start,
            "end_date": resolved_end,
            "signal_file": str(signal_file.resolve()),
            "signal_source_filter": resolved_source,
            "signal_count": sum(len(values) for values in picks.values()),
            "signal_date_count": len(picks),
            "max_positions_per_cohort": max_positions,
            "signal_weight_policy": "equal weight; unequal daily weights are rejected",
            "signal_execution_timing": "signal date + 1 trading day at open",
            "cohort_policy": (
                f"capital split into {hold_days} fixed sleeves; one scheduled sleeve per day"
            ),
        }
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = write_portfolio_backtest_outputs(result, destination)
    outputs["result"] = result
    return outputs


def _validate_equal_weight_contract(signals: pd.DataFrame) -> None:
    """Reject unequal target weights because the cohort engine is equal-weight only."""

    buy = signals.loc[signals["signal_type"].astype(str).str.lower().eq("buy")].copy()
    buy["weight"] = pd.to_numeric(buy["weight"], errors="coerce")
    for date, daily in buy.groupby("date", sort=True):
        weights = daily["weight"].dropna()
        if weights.empty:
            continue
        if weights.nunique() > 1:
            raise ValueError(
                f"unequal signal weights are not supported on {pd.Timestamp(date).date()}; "
                "pre-rank to an equal-weight candidate set"
            )
