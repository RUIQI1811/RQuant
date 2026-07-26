"""Run the constrained portfolio engine from a unified signal CSV."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import polars as pl

from domain.artifacts import WorkflowResult
from domain.execution import BacktestResult
from domain.signals import SignalBook
from domain.tabular import to_polars
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
    signals: Any,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_positions: int | None = None,
) -> SignalBook:
    """Convert unified buy signals to score-ordered per-date candidates."""

    frame = to_polars(signals)
    required = {"date", "symbol", "signal_type"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing signal columns: {', '.join(sorted(missing))}")
    if max_positions is not None and max_positions <= 0:
        raise ValueError("max_positions must be positive")
    expressions = [
        pl.col("date")
        .cast(pl.String)
        .str.slice(0, 10)
        .str.to_date("%Y-%m-%d", strict=False),
        pl.col("symbol").cast(pl.String).str.pad_start(6, "0"),
        pl.col("signal_type").cast(pl.String).str.to_lowercase(),
    ]
    if "score" in frame.columns:
        expressions.append(pl.col("score").cast(pl.Float64, strict=False).alias("_score"))
    frame = (
        frame.with_columns(expressions)
        .drop_nulls(["date", "symbol"])
        .filter(pl.col("signal_type") == "buy")
    )
    if start_date:
        frame = frame.filter(pl.col("date") >= pl.lit(start_date).str.to_date())
    if end_date:
        frame = frame.filter(pl.col("date") <= pl.lit(end_date).str.to_date())
    if "score" in frame.columns:
        frame = frame.sort(
            ["date", "_score", "symbol"],
            descending=[False, True, False],
            nulls_last=True,
            maintain_order=True,
        )
    else:
        frame = frame.sort(["date", "symbol"], maintain_order=True)
    frame = frame.unique(["date", "symbol"], keep="first", maintain_order=True)
    if max_positions is not None:
        frame = frame.group_by("date", maintain_order=True).head(max_positions)
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
    stamp_tax_rate: float = 0.0005,
    transfer_fee_rate: float = 0.00001,
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
    if commission_wan < 0 or stamp_tax_rate < 0 or transfer_fee_rate < 0:
        raise ValueError("transaction-cost rates must be non-negative")

    signals = pl.read_csv(signal_file, schema_overrides={"symbol": pl.String})
    missing = set(SIGNAL_COLUMNS).difference(signals.columns)
    if missing:
        raise ValueError(
            "unified signal file is missing columns: " + ", ".join(sorted(missing))
        )
    signals = signals.with_columns(
        pl.col("date").cast(pl.String).str.slice(0, 10).str.to_date("%Y-%m-%d", strict=False),
        pl.col("symbol").cast(pl.String).str.pad_start(6, "0"),
    )
    if signals["date"].null_count() > 0:
        raise ValueError("signal dates must be valid")
    sources = sorted(str(value) for value in signals["source"].drop_nulls().unique())
    if source is None and len(sources) > 1:
        raise ValueError(
            "signal file contains multiple sources; pass --source explicitly: "
            + ", ".join(sources)
        )
    resolved_source = str(source) if source is not None else (sources[0] if sources else "")
    if source is not None:
        signals = signals.filter(pl.col("source").cast(pl.String) == str(source))
    if signals.is_empty():
        raise ValueError(f"no signals remain for source {resolved_source!r}")

    raw_data = load_raw_data(str(data_dir), end_date=None)
    prepared = clean_market_data(raw_data)
    if not prepared:
        raise ValueError("no usable market data found")
    resolved_start = start_date or signals["date"].min().isoformat()
    if end_date:
        resolved_end = end_date
    else:
        latest_market_date = max(frame.index.max() for frame in prepared.values())
        resolved_end = latest_market_date.strftime("%Y-%m-%d")
    active_signals = signals.filter(
        pl.col("date").is_between(
            pl.lit(resolved_start).str.to_date(),
            pl.lit(resolved_end).str.to_date(),
            closed="both",
        )
    )
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
        fee_model=FeeModel(
            commission_rate=float(commission_wan) / 10000.0,
            stamp_tax_rate=float(stamp_tax_rate),
            transfer_fee_rate=float(transfer_fee_rate),
        ),
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


def _validate_equal_weight_contract(signals: pl.DataFrame) -> None:
    """Reject unequal target weights because the cohort engine is equal-weight only."""

    buy = signals.filter(
        pl.col("signal_type").cast(pl.String).str.to_lowercase() == "buy"
    ).with_columns(pl.col("weight").cast(pl.Float64, strict=False))
    for daily in buy.sort("date").partition_by("date", maintain_order=True):
        date = daily.item(0, "date")
        weights = daily["weight"].drop_nulls()
        if weights.is_empty():
            continue
        if weights.n_unique() > 1:
            raise ValueError(
                f"unequal signal weights are not supported on {date}; "
                "pre-rank to an equal-weight candidate set"
            )
