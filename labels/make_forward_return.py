"""Forward-return label generation for ML training and factor reports."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import polars as pl

from domain.tabular import to_polars


def make_forward_returns(
    prices: Any,
    windows: Iterable[int] = (1, 5, 10, 20),
    date_col: str = "date",
    symbol_col: str = "symbol",
    close_col: str = "close",
) -> pl.DataFrame:
    required = {date_col, symbol_col, close_col}
    frame = to_polars(prices)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    frame = _prepare_prices(frame, date_col, symbol_col, close_col)
    if frame.select([date_col, symbol_col]).is_duplicated().any():
        raise ValueError("duplicate date/symbol rows are not allowed")
    for window in tuple(windows):
        if int(window) <= 0:
            raise ValueError("forward return windows must be positive")
        frame = frame.with_columns(
            (
                pl.col(close_col).shift(-int(window)).over(symbol_col)
                / pl.col(close_col)
                - 1.0
            ).alias(f"forward_return_{int(window)}d")
        )
    return frame.with_columns(
        pl.col(date_col).dt.strftime("%Y-%m-%d")
    ).drop(close_col)


def make_next_open_returns(
    prices: Any,
    windows: Iterable[int] = (1, 5, 10, 20),
    date_col: str = "date",
    symbol_col: str = "symbol",
    open_col: str = "open",
) -> pl.DataFrame:
    """Return labels aligned with next-open entry and open exit after N bars."""

    required = {date_col, symbol_col, open_col}
    frame = to_polars(prices)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    frame = _prepare_prices(frame, date_col, symbol_col, open_col)
    if frame.select([date_col, symbol_col]).is_duplicated().any():
        raise ValueError("duplicate date/symbol rows are not allowed")
    for window in tuple(windows):
        if int(window) <= 0:
            raise ValueError("next-open return windows must be positive")
        frame = frame.with_columns(
            (
                pl.col(open_col).shift(-(int(window) + 1)).over(symbol_col)
                / pl.col(open_col).shift(-1).over(symbol_col)
                - 1.0
            ).alias(f"next_open_return_{int(window)}d")
        )
    return frame.with_columns(
        pl.col(date_col).dt.strftime("%Y-%m-%d")
    ).drop(open_col)


def _prepare_prices(
    frame: pl.DataFrame,
    date_col: str,
    symbol_col: str,
    value_col: str,
) -> pl.DataFrame:
    return (
        frame.select([date_col, symbol_col, value_col])
        .with_columns(
            pl.col(date_col)
            .cast(pl.String)
            .str.slice(0, 10)
            .str.to_date("%Y-%m-%d", strict=False),
            pl.col(symbol_col).cast(pl.String).str.pad_start(6, "0"),
            pl.col(value_col).cast(pl.Float64, strict=False),
        )
        .sort([symbol_col, date_col], maintain_order=True)
    )
