"""Forward-return label generation for ML training and factor reports."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def make_forward_returns(
    prices: pd.DataFrame,
    windows: Iterable[int] = (1, 5, 10, 20),
    date_col: str = "date",
    symbol_col: str = "symbol",
    close_col: str = "close",
) -> pd.DataFrame:
    required = {date_col, symbol_col, close_col}
    missing = sorted(required.difference(prices.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    frame = prices[[date_col, symbol_col, close_col]].copy()
    frame[symbol_col] = frame[symbol_col].astype(str).str.zfill(6)
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = frame.sort_values([symbol_col, date_col]).reset_index(drop=True)

    if frame.duplicated([date_col, symbol_col]).any():
        raise ValueError("duplicate date/symbol rows are not allowed")

    grouped = frame.groupby(symbol_col, sort=False)[close_col]
    for window in tuple(windows):
        if int(window) <= 0:
            raise ValueError("forward return windows must be positive")
        future = grouped.shift(-int(window))
        frame[f"forward_return_{int(window)}d"] = future / frame[close_col] - 1.0

    frame[date_col] = frame[date_col].dt.strftime("%Y-%m-%d")
    return frame.drop(columns=[close_col])
