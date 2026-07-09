"""Benchmark comparison helpers for report generation."""

from __future__ import annotations

import pandas as pd


def align_portfolio_and_benchmark(
    portfolio: pd.DataFrame,
    benchmark: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    left = portfolio.copy()
    right = benchmark.copy()
    left[date_col] = pd.to_datetime(left[date_col])
    right[date_col] = pd.to_datetime(right[date_col])
    return left.merge(right, on=date_col, how="inner", suffixes=("_portfolio", "_benchmark"))
