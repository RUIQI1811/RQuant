"""Shared panel operators with stable semantics across factor families.

Rows are trading dates and columns are symbols. Cross-sectional operators work
across columns; time-series operators work down rows. Family-specific operators
whose published semantics differ remain in their calculator modules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


Panel = pd.DataFrame


def window(value: float | int) -> int:
    """Floor a lookback to the nearest positive trading-day integer."""

    return max(1, int(np.floor(float(value))))


def replace_inf(value: Panel) -> Panel:
    """Replace positive and negative infinity with missing values."""

    return value.replace([np.inf, -np.inf], np.nan)


def safe_div(numerator: Panel, denominator: Panel | float) -> Panel:
    """Divide panels while treating near-zero denominators as unavailable."""

    if isinstance(denominator, pd.DataFrame):
        denominator = denominator.mask(denominator.abs() < 1e-12)
    elif abs(float(denominator)) < 1e-12:
        denominator = np.nan
    return replace_inf(numerator / denominator)


def rank(value: Panel) -> Panel:
    """Return the percentile rank of every daily stock cross-section."""

    return value.rank(axis=1, method="average", pct=True)


def delay(value: Panel, periods: float | int) -> Panel:
    return value.shift(window(periods))


def correlation(left: Panel, right: Panel, periods: float | int) -> Panel:
    lookback = window(periods)
    return left.rolling(lookback, min_periods=lookback).corr(right)


def covariance(left: Panel, right: Panel, periods: float | int) -> Panel:
    lookback = window(periods)
    return left.rolling(lookback, min_periods=lookback).cov(right)


def scale(value: Panel, target: float = 1.0) -> Panel:
    denominator = value.abs().sum(axis=1).replace(0.0, np.nan)
    return value.div(denominator, axis=0) * float(target)


def delta(value: Panel, periods: float | int) -> Panel:
    return value.diff(window(periods))


def signed_power(value: Panel, exponent: Panel | float) -> Panel:
    return np.sign(value) * np.power(value.abs(), exponent)


def decay_linear(value: Panel, periods: float | int) -> Panel:
    lookback = window(periods)
    weights = np.arange(1.0, lookback + 1.0)
    weights /= weights.sum()

    def weighted(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        return float(np.dot(values, weights))

    return value.rolling(lookback, min_periods=lookback).apply(weighted, raw=True)


def ts_min(value: Panel, periods: float | int) -> Panel:
    lookback = window(periods)
    return value.rolling(lookback, min_periods=lookback).min()


def ts_max(value: Panel, periods: float | int) -> Panel:
    lookback = window(periods)
    return value.rolling(lookback, min_periods=lookback).max()


def ts_argmin(value: Panel, periods: float | int) -> Panel:
    lookback = window(periods)
    return value.rolling(lookback, min_periods=lookback).apply(
        lambda values: float(np.argmin(values) + 1), raw=True
    )


def ts_argmax(value: Panel, periods: float | int) -> Panel:
    lookback = window(periods)
    return value.rolling(lookback, min_periods=lookback).apply(
        lambda values: float(np.argmax(values) + 1), raw=True
    )


def ts_rank(value: Panel, periods: float | int) -> Panel:
    lookback = window(periods)

    def last_rank(values: np.ndarray) -> float:
        return float(pd.Series(values).rank(method="average", pct=True).iloc[-1])

    return value.rolling(lookback, min_periods=lookback).apply(last_rank, raw=True)


def ts_sum(value: Panel, periods: float | int) -> Panel:
    lookback = window(periods)
    return value.rolling(lookback, min_periods=lookback).sum()


def product(value: Panel, periods: float | int) -> Panel:
    lookback = window(periods)
    return value.rolling(lookback, min_periods=lookback).apply(np.prod, raw=True)


def stddev(value: Panel, periods: float | int) -> Panel:
    lookback = window(periods)
    return value.rolling(lookback, min_periods=lookback).std(ddof=1)


def element_min(left: Panel, right: Panel) -> Panel:
    return left.combine(right, np.minimum)


def element_max(left: Panel, right: Panel) -> Panel:
    return left.combine(right, np.maximum)


__all__ = [
    "Panel",
    "correlation",
    "covariance",
    "decay_linear",
    "delay",
    "delta",
    "element_max",
    "element_min",
    "product",
    "rank",
    "replace_inf",
    "safe_div",
    "scale",
    "signed_power",
    "stddev",
    "ts_argmax",
    "ts_argmin",
    "ts_max",
    "ts_min",
    "ts_rank",
    "ts_sum",
    "window",
]
