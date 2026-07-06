"""Guotai Junan 191 factor operators and calculators.

Rows are trading dates and columns are six-digit stock symbols.  Time-series
operators work down rows; cross-sectional operators work across columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


Panel = pd.DataFrame
GTJA191_NAMES = tuple(f"gtja_{number:03d}" for number in range(1, 192))


class GTJA191Error(ValueError):
    """Base error for GTJA191 calculation failures."""


class GTJA191DataError(GTJA191Error):
    """Raised when a factor's required point-in-time input is unavailable."""


class GTJA191FormulaError(GTJA191Error):
    """Raised when a source formula cannot be resolved unambiguously."""


@dataclass(frozen=True)
class GTJA191ExternalData:
    """Optional point-in-time market series required by five GTJA factors."""

    benchmark_open: pd.Series | None = None
    benchmark_close: pd.Series | None = None
    mkt: pd.Series | None = None
    smb: pd.Series | None = None
    hml: pd.Series | None = None


@dataclass(frozen=True)
class GTJA191Panels:
    """Aligned wide daily inputs for the GTJA191 calculator."""

    open: Panel
    close: Panel
    high: Panel
    low: Panel
    volume: Panel
    amount: Panel
    vwap: Panel
    returns: Panel
    external: GTJA191ExternalData = field(default_factory=GTJA191ExternalData)
    market_cap: Panel | None = None
    is_st: Panel | None = None
    industry: Panel | None = None


def _window(value: int | float) -> int:
    return max(1, int(np.floor(float(value) + 0.5)))


def normalize_gtja_name(name: str | int) -> str:
    """Normalize supported aliases to the non-conflicting ``gtja_NNN`` form."""

    if isinstance(name, int):
        number = name
    else:
        raw = str(name).strip().lower().replace("-", "_")
        if raw.startswith("gtja_"):
            raw = raw.removeprefix("gtja_")
        elif raw.startswith("gtja"):
            raw = raw.removeprefix("gtja")
        else:
            raise KeyError(f"invalid GTJA191 factor name: {name}")
        try:
            number = int(raw)
        except ValueError as exc:
            raise KeyError(f"invalid GTJA191 factor name: {name}") from exc
    if not 1 <= number <= 191:
        raise KeyError(f"GTJA191 factor number must be in [1, 191], got {number}")
    return f"gtja_{number:03d}"


def sma_cn(value: Panel, periods: int | float, weight: int | float) -> Panel:
    """Chinese SMA: ``Y[t]=(m*X[t]+(n-m)*Y[t-1])/n``."""

    n = float(_window(periods))
    m = float(weight)
    if not 0 < m <= n:
        raise ValueError("SMA weight must be in (0, periods]")
    output = pd.DataFrame(np.nan, index=value.index, columns=value.columns, dtype=float)
    for column in value.columns:
        previous: float | None = None
        for index, raw in value[column].items():
            if pd.isna(raw):
                previous = None
                continue
            current = float(raw)
            previous = current if previous is None else (m * current + (n - m) * previous) / n
            output.at[index, column] = previous
    return output


def wma(value: Panel, periods: int | float) -> Panel:
    """Report WMA with weights proportional to ``0.9**distance``."""

    window = _window(periods)
    weights = np.power(0.9, np.arange(window - 1, -1, -1, dtype=float))
    weights /= weights.sum()
    return value.rolling(window, min_periods=window).apply(
        lambda values: float(np.dot(values, weights)),
        raw=True,
    )


def _distance_from_current(values: np.ndarray, reducer: Callable[[np.ndarray], int]) -> float:
    if np.isnan(values).any():
        return np.nan
    return float(reducer(values[::-1]))


def highday(value: Panel, periods: int | float) -> Panel:
    """Distance from today to the most recent window maximum."""

    window = _window(periods)
    return value.rolling(window, min_periods=window).apply(
        lambda values: _distance_from_current(values, np.argmax),
        raw=True,
    )


def lowday(value: Panel, periods: int | float) -> Panel:
    """Distance from today to the most recent window minimum."""

    window = _window(periods)
    return value.rolling(window, min_periods=window).apply(
        lambda values: _distance_from_current(values, np.argmin),
        raw=True,
    )


def _rolling_regression(
    dependent: Panel,
    independent: Panel,
    periods: int | float,
) -> tuple[Panel, Panel]:
    window = _window(periods)
    left, right = dependent.align(independent, join="outer")
    beta = pd.DataFrame(np.nan, index=left.index, columns=left.columns, dtype=float)
    residual = beta.copy()
    for column in left.columns:
        y_values = left[column].to_numpy(dtype=float)
        x_values = right[column].to_numpy(dtype=float)
        for end in range(window - 1, len(left)):
            start = end - window + 1
            y_window = y_values[start : end + 1]
            x_window = x_values[start : end + 1]
            if np.isnan(y_window).any() or np.isnan(x_window).any():
                continue
            design = np.column_stack([np.ones(window), x_window])
            coefficients, _, _, _ = np.linalg.lstsq(design, y_window, rcond=None)
            beta.iat[end, beta.columns.get_loc(column)] = coefficients[1]
            residual.iat[end, residual.columns.get_loc(column)] = y_window[-1] - (
                coefficients[0] + coefficients[1] * x_window[-1]
            )
    return beta, residual


def regbeta(dependent: Panel, independent: Panel, periods: int | float) -> Panel:
    """Rolling OLS slope of dependent data on independent data with intercept."""

    return _rolling_regression(dependent, independent, periods)[0]


def regresi(dependent: Panel, independent: Panel, periods: int | float) -> Panel:
    """Current residual from rolling OLS with intercept."""

    return _rolling_regression(dependent, independent, periods)[1]


def count(condition: Panel, periods: int | float) -> Panel:
    """Count true observations over a complete rolling window."""

    window = _window(periods)
    numeric = condition.astype(float)
    return numeric.rolling(window, min_periods=window).sum()


def sumif(value: Panel, periods: int | float, condition: Panel) -> Panel:
    """Sum values satisfying a condition over a complete rolling window."""

    window = _window(periods)
    selected = value.where(condition, 0.0).where(value.notna())
    return selected.rolling(window, min_periods=window).sum()
