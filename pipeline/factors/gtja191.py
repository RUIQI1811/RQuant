"""Guotai Junan 191 factor operators and calculators.

Rows are trading dates and columns are six-digit stock symbols.  Time-series
operators work down rows; cross-sectional operators work across columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .alpha101 import (
    correlation,
    covariance,
    decay_linear,
    delay,
    delta,
    element_max,
    element_min,
    product,
    rank,
    scale,
    signed_power,
    stddev,
    ts_argmax,
    ts_argmin,
    ts_max,
    ts_min,
    ts_rank,
    ts_sum,
)


Panel = pd.DataFrame
GTJA191_NAMES = tuple(f"gtja_{number:03d}" for number in range(1, 192))
GTJA191_FORMULA_NOTES: dict[str, str] = {
    "gtja_028": "Use TSMAX(HIGH,9)-TSMIN(LOW,9) in both stochastic terms.",
    "gtja_030": "Map MKT/SMB/HML to explicit external daily factor returns.",
    "gtja_035": "The published OPEN*0.65+OPEN*0.35 term simplifies to OPEN.",
    "gtja_054": "Use the original report's explicit 10-day STD window.",
    "gtja_075": "Benchmark OPEN and CLOSE map to their semantic index fields.",
    "gtja_078": "MA is interpreted as the report-defined rolling MEAN.",
}


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


def mean(value: Panel, periods: int | float) -> Panel:
    """Rolling arithmetic mean with a complete window."""

    window = _window(periods)
    return value.rolling(window, min_periods=window).mean()


def _replace_inf(value: Panel) -> Panel:
    return value.replace([np.inf, -np.inf], np.nan)


def _safe_div(numerator: Panel, denominator: Panel | float) -> Panel:
    if isinstance(denominator, pd.DataFrame):
        denominator = denominator.mask(denominator.abs() < 1e-12)
    elif abs(float(denominator)) < 1e-12:
        denominator = np.nan
    return _replace_inf(numerator / denominator)


def _conditional(
    condition: Panel,
    true_value: Panel | float,
    false_value: Panel | float,
    *,
    valid: Panel | None = None,
) -> Panel:
    output = pd.DataFrame(
        np.where(condition, true_value, false_value),
        index=condition.index,
        columns=condition.columns,
        dtype=float,
    )
    return output if valid is None else output.where(valid)


def _sequence_regbeta(value: Panel, periods: int | float) -> Panel:
    window = _window(periods)
    sequence = np.arange(1.0, window + 1.0)

    def slope(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        design = np.column_stack([np.ones(window), sequence])
        return float(np.linalg.lstsq(design, values, rcond=None)[0][1])

    return value.rolling(window, min_periods=window).apply(slope, raw=True)


def _broadcast_series(series: pd.Series, template: Panel) -> Panel:
    aligned = pd.to_numeric(series, errors="coerce").reindex(template.index)
    values = np.repeat(aligned.to_numpy()[:, None], len(template.columns), axis=1)
    return pd.DataFrame(values, index=template.index, columns=template.columns)


def _multifactor_residual(
    dependent: Panel,
    factors: tuple[pd.Series, ...],
    periods: int | float,
) -> Panel:
    window = _window(periods)
    aligned_factors = [
        pd.to_numeric(series, errors="coerce").reindex(dependent.index).to_numpy(dtype=float)
        for series in factors
    ]
    factor_values = np.column_stack(aligned_factors)
    output = pd.DataFrame(np.nan, index=dependent.index, columns=dependent.columns)
    for column_index, column in enumerate(dependent.columns):
        y_values = dependent[column].to_numpy(dtype=float)
        for end in range(window - 1, len(dependent)):
            start = end - window + 1
            y_window = y_values[start : end + 1]
            x_window = factor_values[start : end + 1]
            if np.isnan(y_window).any() or np.isnan(x_window).any():
                continue
            design = np.column_stack([np.ones(window), x_window])
            coefficients = np.linalg.lstsq(design, y_window, rcond=None)[0]
            output.iat[end, column_index] = y_window[-1] - float(
                design[-1] @ coefficients
            )
    return output


class GTJA191:
    """Calculate Guotai Junan Alpha191 factors on aligned wide panels."""

    def __init__(self, data: GTJA191Panels) -> None:
        self.d = data

    @property
    def names(self) -> tuple[str, ...]:
        return GTJA191_NAMES

    def calculate(self, name: str | int) -> Panel:
        normalized = normalize_gtja_name(name)
        method = getattr(self, normalized, None)
        if method is None:
            raise KeyError(f"GTJA191 factor is not implemented: {normalized}")
        return _replace_inf(method())

    def calculate_many(
        self,
        names: list[str | int] | tuple[str | int, ...] | None = None,
        *,
        on_error: str = "raise",
    ) -> dict[str, Panel]:
        selected = GTJA191_NAMES if names is None else tuple(
            normalize_gtja_name(name) for name in names
        )
        output: dict[str, Panel] = {}
        for name in selected:
            try:
                output[name] = self.calculate(name)
            except GTJA191Error:
                if on_error != "nan":
                    raise
                output[name] = self.d.close.copy() * np.nan
        return output

    def _require_external(self, factor_name: str, *fields: str) -> tuple[pd.Series, ...]:
        missing = [field for field in fields if getattr(self.d.external, field) is None]
        if missing:
            raise GTJA191DataError(
                f"{factor_name} requires external fields: {', '.join(missing)}"
            )
        return tuple(getattr(self.d.external, field) for field in fields)

    def _signed_volume(self) -> Panel:
        previous = delay(self.d.close, 1)
        return self.d.volume.where(self.d.close > previous, -self.d.volume).where(
            self.d.close != previous,
            0.0,
        )

    def _dtm(self) -> Panel:
        previous_open = delay(self.d.open, 1)
        value = element_max(self.d.high - self.d.open, self.d.open - previous_open)
        return value.where(self.d.open > previous_open, 0.0)

    def _dbm(self) -> Panel:
        previous_open = delay(self.d.open, 1)
        value = element_max(self.d.open - self.d.low, self.d.open - previous_open)
        return value.where(self.d.open < previous_open, 0.0)

    def gtja_001(self) -> Panel:
        return -correlation(
            rank(delta(np.log(self.d.volume.mask(self.d.volume <= 0)), 1)),
            rank(_safe_div(self.d.close - self.d.open, self.d.open)),
            6,
        )

    def gtja_002(self) -> Panel:
        position = _safe_div(
            (self.d.close - self.d.low) - (self.d.high - self.d.close),
            self.d.high - self.d.low,
        )
        return -delta(position, 1)

    def gtja_003(self) -> Panel:
        previous = delay(self.d.close, 1)
        up = self.d.close - element_min(self.d.low, previous)
        down = self.d.close - element_max(self.d.high, previous)
        value = up.where(self.d.close > previous, down).where(self.d.close != previous, 0.0)
        return ts_sum(value, 6)

    def gtja_004(self) -> Panel:
        mean8 = mean(self.d.close, 8)
        mean2 = mean(self.d.close, 2)
        std8 = stddev(self.d.close, 8)
        volume_ratio = _safe_div(self.d.volume, mean(self.d.volume, 20))
        result = _conditional(volume_ratio >= 1.0, 1.0, -1.0, valid=volume_ratio.notna())
        result = result.where(~(mean2 < mean8 - std8), 1.0)
        result = result.where(~(mean8 + std8 < mean2), -1.0)
        return result.where(mean8.notna() & mean2.notna() & std8.notna())

    def gtja_005(self) -> Panel:
        return -ts_max(
            correlation(ts_rank(self.d.volume, 5), ts_rank(self.d.high, 5), 5),
            3,
        )

    def gtja_006(self) -> Panel:
        return -rank(np.sign(delta(self.d.open * 0.85 + self.d.high * 0.15, 4)))

    def gtja_007(self) -> Panel:
        spread = self.d.vwap - self.d.close
        return (rank(ts_max(spread, 3)) + rank(ts_min(spread, 3))) * rank(
            delta(self.d.volume, 3)
        )

    def gtja_008(self) -> Panel:
        value = (self.d.high + self.d.low) / 2.0 * 0.2 + self.d.vwap * 0.8
        return -rank(delta(value, 4))

    def gtja_009(self) -> Panel:
        midpoint = (self.d.high + self.d.low) / 2.0
        value = _safe_div(
            (midpoint - delay(midpoint, 1)) * (self.d.high - self.d.low),
            self.d.volume,
        )
        return sma_cn(value, 7, 2)

    def gtja_010(self) -> Panel:
        value = self.d.close.where(self.d.returns >= 0, stddev(self.d.returns, 20))
        return rank(ts_max(value.pow(2), 5))

    def gtja_011(self) -> Panel:
        value = _safe_div(
            (self.d.close - self.d.low) - (self.d.high - self.d.close),
            self.d.high - self.d.low,
        ) * self.d.volume
        return ts_sum(value, 6)

    def gtja_012(self) -> Panel:
        return rank(self.d.open - mean(self.d.vwap, 10)) * -rank(
            (self.d.close - self.d.vwap).abs()
        )

    def gtja_013(self) -> Panel:
        return np.sqrt(self.d.high * self.d.low) - self.d.vwap

    def gtja_014(self) -> Panel:
        return self.d.close - delay(self.d.close, 5)

    def gtja_015(self) -> Panel:
        return _safe_div(self.d.open, delay(self.d.close, 1)) - 1.0

    def gtja_016(self) -> Panel:
        return -ts_max(rank(correlation(rank(self.d.volume), rank(self.d.vwap), 5)), 5)

    def gtja_017(self) -> Panel:
        base = rank(self.d.vwap - ts_max(self.d.vwap, 15))
        return np.power(base, delta(self.d.close, 5))

    def gtja_018(self) -> Panel:
        return _safe_div(self.d.close, delay(self.d.close, 5))

    def gtja_019(self) -> Panel:
        previous = delay(self.d.close, 5)
        lower = _safe_div(self.d.close - previous, previous)
        higher = _safe_div(self.d.close - previous, self.d.close)
        return lower.where(self.d.close < previous, higher).where(
            self.d.close != previous,
            0.0,
        )

    def gtja_020(self) -> Panel:
        previous = delay(self.d.close, 6)
        return _safe_div(self.d.close - previous, previous) * 100.0

    def gtja_021(self) -> Panel:
        return _sequence_regbeta(mean(self.d.close, 6), 6)

    def gtja_022(self) -> Panel:
        ratio = _safe_div(self.d.close - mean(self.d.close, 6), mean(self.d.close, 6))
        return sma_cn(ratio - delay(ratio, 3), 12, 1)

    def gtja_023(self) -> Panel:
        volatility = stddev(self.d.close, 20)
        previous = delay(self.d.close, 1)
        up = sma_cn(volatility.where(self.d.close > previous, 0.0), 20, 1)
        down = sma_cn(volatility.where(self.d.close <= previous, 0.0), 20, 1)
        return _safe_div(up, up + down) * 100.0

    def gtja_024(self) -> Panel:
        return sma_cn(self.d.close - delay(self.d.close, 5), 5, 1)

    def gtja_025(self) -> Panel:
        liquidity = _safe_div(self.d.volume, mean(self.d.volume, 20))
        first = -rank(delta(self.d.close, 7) * (1.0 - rank(decay_linear(liquidity, 9))))
        return first * (1.0 + rank(ts_sum(self.d.returns, 250)))

    def gtja_026(self) -> Panel:
        return mean(self.d.close, 7) - self.d.close + correlation(
            self.d.vwap,
            delay(self.d.close, 5),
            230,
        )

    def gtja_027(self) -> Panel:
        roc3 = _safe_div(self.d.close - delay(self.d.close, 3), delay(self.d.close, 3))
        roc6 = _safe_div(self.d.close - delay(self.d.close, 6), delay(self.d.close, 6))
        return wma((roc3 + roc6) * 100.0, 12)

    def gtja_028(self) -> Panel:
        stochastic = _safe_div(
            self.d.close - ts_min(self.d.low, 9),
            ts_max(self.d.high, 9) - ts_min(self.d.low, 9),
        ) * 100.0
        first = sma_cn(stochastic, 3, 1)
        return 3.0 * first - 2.0 * sma_cn(first, 3, 1)

    def gtja_029(self) -> Panel:
        previous = delay(self.d.close, 6)
        return _safe_div(self.d.close - previous, previous) * self.d.volume

    def gtja_030(self) -> Panel:
        mkt, smb, hml = self._require_external("gtja_030", "mkt", "smb", "hml")
        residual = _multifactor_residual(self.d.returns, (mkt, smb, hml), 60)
        return wma(residual.pow(2), 20)

    def gtja_031(self) -> Panel:
        average = mean(self.d.close, 12)
        return _safe_div(self.d.close - average, average) * 100.0

    def gtja_032(self) -> Panel:
        return -ts_sum(rank(correlation(rank(self.d.high), rank(self.d.volume), 3)), 3)

    def gtja_033(self) -> Panel:
        low5 = ts_min(self.d.low, 5)
        return (-low5 + delay(low5, 5)) * rank(
            (ts_sum(self.d.returns, 240) - ts_sum(self.d.returns, 20)) / 220.0
        ) * ts_rank(self.d.volume, 5)

    def gtja_034(self) -> Panel:
        return _safe_div(mean(self.d.close, 12), self.d.close)

    def gtja_035(self) -> Panel:
        left = rank(decay_linear(delta(self.d.open, 1), 15))
        right = rank(decay_linear(correlation(self.d.volume, self.d.open, 17), 7))
        return -element_min(left, right)

    def gtja_036(self) -> Panel:
        return rank(ts_sum(correlation(rank(self.d.volume), rank(self.d.vwap), 6), 2))

    def gtja_037(self) -> Panel:
        value = ts_sum(self.d.open, 5) * ts_sum(self.d.returns, 5)
        return -rank(value - delay(value, 10))

    def gtja_038(self) -> Panel:
        return (-delta(self.d.high, 2)).where(mean(self.d.high, 20) < self.d.high, 0.0)

    def gtja_039(self) -> Panel:
        left = rank(decay_linear(delta(self.d.close, 2), 8))
        mixed = self.d.vwap * 0.3 + self.d.open * 0.7
        right = rank(
            decay_linear(
                correlation(mixed, ts_sum(mean(self.d.volume, 180), 37), 14),
                12,
            )
        )
        return -(left - right)

    def gtja_040(self) -> Panel:
        previous = delay(self.d.close, 1)
        up = ts_sum(self.d.volume.where(self.d.close > previous, 0.0), 26)
        down = ts_sum(self.d.volume.where(self.d.close <= previous, 0.0), 26)
        return _safe_div(up, down) * 100.0

    def gtja_041(self) -> Panel:
        return -rank(ts_max(delta(self.d.vwap, 3), 5))

    def gtja_042(self) -> Panel:
        return -rank(stddev(self.d.high, 10)) * correlation(self.d.high, self.d.volume, 10)

    def gtja_043(self) -> Panel:
        return ts_sum(self._signed_volume(), 6)

    def gtja_044(self) -> Panel:
        first = ts_rank(decay_linear(correlation(self.d.low, mean(self.d.volume, 10), 7), 6), 4)
        second = ts_rank(decay_linear(delta(self.d.vwap, 3), 10), 15)
        return first + second

    def gtja_045(self) -> Panel:
        return rank(delta(self.d.close * 0.6 + self.d.open * 0.4, 1)) * rank(
            correlation(self.d.vwap, mean(self.d.volume, 150), 15)
        )

    def gtja_046(self) -> Panel:
        return (
            mean(self.d.close, 3)
            + mean(self.d.close, 6)
            + mean(self.d.close, 12)
            + mean(self.d.close, 24)
        ) / (4.0 * self.d.close)

    def gtja_047(self) -> Panel:
        value = _safe_div(
            ts_max(self.d.high, 6) - self.d.close,
            ts_max(self.d.high, 6) - ts_min(self.d.low, 6),
        ) * 100.0
        return sma_cn(value, 9, 1)

    def gtja_048(self) -> Panel:
        signs = (
            np.sign(self.d.close - delay(self.d.close, 1))
            + np.sign(delay(self.d.close, 1) - delay(self.d.close, 2))
            + np.sign(delay(self.d.close, 2) - delay(self.d.close, 3))
        )
        return -rank(signs) * _safe_div(ts_sum(self.d.volume, 5), ts_sum(self.d.volume, 20))

    def _directional_range_parts(self) -> tuple[Panel, Panel]:
        previous_high = delay(self.d.high, 1)
        previous_low = delay(self.d.low, 1)
        movement = element_max(
            (self.d.high - previous_high).abs(),
            (self.d.low - previous_low).abs(),
        )
        current_sum = self.d.high + self.d.low
        previous_sum = previous_high + previous_low
        down = movement.where(current_sum < previous_sum, 0.0)
        up = movement.where(current_sum > previous_sum, 0.0)
        return up, down

    def gtja_049(self) -> Panel:
        up, down = self._directional_range_parts()
        down_sum = ts_sum(down, 12)
        up_sum = ts_sum(up, 12)
        return _safe_div(down_sum, down_sum + up_sum)

    def gtja_050(self) -> Panel:
        up, down = self._directional_range_parts()
        up_sum = ts_sum(up, 12)
        down_sum = ts_sum(down, 12)
        total = up_sum + down_sum
        return _safe_div(up_sum, total) - _safe_div(down_sum, total)

    def gtja_051(self) -> Panel:
        up, down = self._directional_range_parts()
        up_sum = ts_sum(up, 12)
        return _safe_div(up_sum, up_sum + ts_sum(down, 12))

    def gtja_052(self) -> Panel:
        typical = (self.d.high + self.d.low + self.d.close) / 3.0
        previous = delay(typical, 1)
        numerator = ts_sum(element_max(self.d.high - previous, self.d.high * 0.0), 26)
        denominator = ts_sum(element_max(previous - self.d.low, self.d.low * 0.0), 26)
        return _safe_div(numerator, denominator) * 100.0

    def gtja_053(self) -> Panel:
        return count(self.d.close > delay(self.d.close, 1), 12) / 12.0 * 100.0

    def gtja_054(self) -> Panel:
        value = stddev((self.d.close - self.d.open).abs(), 10) + (self.d.close - self.d.open)
        return -rank(value + correlation(self.d.close, self.d.open, 10))

    def gtja_055(self) -> Panel:
        previous_close = delay(self.d.close, 1)
        previous_open = delay(self.d.open, 1)
        previous_low = delay(self.d.low, 1)
        high_gap = (self.d.high - previous_close).abs()
        low_gap = (self.d.low - previous_close).abs()
        cross_gap = (self.d.high - previous_low).abs()
        open_gap = (previous_close - previous_open).abs()
        first = high_gap + low_gap / 2.0 + open_gap / 4.0
        second = low_gap + high_gap / 2.0 + open_gap / 4.0
        third = cross_gap + open_gap / 4.0
        denominator = third.where(~((low_gap > cross_gap) & (low_gap > high_gap)), second)
        denominator = denominator.where(~((high_gap > low_gap) & (high_gap > cross_gap)), first)
        numerator = 16.0 * (
            self.d.close - previous_close
            + (self.d.close - self.d.open) / 2.0
            + previous_close
            - previous_open
        )
        value = _safe_div(numerator, denominator) * element_max(high_gap, low_gap)
        return ts_sum(value, 20)

    def gtja_056(self) -> Panel:
        left = rank(self.d.open - ts_min(self.d.open, 12))
        corr = correlation(
            ts_sum((self.d.high + self.d.low) / 2.0, 19),
            ts_sum(mean(self.d.volume, 40), 19),
            13,
        )
        right = rank(rank(corr).pow(5))
        return _conditional(left < right, 1.0, 0.0, valid=left.notna() & right.notna())

    def gtja_057(self) -> Panel:
        value = _safe_div(
            self.d.close - ts_min(self.d.low, 9),
            ts_max(self.d.high, 9) - ts_min(self.d.low, 9),
        ) * 100.0
        return sma_cn(value, 3, 1)

    def gtja_058(self) -> Panel:
        return count(self.d.close > delay(self.d.close, 1), 20) / 20.0 * 100.0

    def gtja_059(self) -> Panel:
        previous = delay(self.d.close, 1)
        up = self.d.close - element_min(self.d.low, previous)
        down = self.d.close - element_max(self.d.high, previous)
        value = up.where(self.d.close > previous, down).where(self.d.close != previous, 0.0)
        return ts_sum(value, 20)

    def gtja_060(self) -> Panel:
        value = _safe_div(
            (self.d.close - self.d.low) - (self.d.high - self.d.close),
            self.d.high - self.d.low,
        ) * self.d.volume
        return ts_sum(value, 20)

    def gtja_061(self) -> Panel:
        left = rank(decay_linear(delta(self.d.vwap, 1), 12))
        right = rank(decay_linear(rank(correlation(self.d.low, mean(self.d.volume, 80), 8)), 17))
        return -element_max(left, right)

    def gtja_062(self) -> Panel:
        return -correlation(self.d.high, rank(self.d.volume), 5)

    def gtja_063(self) -> Panel:
        change = self.d.close - delay(self.d.close, 1)
        return _safe_div(sma_cn(change.clip(lower=0), 6, 1), sma_cn(change.abs(), 6, 1)) * 100.0

    def gtja_064(self) -> Panel:
        left = rank(decay_linear(correlation(rank(self.d.vwap), rank(self.d.volume), 4), 4))
        right = rank(
            decay_linear(
                ts_max(correlation(rank(self.d.close), rank(mean(self.d.volume, 60)), 4), 13),
                14,
            )
        )
        return -element_max(left, right)

    def gtja_065(self) -> Panel:
        return _safe_div(mean(self.d.close, 6), self.d.close)

    def gtja_066(self) -> Panel:
        average = mean(self.d.close, 6)
        return _safe_div(self.d.close - average, average) * 100.0

    def gtja_067(self) -> Panel:
        change = self.d.close - delay(self.d.close, 1)
        return _safe_div(sma_cn(change.clip(lower=0), 24, 1), sma_cn(change.abs(), 24, 1)) * 100.0

    def gtja_068(self) -> Panel:
        midpoint = (self.d.high + self.d.low) / 2.0
        value = _safe_div(
            (midpoint - delay(midpoint, 1)) * (self.d.high - self.d.low),
            self.d.volume,
        )
        return sma_cn(value, 15, 2)

    def gtja_069(self) -> Panel:
        dtm = ts_sum(self._dtm(), 20)
        dbm = ts_sum(self._dbm(), 20)
        difference = dtm - dbm
        result = _safe_div(difference, dbm)
        result = result.where(dtm <= dbm, _safe_div(difference, dtm))
        return result.where(dtm != dbm, 0.0)

    def gtja_070(self) -> Panel:
        return stddev(self.d.amount, 6)

    def gtja_071(self) -> Panel:
        average = mean(self.d.close, 24)
        return _safe_div(self.d.close - average, average) * 100.0

    def gtja_072(self) -> Panel:
        value = _safe_div(
            ts_max(self.d.high, 6) - self.d.close,
            ts_max(self.d.high, 6) - ts_min(self.d.low, 6),
        ) * 100.0
        return sma_cn(value, 15, 1)

    def gtja_073(self) -> Panel:
        first = ts_rank(
            decay_linear(decay_linear(correlation(self.d.close, self.d.volume, 10), 16), 4),
            5,
        )
        second = rank(decay_linear(correlation(self.d.vwap, mean(self.d.volume, 30), 4), 3))
        return -(first - second)

    def gtja_074(self) -> Panel:
        first = rank(
            correlation(
                ts_sum(self.d.low * 0.35 + self.d.vwap * 0.65, 20),
                ts_sum(mean(self.d.volume, 40), 20),
                7,
            )
        )
        second = rank(correlation(rank(self.d.vwap), rank(self.d.volume), 6))
        return first + second

    def gtja_075(self) -> Panel:
        benchmark_open, benchmark_close = self._require_external(
            "gtja_075", "benchmark_open", "benchmark_close"
        )
        benchmark_down = benchmark_close < benchmark_open
        down_panel = _broadcast_series(benchmark_down.astype(float), self.d.close).astype(bool)
        numerator = count((self.d.close > self.d.open) & down_panel, 50)
        denominator = count(down_panel, 50)
        return _safe_div(numerator, denominator)

    def gtja_076(self) -> Panel:
        value = _safe_div(self.d.returns.abs(), self.d.volume)
        return _safe_div(stddev(value, 20), mean(value, 20))

    def gtja_077(self) -> Panel:
        midpoint = (self.d.high + self.d.low) / 2.0
        left = rank(decay_linear(midpoint - self.d.vwap, 20))
        right = rank(decay_linear(correlation(midpoint, mean(self.d.volume, 40), 3), 6))
        return element_min(left, right)

    def gtja_078(self) -> Panel:
        typical = (self.d.high + self.d.low + self.d.close) / 3.0
        typical_mean = mean(typical, 12)
        denominator = mean((self.d.close - typical_mean).abs(), 12) * 0.015
        return _safe_div(typical - typical_mean, denominator)

    def gtja_079(self) -> Panel:
        change = self.d.close - delay(self.d.close, 1)
        return _safe_div(sma_cn(change.clip(lower=0), 12, 1), sma_cn(change.abs(), 12, 1)) * 100.0

    def gtja_080(self) -> Panel:
        previous = delay(self.d.volume, 5)
        return _safe_div(self.d.volume - previous, previous) * 100.0

    def gtja_081(self) -> Panel:
        return sma_cn(self.d.volume, 21, 2)

    def gtja_082(self) -> Panel:
        value = _safe_div(ts_max(self.d.high, 6) - self.d.close, ts_max(self.d.high, 6) - ts_min(self.d.low, 6)) * 100
        return sma_cn(value, 20, 1)

    def gtja_083(self) -> Panel:
        return -rank(covariance(rank(self.d.high), rank(self.d.volume), 5))

    def gtja_084(self) -> Panel:
        return ts_sum(self._signed_volume(), 20)

    def gtja_085(self) -> Panel:
        return ts_rank(_safe_div(self.d.volume, mean(self.d.volume, 20)), 20) * ts_rank(-delta(self.d.close, 7), 8)

    def gtja_086(self) -> Panel:
        slope = (delay(self.d.close, 20) - delay(self.d.close, 10)) / 10 - (delay(self.d.close, 10) - self.d.close) / 10
        middle = -(self.d.close - delay(self.d.close, 1))
        result = middle.where(slope >= 0, 1.0)
        return result.where(slope <= 0.25, -1.0).where(slope.notna())

    def gtja_087(self) -> Panel:
        first = rank(decay_linear(delta(self.d.vwap, 4), 7))
        second_input = _safe_div(self.d.low - self.d.vwap, self.d.open - (self.d.high + self.d.low) / 2)
        return -(first + ts_rank(decay_linear(second_input, 11), 7))

    def gtja_088(self) -> Panel:
        previous = delay(self.d.close, 20)
        return _safe_div(self.d.close - previous, previous) * 100

    def gtja_089(self) -> Panel:
        difference = sma_cn(self.d.close, 13, 2) - sma_cn(self.d.close, 27, 2)
        return 2 * (difference - sma_cn(difference, 10, 2))

    def gtja_090(self) -> Panel:
        return -rank(correlation(rank(self.d.vwap), rank(self.d.volume), 5))

    def gtja_091(self) -> Panel:
        return -rank(self.d.close - ts_max(self.d.close, 5)) * rank(correlation(mean(self.d.volume, 40), self.d.low, 5))

    def gtja_092(self) -> Panel:
        left = rank(decay_linear(delta(self.d.close * .35 + self.d.vwap * .65, 2), 3))
        right = ts_rank(decay_linear(correlation(mean(self.d.volume, 180), self.d.close, 13).abs(), 5), 15)
        return -element_max(left, right)

    def gtja_093(self) -> Panel:
        previous = delay(self.d.open, 1)
        value = element_max(self.d.open - self.d.low, self.d.open - previous)
        return ts_sum(value.where(self.d.open < previous, 0.0), 20)

    def gtja_094(self) -> Panel:
        return ts_sum(self._signed_volume(), 30)

    def gtja_095(self) -> Panel:
        return stddev(self.d.amount, 20)

    def gtja_096(self) -> Panel:
        stochastic = _safe_div(self.d.close - ts_min(self.d.low, 9), ts_max(self.d.high, 9) - ts_min(self.d.low, 9)) * 100
        return sma_cn(sma_cn(stochastic, 3, 1), 3, 1)

    def gtja_097(self) -> Panel:
        return stddev(self.d.volume, 10)

    def gtja_098(self) -> Panel:
        average = mean(self.d.close, 100)
        slope = _safe_div(delta(average, 100), delay(self.d.close, 100))
        return (-(self.d.close - ts_min(self.d.close, 100))).where(slope <= .05, -delta(self.d.close, 3))

    def gtja_099(self) -> Panel:
        return -rank(covariance(rank(self.d.close), rank(self.d.volume), 5))

    def gtja_100(self) -> Panel:
        return stddev(self.d.volume, 20)

    def gtja_101(self) -> Panel:
        left = rank(correlation(self.d.close, ts_sum(mean(self.d.volume, 30), 37), 15))
        right = rank(correlation(rank(self.d.high * .1 + self.d.vwap * .9), rank(self.d.volume), 11))
        return _conditional(left < right, -1.0, 0.0, valid=left.notna() & right.notna())

    def gtja_102(self) -> Panel:
        change = self.d.volume - delay(self.d.volume, 1)
        return _safe_div(sma_cn(change.clip(lower=0), 6, 1), sma_cn(change.abs(), 6, 1)) * 100

    def gtja_103(self) -> Panel:
        return (20 - lowday(self.d.low, 20)) / 20 * 100

    def gtja_104(self) -> Panel:
        return -delta(correlation(self.d.high, self.d.volume, 5), 5) * rank(stddev(self.d.close, 20))

    def gtja_105(self) -> Panel:
        return -correlation(rank(self.d.open), rank(self.d.volume), 10)

    def gtja_106(self) -> Panel:
        return self.d.close - delay(self.d.close, 20)

    def gtja_107(self) -> Panel:
        return -rank(self.d.open - delay(self.d.high, 1)) * rank(self.d.open - delay(self.d.close, 1)) * rank(self.d.open - delay(self.d.low, 1))

    def gtja_108(self) -> Panel:
        base = rank(self.d.high - ts_min(self.d.high, 2))
        exponent = rank(correlation(self.d.vwap, mean(self.d.volume, 120), 6))
        return -np.power(base, exponent)

    def gtja_109(self) -> Panel:
        first = sma_cn(self.d.high - self.d.low, 10, 2)
        return _safe_div(first, sma_cn(first, 10, 2))

    def gtja_110(self) -> Panel:
        previous = delay(self.d.close, 1)
        numerator = ts_sum((self.d.high - previous).clip(lower=0), 20)
        denominator = ts_sum((previous - self.d.low).clip(lower=0), 20)
        return _safe_div(numerator, denominator) * 100

    def gtja_111(self) -> Panel:
        raw = _safe_div(((self.d.close - self.d.low) - (self.d.high - self.d.close)) * self.d.volume, self.d.high - self.d.low)
        return sma_cn(raw, 11, 2) - sma_cn(raw, 4, 2)

    def gtja_112(self) -> Panel:
        change = self.d.close - delay(self.d.close, 1)
        positive = ts_sum(change.clip(lower=0), 12)
        negative = ts_sum((-change).clip(lower=0), 12)
        return _safe_div(positive - negative, positive + negative) * 100

    def gtja_113(self) -> Panel:
        return -rank(mean(delay(self.d.close, 5), 20)) * correlation(self.d.close, self.d.volume, 2) * rank(correlation(ts_sum(self.d.close, 5), ts_sum(self.d.close, 20), 2))

    def gtja_114(self) -> Panel:
        ratio = _safe_div(self.d.high - self.d.low, mean(self.d.close, 5))
        return _safe_div(rank(delay(ratio, 2)) * rank(rank(self.d.volume)), _safe_div(ratio, self.d.vwap - self.d.close))

    def gtja_115(self) -> Panel:
        base = rank(correlation(self.d.high * .9 + self.d.close * .1, mean(self.d.volume, 30), 10))
        exponent = rank(correlation(ts_rank((self.d.high + self.d.low) / 2, 4), ts_rank(self.d.volume, 10), 7))
        return np.power(base, exponent)

    def gtja_116(self) -> Panel:
        return _sequence_regbeta(self.d.close, 20)

    def gtja_117(self) -> Panel:
        return ts_rank(self.d.volume, 32) * (1 - ts_rank(self.d.close + self.d.high - self.d.low, 16)) * (1 - ts_rank(self.d.returns, 32))

    def gtja_118(self) -> Panel:
        return _safe_div(ts_sum(self.d.high - self.d.open, 20), ts_sum(self.d.open - self.d.low, 20)) * 100

    def gtja_119(self) -> Panel:
        left = rank(decay_linear(correlation(self.d.vwap, ts_sum(mean(self.d.volume, 5), 26), 5), 7))
        inner = ts_min(correlation(rank(self.d.open), rank(mean(self.d.volume, 15)), 21), 9)
        right = rank(decay_linear(ts_rank(inner, 7), 8))
        return left - right

    def gtja_120(self) -> Panel:
        return _safe_div(rank(self.d.vwap - self.d.close), rank(self.d.vwap + self.d.close))

    def gtja_121(self) -> Panel:
        base = rank(self.d.vwap - ts_min(self.d.vwap, 12))
        exponent = ts_rank(correlation(ts_rank(self.d.vwap, 20), ts_rank(mean(self.d.volume, 60), 2), 18), 3)
        return -np.power(base, exponent)

    def gtja_122(self) -> Panel:
        smoothed = sma_cn(sma_cn(sma_cn(np.log(self.d.close), 13, 2), 13, 2), 13, 2)
        return _safe_div(smoothed - delay(smoothed, 1), delay(smoothed, 1))

    def gtja_123(self) -> Panel:
        left = rank(correlation(ts_sum((self.d.high + self.d.low) / 2, 20), ts_sum(mean(self.d.volume, 60), 20), 9))
        right = rank(correlation(self.d.low, self.d.volume, 6))
        return _conditional(left < right, -1.0, 0.0, valid=left.notna() & right.notna())

    def gtja_124(self) -> Panel:
        return _safe_div(self.d.close - self.d.vwap, decay_linear(rank(ts_max(self.d.close, 30)), 2))

    def gtja_125(self) -> Panel:
        numerator = rank(decay_linear(correlation(self.d.vwap, mean(self.d.volume, 80), 17), 20))
        denominator = rank(decay_linear(delta(self.d.close * .5 + self.d.vwap * .5, 3), 16))
        return _safe_div(numerator, denominator)

    def gtja_126(self) -> Panel:
        return (self.d.close + self.d.high + self.d.low) / 3

    def gtja_127(self) -> Panel:
        maximum = ts_max(self.d.close, 12)
        ratio = _safe_div(100 * (self.d.close - maximum), maximum)
        return np.sqrt(mean(ratio.pow(2), 12))

    def gtja_128(self) -> Panel:
        typical = (self.d.high + self.d.low + self.d.close) / 3
        previous = delay(typical, 1)
        up = ts_sum((typical * self.d.volume).where(typical > previous, 0.0), 14)
        down = ts_sum((typical * self.d.volume).where(typical < previous, 0.0), 14)
        return 100 - _safe_div(pd.DataFrame(100.0, index=up.index, columns=up.columns), 1 + _safe_div(up, down))

    def gtja_129(self) -> Panel:
        change = self.d.close - delay(self.d.close, 1)
        return ts_sum((-change).where(change < 0, 0.0), 12)

    def gtja_130(self) -> Panel:
        numerator = rank(decay_linear(correlation((self.d.high + self.d.low) / 2, mean(self.d.volume, 40), 9), 10))
        denominator = rank(decay_linear(correlation(rank(self.d.vwap), rank(self.d.volume), 7), 3))
        return _safe_div(numerator, denominator)

    def gtja_131(self) -> Panel:
        base = rank(delta(self.d.vwap, 1))
        exponent = ts_rank(correlation(self.d.close, mean(self.d.volume, 50), 18), 18)
        return np.power(base, exponent)

    def gtja_132(self) -> Panel:
        return mean(self.d.amount, 20)

    def gtja_133(self) -> Panel:
        return (20 - highday(self.d.high, 20)) / 20 * 100 - (20 - lowday(self.d.low, 20)) / 20 * 100

    def gtja_134(self) -> Panel:
        previous = delay(self.d.close, 12)
        return _safe_div(self.d.close - previous, previous) * self.d.volume

    def gtja_135(self) -> Panel:
        return sma_cn(delay(_safe_div(self.d.close, delay(self.d.close, 20)), 1), 20, 1)

    def gtja_136(self) -> Panel:
        return -rank(delta(self.d.returns, 3)) * correlation(self.d.open, self.d.volume, 10)

    def gtja_137(self) -> Panel:
        previous_close = delay(self.d.close, 1); previous_open = delay(self.d.open, 1); previous_low = delay(self.d.low, 1)
        a = (self.d.high - previous_close).abs(); b = (self.d.low - previous_close).abs(); c = (self.d.high - previous_low).abs(); d = (previous_close - previous_open).abs()
        denominator = (c + d / 4).where(~((b > c) & (b > a)), b + a / 2 + d / 4)
        denominator = denominator.where(~((a > b) & (a > c)), a + b / 2 + d / 4)
        numerator = 16 * (self.d.close - previous_close + (self.d.close - self.d.open) / 2 + previous_close - previous_open)
        return _safe_div(numerator, denominator) * element_max(a, b)

    def gtja_138(self) -> Panel:
        left = rank(decay_linear(delta(self.d.low * .7 + self.d.vwap * .3, 3), 20))
        inner = correlation(ts_rank(self.d.low, 8), ts_rank(mean(self.d.volume, 60), 17), 5)
        right = ts_rank(decay_linear(ts_rank(inner, 19), 16), 7)
        return -(left - right)

    def gtja_139(self) -> Panel:
        return -correlation(self.d.open, self.d.volume, 10)

    def gtja_140(self) -> Panel:
        left = rank(decay_linear((rank(self.d.open) + rank(self.d.low)) - (rank(self.d.high) + rank(self.d.close)), 8))
        right = ts_rank(decay_linear(correlation(ts_rank(self.d.close, 8), ts_rank(mean(self.d.volume, 60), 20), 8), 7), 3)
        return element_min(left, right)

    def gtja_141(self) -> Panel:
        return -rank(correlation(rank(self.d.high), rank(mean(self.d.volume, 15)), 9))

    def gtja_142(self) -> Panel:
        return -rank(ts_rank(self.d.close, 10)) * rank(delta(delta(self.d.close, 1), 1)) * rank(ts_rank(_safe_div(self.d.volume, mean(self.d.volume, 20)), 5))

    def gtja_143(self) -> Panel:
        output = pd.DataFrame(np.nan, index=self.d.close.index, columns=self.d.close.columns)
        returns = _safe_div(self.d.close - delay(self.d.close, 1), delay(self.d.close, 1))
        for column_index, column in enumerate(output.columns):
            previous = 1.0
            for row_index in range(len(output)):
                value = returns.iat[row_index, column_index]
                if pd.notna(value) and value > 0:
                    previous *= 1 + float(value)
                output.iat[row_index, column_index] = previous
        return output

    def gtja_144(self) -> Panel:
        value = _safe_div(self.d.returns.abs(), self.d.amount)
        condition = self.d.close < delay(self.d.close, 1)
        return _safe_div(sumif(value, 20, condition), count(condition, 20))

    def gtja_145(self) -> Panel:
        return _safe_div(mean(self.d.volume, 9) - mean(self.d.volume, 26), mean(self.d.volume, 12)) * 100

    def gtja_146(self) -> Panel:
        smoothed = sma_cn(self.d.returns, 61, 2)
        deviation = self.d.returns - smoothed
        return _safe_div(mean(deviation, 20) * deviation, sma_cn(deviation.pow(2), 60, 2))

    def gtja_147(self) -> Panel:
        return _sequence_regbeta(mean(self.d.close, 12), 12)

    def gtja_148(self) -> Panel:
        left = rank(correlation(self.d.open, ts_sum(mean(self.d.volume, 60), 9), 6))
        right = rank(self.d.open - ts_min(self.d.open, 14))
        return _conditional(left < right, -1.0, 0.0, valid=left.notna() & right.notna())

    def gtja_149(self) -> Panel:
        (benchmark_close,) = self._require_external("gtja_149", "benchmark_close")
        benchmark = _broadcast_series(benchmark_close, self.d.close)
        benchmark_return = _safe_div(benchmark - delay(benchmark, 1), delay(benchmark, 1))
        down = benchmark_return < 0
        stock_return = self.d.returns.where(down)
        return regbeta(stock_return, benchmark_return.where(down), 252)

    def gtja_150(self) -> Panel:
        return (self.d.close + self.d.high + self.d.low) / 3 * self.d.volume

    def gtja_151(self) -> Panel:
        return sma_cn(self.d.close - delay(self.d.close, 20), 20, 1)

    def gtja_152(self) -> Panel:
        inner = delay(sma_cn(delay(_safe_div(self.d.close, delay(self.d.close, 9)), 1), 9, 1), 1)
        return sma_cn(mean(inner, 12) - mean(inner, 26), 9, 1)

    def gtja_153(self) -> Panel:
        return (mean(self.d.close, 3) + mean(self.d.close, 6) + mean(self.d.close, 12) + mean(self.d.close, 24)) / 4

    def gtja_154(self) -> Panel:
        left = self.d.vwap - ts_min(self.d.vwap, 16)
        right = correlation(self.d.vwap, mean(self.d.volume, 180), 18)
        return _conditional(left < right, 1.0, 0.0, valid=left.notna() & right.notna())

    def gtja_155(self) -> Panel:
        difference = sma_cn(self.d.volume, 13, 2) - sma_cn(self.d.volume, 27, 2)
        return difference - sma_cn(difference, 10, 2)

    def gtja_156(self) -> Panel:
        left = rank(decay_linear(delta(self.d.vwap, 5), 3))
        mixed = self.d.open * .15 + self.d.low * .85
        right = rank(decay_linear(-_safe_div(delta(mixed, 2), mixed), 3))
        return -element_max(left, right)

    def gtja_157(self) -> Panel:
        inner = -rank(delta(self.d.close - 1, 5))
        first = ts_min(product(rank(rank(scale(np.log(ts_sum(ts_min(rank(rank(inner)), 2), 1))))), 1), 5)
        return first + ts_rank(delay(-self.d.returns, 6), 5)

    def gtja_158(self) -> Panel:
        smooth = sma_cn(self.d.close, 15, 2)
        return _safe_div((self.d.high - smooth) - (self.d.low - smooth), self.d.close)

    def gtja_159(self) -> Panel:
        previous = delay(self.d.close, 1)
        low = element_min(self.d.low, previous); high = element_max(self.d.high, previous)
        def component(window: int) -> Panel:
            return _safe_div(self.d.close - ts_sum(low, window), ts_sum(high - low, window))
        value = component(6) * 12 * 24 + component(12) * 6 * 24 + component(24) * 6 * 12
        return value * 100 / (6 * 12 + 6 * 24 + 12 * 24)

    def gtja_160(self) -> Panel:
        volatility = stddev(self.d.close, 20)
        return sma_cn(volatility.where(self.d.close <= delay(self.d.close, 1), 0.0), 20, 1)

    def _true_range(self) -> Panel:
        previous = delay(self.d.close, 1)
        return element_max(
            element_max(self.d.high - self.d.low, (self.d.high - previous).abs()),
            (self.d.low - previous).abs(),
        )

    def _adx(self) -> Panel:
        hd = self.d.high - delay(self.d.high, 1)
        ld = delay(self.d.low, 1) - self.d.low
        tr_sum = ts_sum(self._true_range(), 14)
        plus = _safe_div(ts_sum(ld.where((ld > 0) & (ld > hd), 0.0), 14) * 100, tr_sum)
        minus = _safe_div(ts_sum(hd.where((hd > 0) & (hd > ld), 0.0), 14) * 100, tr_sum)
        return mean(_safe_div((plus - minus).abs(), plus + minus) * 100, 6)

    def _accumulated_deviation(self, window: int) -> Panel:
        accumulated = ts_sum(self.d.close - mean(self.d.close, window), window)
        return _safe_div(ts_max(accumulated, window) - ts_min(accumulated, window), stddev(self.d.close, window))

    def gtja_161(self) -> Panel:
        return mean(self._true_range(), 12)

    def gtja_162(self) -> Panel:
        change = self.d.close - delay(self.d.close, 1)
        rsi = _safe_div(sma_cn(change.clip(lower=0), 12, 1), sma_cn(change.abs(), 12, 1)) * 100
        return _safe_div(rsi - ts_min(rsi, 12), ts_max(rsi, 12) - ts_min(rsi, 12))

    def gtja_163(self) -> Panel:
        return rank(-self.d.returns * mean(self.d.volume, 20) * self.d.vwap * (self.d.high - self.d.close))

    def gtja_164(self) -> Panel:
        change = self.d.close - delay(self.d.close, 1)
        reciprocal = _safe_div(pd.DataFrame(1.0, index=change.index, columns=change.columns), change).where(change > 0, 1.0)
        value = _safe_div((reciprocal - ts_min(reciprocal, 12)) * 100, self.d.high - self.d.low)
        return sma_cn(value, 13, 2)

    def gtja_165(self) -> Panel:
        return self._accumulated_deviation(48)

    def gtja_166(self) -> Panel:
        centered = self.d.returns - mean(self.d.returns, 20)
        numerator = -20 * (19 ** 1.5) * ts_sum(centered.pow(3), 20)
        denominator = 19 * 18 * ts_sum(centered.pow(2), 20).pow(1.5)
        return _safe_div(numerator, denominator)

    def gtja_167(self) -> Panel:
        change = self.d.close - delay(self.d.close, 1)
        return ts_sum(change.clip(lower=0), 12)

    def gtja_168(self) -> Panel:
        return -_safe_div(self.d.volume, mean(self.d.volume, 20))

    def gtja_169(self) -> Panel:
        inner = delay(sma_cn(self.d.close - delay(self.d.close, 1), 9, 1), 1)
        return sma_cn(mean(inner, 12) - mean(inner, 26), 10, 1)

    def gtja_170(self) -> Panel:
        first = _safe_div(rank(_safe_div(pd.DataFrame(1.0, index=self.d.close.index, columns=self.d.close.columns), self.d.close)) * self.d.volume, mean(self.d.volume, 20))
        second = _safe_div(self.d.high * rank(self.d.high - self.d.close), mean(self.d.high, 5))
        return first * second - rank(self.d.vwap - delay(self.d.vwap, 5))

    def gtja_171(self) -> Panel:
        numerator = -(self.d.low - self.d.close) * self.d.open.pow(5)
        denominator = (self.d.close - self.d.high) * self.d.close.pow(5)
        return _safe_div(numerator, denominator)

    def gtja_172(self) -> Panel:
        return self._adx()

    def gtja_173(self) -> Panel:
        first = sma_cn(self.d.close, 13, 2)
        second = sma_cn(first, 13, 2)
        log_first = sma_cn(np.log(self.d.close), 13, 2)
        return 3 * first - 2 * second + sma_cn(sma_cn(log_first, 13, 2), 13, 2)

    def gtja_174(self) -> Panel:
        volatility = stddev(self.d.close, 20)
        return sma_cn(volatility.where(self.d.close > delay(self.d.close, 1), 0.0), 20, 1)

    def gtja_175(self) -> Panel:
        return mean(self._true_range(), 6)

    def gtja_176(self) -> Panel:
        stochastic = _safe_div(self.d.close - ts_min(self.d.low, 12), ts_max(self.d.high, 12) - ts_min(self.d.low, 12))
        return correlation(rank(stochastic), rank(self.d.volume), 6)

    def gtja_177(self) -> Panel:
        return (20 - highday(self.d.high, 20)) / 20 * 100

    def gtja_178(self) -> Panel:
        return self.d.returns * self.d.volume

    def gtja_179(self) -> Panel:
        return rank(correlation(self.d.vwap, self.d.volume, 4)) * rank(correlation(rank(self.d.low), rank(mean(self.d.volume, 50)), 12))

    def gtja_180(self) -> Panel:
        change = delta(self.d.close, 7)
        active = -ts_rank(change.abs(), 60) * np.sign(change)
        return active.where(self.d.volume > mean(self.d.volume, 20), -self.d.volume)

    def gtja_181(self) -> Panel:
        (benchmark_close,) = self._require_external("gtja_181", "benchmark_close")
        benchmark = _broadcast_series(benchmark_close, self.d.close)
        stock_deviation = self.d.returns - mean(self.d.returns, 20)
        benchmark_deviation = benchmark - mean(benchmark, 20)
        return _safe_div(ts_sum(stock_deviation - benchmark_deviation.pow(2), 20), ts_sum(benchmark_deviation.pow(3), 20))

    def gtja_182(self) -> Panel:
        benchmark_open, benchmark_close = self._require_external("gtja_182", "benchmark_open", "benchmark_close")
        b_open = _broadcast_series(benchmark_open, self.d.close)
        b_close = _broadcast_series(benchmark_close, self.d.close)
        same = ((self.d.close > self.d.open) & (b_close > b_open)) | ((self.d.close < self.d.open) & (b_close < b_open))
        return count(same, 20) / 20

    def gtja_183(self) -> Panel:
        return self._accumulated_deviation(24)

    def gtja_184(self) -> Panel:
        return rank(correlation(delay(self.d.open - self.d.close, 1), self.d.close, 200)) + rank(self.d.open - self.d.close)

    def gtja_185(self) -> Panel:
        return rank(-np.power(1 - _safe_div(self.d.open, self.d.close), 2))

    def gtja_186(self) -> Panel:
        adx = self._adx()
        return (adx + delay(adx, 6)) / 2

    def gtja_187(self) -> Panel:
        previous = delay(self.d.open, 1)
        value = element_max(self.d.high - self.d.open, self.d.open - previous)
        return ts_sum(value.where(self.d.open > previous, 0.0), 20)

    def gtja_188(self) -> Panel:
        price_range = self.d.high - self.d.low
        smooth = sma_cn(price_range, 11, 2)
        return _safe_div(price_range - smooth, smooth) * 100

    def gtja_189(self) -> Panel:
        return mean((self.d.close - mean(self.d.close, 6)).abs(), 6)

    def gtja_190(self) -> Panel:
        daily = _safe_div(self.d.close, delay(self.d.close, 1)) - 1
        threshold = np.power(_safe_div(self.d.close, delay(self.d.close, 19)), 1 / 20) - 1
        squared = (daily - threshold).pow(2)
        above = daily > threshold
        below = daily < threshold
        numerator = (count(above, 20) - 1) * sumif(squared, 20, below)
        denominator = count(below, 20) * sumif(squared, 20, above)
        return np.log(_safe_div(numerator, denominator))

    def gtja_191(self) -> Panel:
        return correlation(mean(self.d.volume, 20), self.d.low, 5) + (self.d.high + self.d.low) / 2 - self.d.close
