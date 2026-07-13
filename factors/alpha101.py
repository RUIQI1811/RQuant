"""WorldQuant 101 Formulaic Alphas for panel-form daily A-share data.

The formulas follow the Alpha101 page linked in the project documentation.  A
panel is represented by a DataFrame whose rows are trading dates and columns
are symbols.  Cross-sectional operators therefore work across columns and
time-series operators work down rows.

The original formulas contain fractional lookback constants.  They are rounded
to the nearest positive trading-day integer before being passed to pandas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from factors.operators import (
    correlation,
    covariance,
    decay_linear,
    delay,
    delta,
    element_max,
    element_min,
    product,
    rank,
    replace_inf as _replace_inf,
    safe_div as _safe_div,
    scale,
    signed_power,
    stddev,
    ts_argmax,
    ts_argmin,
    ts_max,
    ts_min,
    ts_rank,
    ts_sum,
    window as _window,
)


Panel = pd.DataFrame


class Alpha101DataError(ValueError):
    """Raised when an alpha requires a field that is not available."""


def _signed_condition(
    condition: Panel,
    *,
    true_value: float = 1.0,
    false_value: float = -1.0,
    valid: Panel | None = None,
) -> Panel:
    out = pd.DataFrame(
        np.where(condition, true_value, false_value),
        index=condition.index,
        columns=condition.columns,
        dtype=float,
    )
    return out if valid is None else out.where(valid)


def indneutralize(value: Panel, groups: Panel | None, group_name: str) -> Panel:
    """Demean each date cross-section within an industry classification."""
    if groups is None or groups.isna().all().all():
        raise Alpha101DataError(
            f"industry-neutral alpha requires '{group_name}' classification data"
        )
    aligned = groups.reindex(index=value.index, columns=value.columns)
    out = value.copy() * np.nan
    for date in value.index:
        row = pd.DataFrame({"value": value.loc[date], "group": aligned.loc[date]})
        out.loc[date] = row["value"] - row.groupby("group", dropna=True)["value"].transform("mean")
    return out


@dataclass(frozen=True)
class Alpha101Panels:
    open: Panel
    close: Panel
    high: Panel
    low: Panel
    volume: Panel
    vwap: Panel
    returns: Panel
    cap: Panel | None = None
    sector: Panel | None = None
    industry: Panel | None = None
    subindustry: Panel | None = None
    is_st: Panel | None = None
    turnover_value: Panel | None = None

    def adv(self, periods: float | int) -> Panel:
        return ts_sum(self.volume, periods) / _window(periods)


class Alpha101:
    """Calculate any of the 101 WorldQuant alphas on aligned wide panels."""

    def __init__(self, data: Alpha101Panels) -> None:
        self.d = data

    @property
    def names(self) -> tuple[str, ...]:
        return ALPHA101_NAMES

    def calculate(self, name: str | int) -> Panel:
        normalized = normalize_alpha_name(name)
        method = getattr(self, normalized, None)
        if method is None:
            raise KeyError(f"unknown Alpha101 factor: {name}")
        return _replace_inf(method())

    def calculate_many(
        self,
        names: Iterable[str | int] | None = None,
        *,
        on_error: str = "raise",
    ) -> dict[str, Panel]:
        selected = ALPHA101_NAMES if names is None else tuple(normalize_alpha_name(name) for name in names)
        output: dict[str, Panel] = {}
        for name in selected:
            try:
                output[name] = self.calculate(name)
            except Alpha101DataError:
                if on_error == "nan":
                    output[name] = self.d.close.copy() * np.nan
                else:
                    raise
        return output

    def _neutral(self, value: Panel, level: str) -> Panel:
        return indneutralize(value, getattr(self.d, level), level)

    def alpha_001(self) -> Panel:
        x = self.d.close.copy()
        negative = self.d.returns < 0
        x = x.where(~negative, stddev(self.d.returns, 20))
        return rank(ts_argmax(signed_power(x, 2.0), 5)) - 0.5

    def alpha_002(self) -> Panel:
        return -correlation(rank(delta(np.log(self.d.volume.replace(0, np.nan)), 2)), rank(_safe_div(self.d.close - self.d.open, self.d.open)), 6)

    def alpha_003(self) -> Panel:
        return -correlation(rank(self.d.open), rank(self.d.volume), 10)

    def alpha_004(self) -> Panel:
        return -ts_rank(rank(self.d.low), 9)

    def alpha_005(self) -> Panel:
        return rank(self.d.open - ts_sum(self.d.vwap, 10) / 10) * -rank(self.d.close - self.d.vwap).abs()

    def alpha_006(self) -> Panel:
        return -correlation(self.d.open, self.d.volume, 10)

    def alpha_007(self) -> Panel:
        change = delta(self.d.close, 7)
        value = -ts_rank(change.abs(), 60) * np.sign(change)
        return value.where(self.d.volume > self.d.adv(20), -1.0)

    def alpha_008(self) -> Panel:
        x = ts_sum(self.d.open, 5) * ts_sum(self.d.returns, 5)
        return -rank(x - delay(x, 10))

    def alpha_009(self) -> Panel:
        change = delta(self.d.close, 1)
        return change.where(ts_min(change, 5) > 0, change.where(ts_max(change, 5) < 0, -change))

    def alpha_010(self) -> Panel:
        change = delta(self.d.close, 1)
        value = change.where(ts_min(change, 4) > 0, change.where(ts_max(change, 4) < 0, -change))
        return rank(value)

    def alpha_011(self) -> Panel:
        spread = self.d.vwap - self.d.close
        return (rank(ts_max(spread, 3)) + rank(ts_min(spread, 3))) * rank(delta(self.d.volume, 3))

    def alpha_012(self) -> Panel:
        return np.sign(delta(self.d.volume, 1)) * -delta(self.d.close, 1)

    def alpha_013(self) -> Panel:
        return -rank(covariance(rank(self.d.close), rank(self.d.volume), 5))

    def alpha_014(self) -> Panel:
        return -rank(delta(self.d.returns, 3)) * correlation(self.d.open, self.d.volume, 10)

    def alpha_015(self) -> Panel:
        return -ts_sum(rank(correlation(rank(self.d.high), rank(self.d.volume), 3)), 3)

    def alpha_016(self) -> Panel:
        return -rank(covariance(rank(self.d.high), rank(self.d.volume), 5))

    def alpha_017(self) -> Panel:
        return -rank(ts_rank(self.d.close, 10)) * rank(delta(delta(self.d.close, 1), 1)) * rank(ts_rank(_safe_div(self.d.volume, self.d.adv(20)), 5))

    def alpha_018(self) -> Panel:
        return -rank(stddev((self.d.close - self.d.open).abs(), 5) + (self.d.close - self.d.open) + correlation(self.d.close, self.d.open, 10))

    def alpha_019(self) -> Panel:
        return -np.sign((self.d.close - delay(self.d.close, 7)) + delta(self.d.close, 7)) * (1 + rank(1 + ts_sum(self.d.returns, 250)))

    def alpha_020(self) -> Panel:
        return -rank(self.d.open - delay(self.d.high, 1)) * rank(self.d.open - delay(self.d.close, 1)) * rank(self.d.open - delay(self.d.low, 1))

    def alpha_021(self) -> Panel:
        mean8 = ts_sum(self.d.close, 8) / 8
        mean2 = ts_sum(self.d.close, 2) / 2
        result = pd.DataFrame(-1.0, index=self.d.close.index, columns=self.d.close.columns)
        result = result.where(~(mean2 < mean8 - stddev(self.d.close, 8)), 1.0)
        result = result.where(~(mean8 + stddev(self.d.close, 8) < mean2), -1.0)
        middle = ~((mean8 + stddev(self.d.close, 8) < mean2) | (mean2 < mean8 - stddev(self.d.close, 8)))
        ratio = _safe_div(self.d.volume, self.d.adv(20))
        result = result.where(~middle, _signed_condition(ratio >= 1, valid=ratio.notna()))
        return result

    def alpha_022(self) -> Panel:
        return -delta(correlation(self.d.high, self.d.volume, 5), 5) * rank(stddev(self.d.close, 20))

    def alpha_023(self) -> Panel:
        return -delta(self.d.high, 2).where(ts_sum(self.d.high, 20) / 20 < self.d.high, 0.0)

    def alpha_024(self) -> Panel:
        slope = _safe_div(delta(ts_sum(self.d.close, 100) / 100, 100), delay(self.d.close, 100))
        return (-(self.d.close - ts_min(self.d.close, 100))).where(slope <= 0.05, -delta(self.d.close, 3))

    def alpha_025(self) -> Panel:
        return rank(-self.d.returns * self.d.adv(20) * self.d.vwap * (self.d.high - self.d.close))

    def alpha_026(self) -> Panel:
        return -ts_max(correlation(ts_rank(self.d.volume, 5), ts_rank(self.d.high, 5), 5), 3)

    def alpha_027(self) -> Panel:
        value = rank(ts_sum(correlation(rank(self.d.volume), rank(self.d.vwap), 6), 2) / 2)
        return _signed_condition(value > 0.5, true_value=-1.0, false_value=1.0, valid=value.notna())

    def alpha_028(self) -> Panel:
        return scale(correlation(self.d.adv(20), self.d.low, 5) + (self.d.high + self.d.low) / 2 - self.d.close)

    def alpha_029(self) -> Panel:
        inner = -rank(delta(self.d.close - 1, 5))
        first = ts_min(product(rank(rank(scale(np.log(ts_sum(ts_min(rank(rank(inner)), 2), 1))))), 1), 5)
        return first + ts_rank(delay(-self.d.returns, 6), 5)

    def alpha_030(self) -> Panel:
        signs = np.sign(self.d.close - delay(self.d.close, 1)) + np.sign(delay(self.d.close, 1) - delay(self.d.close, 2)) + np.sign(delay(self.d.close, 2) - delay(self.d.close, 3))
        return _safe_div((1 - rank(signs)) * ts_sum(self.d.volume, 5), ts_sum(self.d.volume, 20))

    def alpha_031(self) -> Panel:
        first = rank(rank(rank(decay_linear(-rank(rank(delta(self.d.close, 10))), 10))))
        return first + rank(-delta(self.d.close, 3)) + np.sign(scale(correlation(self.d.adv(20), self.d.low, 12)))

    def alpha_032(self) -> Panel:
        return scale(ts_sum(self.d.close, 7) / 7 - self.d.close) + 20 * scale(correlation(self.d.vwap, delay(self.d.close, 5), 230))

    def alpha_033(self) -> Panel:
        return rank(-(1 - _safe_div(self.d.open, self.d.close)))

    def alpha_034(self) -> Panel:
        return rank((1 - rank(_safe_div(stddev(self.d.returns, 2), stddev(self.d.returns, 5)))) + (1 - rank(delta(self.d.close, 1))))

    def alpha_035(self) -> Panel:
        return ts_rank(self.d.volume, 32) * (1 - ts_rank((self.d.close + self.d.high) - self.d.low, 16)) * (1 - ts_rank(self.d.returns, 32))

    def alpha_036(self) -> Panel:
        return (
            2.21 * rank(correlation(self.d.close - self.d.open, delay(self.d.volume, 1), 15))
            + 0.7 * rank(self.d.open - self.d.close)
            + 0.73 * rank(ts_rank(delay(-self.d.returns, 6), 5))
            + rank(correlation(self.d.vwap, self.d.adv(20), 6).abs())
            + 0.6 * rank((ts_sum(self.d.close, 200) / 200 - self.d.open) * (self.d.close - self.d.open))
        )

    def alpha_037(self) -> Panel:
        return rank(correlation(delay(self.d.open - self.d.close, 1), self.d.close, 200)) + rank(self.d.open - self.d.close)

    def alpha_038(self) -> Panel:
        return -rank(ts_rank(self.d.close, 10)) * rank(_safe_div(self.d.close, self.d.open))

    def alpha_039(self) -> Panel:
        return -rank(delta(self.d.close, 7) * (1 - rank(decay_linear(_safe_div(self.d.volume, self.d.adv(20)), 9)))) * (1 + rank(ts_sum(self.d.returns, 250)))

    def alpha_040(self) -> Panel:
        return -rank(stddev(self.d.high, 10)) * correlation(self.d.high, self.d.volume, 10)

    def alpha_041(self) -> Panel:
        return np.sqrt(self.d.high * self.d.low) - self.d.vwap

    def alpha_042(self) -> Panel:
        return _safe_div(rank(self.d.vwap - self.d.close), rank(self.d.vwap + self.d.close))

    def alpha_043(self) -> Panel:
        return ts_rank(_safe_div(self.d.volume, self.d.adv(20)), 20) * ts_rank(-delta(self.d.close, 7), 8)

    def alpha_044(self) -> Panel:
        return -correlation(self.d.high, rank(self.d.volume), 5)

    def alpha_045(self) -> Panel:
        return -rank(ts_sum(delay(self.d.close, 5), 20) / 20) * correlation(self.d.close, self.d.volume, 2) * rank(correlation(ts_sum(self.d.close, 5), ts_sum(self.d.close, 20), 2))

    def alpha_046(self) -> Panel:
        slope = (delay(self.d.close, 20) - delay(self.d.close, 10)) / 10 - (delay(self.d.close, 10) - self.d.close) / 10
        value = -(self.d.close - delay(self.d.close, 1))
        value = value.where(slope >= 0, 1.0)
        return value.where(slope <= 0.25, -1.0).where(slope.notna())

    def alpha_047(self) -> Panel:
        first = _safe_div(rank(_safe_div(1.0, self.d.close)) * self.d.volume, self.d.adv(20))
        second = _safe_div(self.d.high * rank(self.d.high - self.d.close), ts_sum(self.d.high, 5) / 5)
        return first * second - rank(self.d.vwap - delay(self.d.vwap, 5))

    def alpha_048(self) -> Panel:
        numerator = correlation(delta(self.d.close, 1), delta(delay(self.d.close, 1), 1), 250) * delta(self.d.close, 1)
        neutral = self._neutral(_safe_div(numerator, self.d.close), "subindustry")
        denominator = ts_sum(signed_power(_safe_div(delta(self.d.close, 1), delay(self.d.close, 1)), 2), 250)
        return _safe_div(neutral, denominator)

    def alpha_049(self) -> Panel:
        slope = (delay(self.d.close, 20) - delay(self.d.close, 10)) / 10 - (delay(self.d.close, 10) - self.d.close) / 10
        return (-(self.d.close - delay(self.d.close, 1))).where(slope >= -0.1, 1.0)

    def alpha_050(self) -> Panel:
        return -ts_max(rank(correlation(rank(self.d.volume), rank(self.d.vwap), 5)), 5)

    def alpha_051(self) -> Panel:
        slope = (delay(self.d.close, 20) - delay(self.d.close, 10)) / 10 - (delay(self.d.close, 10) - self.d.close) / 10
        return (-(self.d.close - delay(self.d.close, 1))).where(slope >= -0.05, 1.0)

    def alpha_052(self) -> Panel:
        return (-ts_min(self.d.low, 5) + delay(ts_min(self.d.low, 5), 5)) * rank((ts_sum(self.d.returns, 240) - ts_sum(self.d.returns, 20)) / 220) * ts_rank(self.d.volume, 5)

    def alpha_053(self) -> Panel:
        value = _safe_div((self.d.close - self.d.low) - (self.d.high - self.d.close), self.d.close - self.d.low)
        return -delta(value, 9)

    def alpha_054(self) -> Panel:
        numerator = -(self.d.low - self.d.close) * np.power(self.d.open, 5)
        denominator = (self.d.low - self.d.high) * np.power(self.d.close, 5)
        return _safe_div(numerator, denominator)

    def alpha_055(self) -> Panel:
        oscillator = _safe_div(self.d.close - ts_min(self.d.low, 12), ts_max(self.d.high, 12) - ts_min(self.d.low, 12))
        return -correlation(rank(oscillator), rank(self.d.volume), 6)

    def alpha_056(self) -> Panel:
        if self.d.cap is None or self.d.cap.isna().all().all():
            raise Alpha101DataError("alpha_056 requires market-cap data ('cap' or 'market_cap')")
        return -rank(_safe_div(ts_sum(self.d.returns, 10), ts_sum(ts_sum(self.d.returns, 2), 3))) * rank(self.d.returns * self.d.cap)

    def alpha_057(self) -> Panel:
        return -_safe_div(self.d.close - self.d.vwap, decay_linear(rank(ts_argmax(self.d.close, 30)), 2))

    def alpha_058(self) -> Panel:
        neutral = self._neutral(self.d.vwap, "sector")
        return -ts_rank(decay_linear(correlation(neutral, self.d.volume, 3.92795), 7.89291), 5.50322)

    def alpha_059(self) -> Panel:
        # The two terms in the published expression are both VWAP, so its
        # fitted mixing coefficient cancels exactly.  Keep the formula's
        # output while making the actual input explicit.
        neutral = self._neutral(self.d.vwap, "industry")
        return -ts_rank(decay_linear(correlation(neutral, self.d.volume, 4.25197), 16.2289), 8.19648)

    def alpha_060(self) -> Panel:
        position = _safe_div((self.d.close - self.d.low) - (self.d.high - self.d.close), self.d.high - self.d.low) * self.d.volume
        return -(2 * scale(rank(position)) - scale(rank(ts_argmax(self.d.close, 10))))

    def alpha_061(self) -> Panel:
        left = rank(self.d.vwap - ts_min(self.d.vwap, 16.1219))
        right = rank(correlation(self.d.vwap, self.d.adv(180), 17.9282))
        return _signed_condition(left < right, valid=left.notna() & right.notna())

    def alpha_062(self) -> Panel:
        left = rank(correlation(self.d.vwap, ts_sum(self.d.adv(20), 22.4101), 9.91009))
        comparison = (2 * rank(self.d.open)) < (rank((self.d.high + self.d.low) / 2) + rank(self.d.high))
        right = rank(comparison.astype(float))
        return _signed_condition(left < right, true_value=-1.0, false_value=1.0, valid=left.notna() & right.notna())

    def alpha_063(self) -> Panel:
        first = rank(decay_linear(delta(self._neutral(self.d.close, "industry"), 2.25164), 8.22237))
        mixed = self.d.vwap * 0.318108 + self.d.open * (1 - 0.318108)
        second = rank(decay_linear(correlation(mixed, ts_sum(self.d.adv(180), 37.2467), 13.557), 12.2883))
        return -(first - second)

    def alpha_064(self) -> Panel:
        mixed = self.d.open * 0.178404 + self.d.low * (1 - 0.178404)
        left = rank(correlation(ts_sum(mixed, 12.7054), ts_sum(self.d.adv(120), 12.7054), 16.6208))
        right_value = ((self.d.high + self.d.low) / 2) * 0.178404 + self.d.vwap * (1 - 0.178404)
        right = rank(delta(right_value, 3.69741))
        return _signed_condition(left < right, true_value=-1.0, false_value=1.0, valid=left.notna() & right.notna())

    def alpha_065(self) -> Panel:
        mixed = self.d.open * 0.00817205 + self.d.vwap * (1 - 0.00817205)
        left = rank(correlation(mixed, ts_sum(self.d.adv(60), 8.6911), 6.40374))
        right = rank(self.d.open - ts_min(self.d.open, 13.635))
        return _signed_condition(left < right, true_value=-1.0, false_value=1.0, valid=left.notna() & right.notna())

    def alpha_066(self) -> Panel:
        first = rank(decay_linear(delta(self.d.vwap, 3.51013), 7.23052))
        # Both sides of the original weighted expression are low; the
        # coefficient is therefore algebraically irrelevant.
        second_input = _safe_div(self.d.low - self.d.vwap, self.d.open - (self.d.high + self.d.low) / 2)
        second = ts_rank(decay_linear(second_input, 11.4157), 6.72611)
        return -(first + second)

    def alpha_067(self) -> Panel:
        base = rank(self.d.high - ts_min(self.d.high, 2.14593))
        exponent = rank(correlation(self._neutral(self.d.vwap, "sector"), self._neutral(self.d.adv(20), "subindustry"), 6.02936))
        return -np.power(base, exponent)

    def alpha_068(self) -> Panel:
        left = ts_rank(correlation(rank(self.d.high), rank(self.d.adv(15)), 8.91644), 13.9333)
        mixed = self.d.close * 0.518371 + self.d.low * (1 - 0.518371)
        right = rank(delta(mixed, 1.06157))
        return _signed_condition(left < right, true_value=-1.0, false_value=1.0, valid=left.notna() & right.notna())

    def alpha_069(self) -> Panel:
        base = rank(ts_max(delta(self._neutral(self.d.vwap, "industry"), 2.72412), 4.79344))
        mixed = self.d.close * 0.490655 + self.d.vwap * (1 - 0.490655)
        exponent = ts_rank(correlation(mixed, self.d.adv(20), 4.92416), 9.0615)
        return -np.power(base, exponent)

    def alpha_070(self) -> Panel:
        base = rank(delta(self.d.vwap, 1.29456))
        exponent = ts_rank(correlation(self._neutral(self.d.close, "industry"), self.d.adv(50), 17.8256), 17.9171)
        return -np.power(base, exponent)

    def alpha_071(self) -> Panel:
        left = ts_rank(
            decay_linear(
                correlation(ts_rank(self.d.close, 3.43976), ts_rank(self.d.adv(180), 12.0647), 18.0175),
                4.20501,
            ),
            15.6948,
        )
        right = ts_rank(
            decay_linear(np.power(rank((self.d.low + self.d.open) - (self.d.vwap + self.d.vwap)), 2), 16.4662),
            4.4388,
        )
        return element_max(left, right)

    def alpha_072(self) -> Panel:
        numerator = rank(decay_linear(correlation((self.d.high + self.d.low) / 2, self.d.adv(40), 8.93345), 10.1519))
        denominator = rank(
            decay_linear(
                correlation(ts_rank(self.d.vwap, 3.72469), ts_rank(self.d.volume, 18.5188), 6.86671),
                2.95011,
            )
        )
        return _safe_div(numerator, denominator)

    def alpha_073(self) -> Panel:
        first = rank(decay_linear(delta(self.d.vwap, 4.72775), 2.91864))
        mixed = self.d.open * 0.147155 + self.d.low * (1 - 0.147155)
        second_input = -_safe_div(delta(mixed, 2.03608), mixed)
        second = ts_rank(decay_linear(second_input, 3.33829), 16.7411)
        return -element_max(first, second)

    def alpha_074(self) -> Panel:
        left = rank(correlation(self.d.close, ts_sum(self.d.adv(30), 37.4843), 15.1365))
        mixed = self.d.high * 0.0261661 + self.d.vwap * (1 - 0.0261661)
        right = rank(correlation(rank(mixed), rank(self.d.volume), 11.4791))
        return _signed_condition(left < right, true_value=-1.0, false_value=1.0, valid=left.notna() & right.notna())

    def alpha_075(self) -> Panel:
        left = rank(correlation(self.d.vwap, self.d.volume, 4.24304))
        right = rank(correlation(rank(self.d.low), rank(self.d.adv(50)), 12.4413))
        return _signed_condition(left < right, valid=left.notna() & right.notna())

    def alpha_076(self) -> Panel:
        first = rank(decay_linear(delta(self.d.vwap, 1.24383), 11.8259))
        neutral = self._neutral(self.d.low, "sector")
        second = ts_rank(
            decay_linear(ts_rank(correlation(neutral, self.d.adv(81), 8.14941), 19.569), 17.1543),
            19.383,
        )
        return -element_max(first, second)

    def alpha_077(self) -> Panel:
        first = rank(decay_linear((self.d.high + self.d.low) / 2 - self.d.vwap, 20.0451))
        second = rank(decay_linear(correlation((self.d.high + self.d.low) / 2, self.d.adv(40), 3.1614), 5.64125))
        return element_min(first, second)

    def alpha_078(self) -> Panel:
        mixed = self.d.low * 0.352233 + self.d.vwap * (1 - 0.352233)
        base = rank(correlation(ts_sum(mixed, 19.7428), ts_sum(self.d.adv(40), 19.7428), 6.83313))
        exponent = rank(correlation(rank(self.d.vwap), rank(self.d.volume), 5.77492))
        return np.power(base, exponent)

    def alpha_079(self) -> Panel:
        mixed = self.d.close * 0.60733 + self.d.open * (1 - 0.60733)
        left = rank(delta(self._neutral(mixed, "sector"), 1.23438))
        right = rank(correlation(ts_rank(self.d.vwap, 3.60973), ts_rank(self.d.adv(150), 9.18637), 14.6644))
        return _signed_condition(left < right, valid=left.notna() & right.notna())

    def alpha_080(self) -> Panel:
        mixed = self.d.open * 0.868128 + self.d.high * (1 - 0.868128)
        base = rank(np.sign(delta(self._neutral(mixed, "industry"), 4.04545)))
        exponent = ts_rank(correlation(self.d.high, self.d.adv(10), 5.11456), 5.53756)
        return -np.power(base, exponent)

    def alpha_081(self) -> Panel:
        corr = correlation(self.d.vwap, ts_sum(self.d.adv(10), 49.6054), 8.47743)
        left = rank(np.log(product(rank(np.power(rank(corr), 4)), 14.9655)))
        right = rank(correlation(rank(self.d.vwap), rank(self.d.volume), 5.07914))
        return _signed_condition(left < right, true_value=-1.0, false_value=1.0, valid=left.notna() & right.notna())

    def alpha_082(self) -> Panel:
        first = rank(decay_linear(delta(self.d.open, 1.46063), 14.8717))
        neutral_volume = self._neutral(self.d.volume, "sector")
        second = ts_rank(
            decay_linear(correlation(neutral_volume, self.d.open, 17.4842), 6.92131),
            13.4283,
        )
        return -element_min(first, second)

    def alpha_083(self) -> Panel:
        normalized_range = _safe_div(self.d.high - self.d.low, ts_sum(self.d.close, 5) / 5)
        numerator = rank(delay(normalized_range, 2)) * rank(rank(self.d.volume))
        denominator = _safe_div(normalized_range, self.d.vwap - self.d.close)
        return _safe_div(numerator, denominator)

    def alpha_084(self) -> Panel:
        base = ts_rank(self.d.vwap - ts_max(self.d.vwap, 15.3217), 20.7127)
        return signed_power(base, delta(self.d.close, 4.96796))

    def alpha_085(self) -> Panel:
        mixed = self.d.high * 0.876703 + self.d.close * (1 - 0.876703)
        base = rank(correlation(mixed, self.d.adv(30), 9.61331))
        exponent = rank(correlation(ts_rank((self.d.high + self.d.low) / 2, 3.70596), ts_rank(self.d.volume, 10.1595), 7.11408))
        return np.power(base, exponent)

    def alpha_086(self) -> Panel:
        left = ts_rank(correlation(self.d.close, ts_sum(self.d.adv(20), 14.7444), 6.00049), 20.4195)
        # Open cancels from the published expression.
        right = rank(self.d.close - self.d.vwap)
        return _signed_condition(left < right, true_value=-1.0, false_value=1.0, valid=left.notna() & right.notna())

    def alpha_087(self) -> Panel:
        mixed = self.d.close * 0.369701 + self.d.vwap * (1 - 0.369701)
        first = rank(decay_linear(delta(mixed, 1.91233), 2.65461))
        neutral_adv = self._neutral(self.d.adv(81), "industry")
        second = ts_rank(decay_linear(correlation(neutral_adv, self.d.close, 13.4132).abs(), 4.89768), 14.4535)
        return -element_max(first, second)

    def alpha_088(self) -> Panel:
        first = rank(decay_linear((rank(self.d.open) + rank(self.d.low)) - (rank(self.d.high) + rank(self.d.close)), 8.06882))
        second = ts_rank(
            decay_linear(correlation(ts_rank(self.d.close, 8.44728), ts_rank(self.d.adv(60), 20.6966), 8.01266), 6.65053),
            2.61957,
        )
        return element_min(first, second)

    def alpha_089(self) -> Panel:
        first = ts_rank(decay_linear(correlation(self.d.low, self.d.adv(10), 6.94279), 5.51607), 3.79744)
        second = ts_rank(decay_linear(delta(self._neutral(self.d.vwap, "industry"), 3.48158), 10.1466), 15.3012)
        return first - second

    def alpha_090(self) -> Panel:
        base = rank(self.d.close - ts_max(self.d.close, 4.66719))
        exponent = ts_rank(correlation(self._neutral(self.d.adv(40), "subindustry"), self.d.low, 5.38375), 3.21856)
        return -np.power(base, exponent)

    def alpha_091(self) -> Panel:
        neutral = self._neutral(self.d.close, "industry")
        first = ts_rank(decay_linear(decay_linear(correlation(neutral, self.d.volume, 9.74928), 16.398), 3.83219), 4.8667)
        second = rank(decay_linear(correlation(self.d.vwap, self.d.adv(30), 4.01303), 2.6809))
        return -(first - second)

    def alpha_092(self) -> Panel:
        condition = ((self.d.high + self.d.low) / 2 + self.d.close) < (self.d.low + self.d.open)
        first = ts_rank(decay_linear(condition.astype(float), 14.7221), 18.8683)
        second = ts_rank(decay_linear(correlation(rank(self.d.low), rank(self.d.adv(30)), 7.58555), 6.94024), 6.80584)
        return element_min(first, second)

    def alpha_093(self) -> Panel:
        numerator = ts_rank(decay_linear(correlation(self._neutral(self.d.vwap, "industry"), self.d.adv(81), 17.4193), 19.848), 7.54455)
        mixed = self.d.close * 0.524434 + self.d.vwap * (1 - 0.524434)
        denominator = rank(decay_linear(delta(mixed, 2.77377), 16.2664))
        return _safe_div(numerator, denominator)

    def alpha_094(self) -> Panel:
        base = rank(self.d.vwap - ts_min(self.d.vwap, 11.5783))
        exponent = ts_rank(correlation(ts_rank(self.d.vwap, 19.6462), ts_rank(self.d.adv(60), 4.02992), 18.0926), 2.70756)
        return -np.power(base, exponent)

    def alpha_095(self) -> Panel:
        left = rank(self.d.open - ts_min(self.d.open, 12.4105))
        corr = correlation(ts_sum((self.d.high + self.d.low) / 2, 19.1351), ts_sum(self.d.adv(40), 19.1351), 12.8742)
        right = ts_rank(np.power(rank(corr), 5), 11.7584)
        return _signed_condition(left < right, valid=left.notna() & right.notna())

    def alpha_096(self) -> Panel:
        first = ts_rank(decay_linear(correlation(rank(self.d.vwap), rank(self.d.volume), 3.83878), 4.16783), 8.38151)
        corr = correlation(ts_rank(self.d.close, 7.45404), ts_rank(self.d.adv(60), 4.13242), 3.65459)
        second = ts_rank(decay_linear(ts_argmax(corr, 12.6556), 14.0365), 13.4143)
        return -element_max(first, second)

    def alpha_097(self) -> Panel:
        mixed = self.d.low * 0.721001 + self.d.vwap * (1 - 0.721001)
        first = rank(decay_linear(delta(self._neutral(mixed, "industry"), 3.3705), 20.4523))
        corr = correlation(ts_rank(self.d.low, 7.87871), ts_rank(self.d.adv(60), 17.255), 4.97547)
        second = ts_rank(decay_linear(ts_rank(corr, 18.5925), 15.7152), 6.71659)
        return -(first - second)

    def alpha_098(self) -> Panel:
        first = rank(decay_linear(correlation(self.d.vwap, ts_sum(self.d.adv(5), 26.4719), 4.58418), 7.18088))
        corr = correlation(rank(self.d.open), rank(self.d.adv(15)), 20.8187)
        second = rank(decay_linear(ts_rank(ts_argmin(corr, 8.62571), 6.95668), 8.07206))
        return first - second

    def alpha_099(self) -> Panel:
        left = rank(correlation(ts_sum((self.d.high + self.d.low) / 2, 19.8975), ts_sum(self.d.adv(60), 19.8975), 8.8136))
        right = rank(correlation(self.d.low, self.d.volume, 6.28259))
        return _signed_condition(left < right, true_value=-1.0, false_value=1.0, valid=left.notna() & right.notna())

    def alpha_100(self) -> Panel:
        position = _safe_div((self.d.close - self.d.low) - (self.d.high - self.d.close), self.d.high - self.d.low) * self.d.volume
        first = 1.5 * scale(self._neutral(self._neutral(rank(position), "subindustry"), "subindustry"))
        second_input = correlation(self.d.close, rank(self.d.adv(20)), 5) - rank(ts_argmin(self.d.close, 30))
        second = scale(self._neutral(second_input, "subindustry"))
        return -(first - second) * _safe_div(self.d.volume, self.d.adv(20))

    def alpha_101(self) -> Panel:
        return _safe_div(self.d.close - self.d.open, (self.d.high - self.d.low) + 0.001)


ALPHA101_NAMES = tuple(f"alpha_{number:03d}" for number in range(1, 102))


def normalize_alpha_name(name: str | int) -> str:
    if isinstance(name, int):
        number = name
    else:
        raw = str(name).strip().lower().replace("-", "_")
        if raw.startswith("alpha_"):
            raw = raw.removeprefix("alpha_")
        elif raw.startswith("alpha"):
            raw = raw.removeprefix("alpha")
        try:
            number = int(raw)
        except ValueError as exc:
            raise KeyError(f"invalid Alpha101 factor name: {name}") from exc
    if not 1 <= number <= 101:
        raise KeyError(f"Alpha101 factor number must be in [1, 101], got {number}")
    return f"alpha_{number:03d}"


def _wide_field(
    raw_data: Mapping[str, pd.DataFrame],
    aliases: Sequence[str],
    *,
    dates: pd.DatetimeIndex,
    symbols: Sequence[str],
) -> Panel | None:
    series: dict[str, pd.Series] = {}
    for raw_symbol, raw_frame in raw_data.items():
        if raw_frame is None or raw_frame.empty:
            continue
        frame = raw_frame.copy()
        frame.columns = [str(column).lower() for column in frame.columns]
        column = next((alias for alias in aliases if alias in frame.columns), None)
        if column is None or "date" not in frame.columns:
            continue
        index = pd.to_datetime(frame["date"])
        values = pd.to_numeric(frame[column], errors="coerce")
        value_series = pd.Series(values.to_numpy(), index=index)
        series[str(raw_symbol).zfill(6)] = value_series.groupby(level=0).last()
    if not series:
        return None
    return pd.DataFrame(series).reindex(index=dates, columns=symbols).sort_index()


def _classification_panel(
    metadata: pd.DataFrame | Mapping[str, Mapping[str, object]] | None,
    column: str,
    *,
    dates: pd.DatetimeIndex,
    symbols: Sequence[str],
) -> Panel | None:
    if metadata is None:
        return None
    if isinstance(metadata, Mapping):
        values = {
            str(symbol).zfill(6): fields.get(column)
            for symbol, fields in metadata.items()
            if isinstance(fields, Mapping)
        }
    else:
        frame = metadata.copy()
        frame.columns = [str(value).lower() for value in frame.columns]
        symbol_col = next((value for value in ("symbol", "code", "ts_code") if value in frame.columns), None)
        if symbol_col is None or column not in frame.columns:
            return None
        symbol_values = frame[symbol_col].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(6)
        values = dict(zip(symbol_values, frame[column]))
    row = pd.Series(values, index=symbols, dtype="object")
    return pd.DataFrame([row.to_numpy()] * len(dates), index=dates, columns=symbols)


def build_alpha101_panels(
    raw_data: Mapping[str, pd.DataFrame],
    *,
    metadata: pd.DataFrame | Mapping[str, Mapping[str, object]] | None = None,
) -> Alpha101Panels:
    """Build aligned panels from the repository's per-symbol raw CSV frames.

    VWAP uses an explicit ``vwap``/``avg`` column when available; otherwise it
    falls back to typical price ``(high + low + close) / 3``.  ``advN`` is the
    N-day mean of the input volume field, matching the formulas' volume units.
    """
    if not raw_data:
        raise Alpha101DataError("raw_data is empty")
    symbols = sorted(str(symbol).zfill(6) for symbol in raw_data)
    date_values: list[pd.Timestamp] = []
    for frame in raw_data.values():
        if frame is not None and not frame.empty and "date" in [str(column).lower() for column in frame.columns]:
            normalized = frame.copy()
            normalized.columns = [str(column).lower() for column in normalized.columns]
            date_values.extend(pd.to_datetime(normalized["date"]).tolist())
    if not date_values:
        raise Alpha101DataError("raw_data contains no date values")
    dates = pd.DatetimeIndex(sorted(set(date_values)))

    required: dict[str, Panel] = {}
    for field in ("open", "close", "high", "low", "volume"):
        panel = _wide_field(raw_data, (field,), dates=dates, symbols=symbols)
        if panel is None:
            raise Alpha101DataError(f"raw_data is missing required '{field}' values")
        required[field] = panel

    explicit_vwap = _wide_field(raw_data, ("vwap", "avg"), dates=dates, symbols=symbols)
    typical_price = (required["high"] + required["low"] + required["close"]) / 3.0
    vwap = typical_price if explicit_vwap is None else explicit_vwap.combine_first(typical_price)
    cap = _wide_field(raw_data, ("cap", "market_cap", "total_mv"), dates=dates, symbols=symbols)
    is_st = _wide_field(raw_data, ("is_st",), dates=dates, symbols=symbols)
    turnover_value = _wide_field(raw_data, ("turnover_value",), dates=dates, symbols=symbols)
    if turnover_value is None:
        amount = _wide_field(raw_data, ("amount",), dates=dates, symbols=symbols)
        turnover_value = (
            amount * 1000.0
            if amount is not None
            else required["close"] * required["volume"]
        )

    sector = _classification_panel(metadata, "sector", dates=dates, symbols=symbols)
    industry = _classification_panel(metadata, "industry", dates=dates, symbols=symbols)
    subindustry = _classification_panel(metadata, "subindustry", dates=dates, symbols=symbols)
    # A single industry column is still useful for every neutralization level in
    # A-share datasets that do not carry a three-level classification hierarchy.
    sector = sector if sector is not None else industry
    subindustry = subindustry if subindustry is not None else industry

    return Alpha101Panels(
        open=required["open"],
        close=required["close"],
        high=required["high"],
        low=required["low"],
        volume=required["volume"],
        vwap=vwap,
        returns=required["close"].pct_change(fill_method=None),
        cap=cap,
        sector=sector,
        industry=industry,
        subindustry=subindustry,
        is_st=is_st,
        turnover_value=turnover_value,
    )


def alpha101_to_long(
    raw_data: Mapping[str, pd.DataFrame],
    factor_name: str | int,
    *,
    metadata: pd.DataFrame | Mapping[str, Mapping[str, object]] | None = None,
) -> pd.DataFrame:
    """Calculate one Alpha101 factor and return FactorTester's long schema."""
    panels = build_alpha101_panels(raw_data, metadata=metadata)
    name = normalize_alpha_name(factor_name)
    values = Alpha101(panels).calculate(name)
    factor_long = values.rename_axis(index="date", columns="symbol").stack(future_stack=True).rename("factor_value")
    close_long = panels.close.rename_axis(index="date", columns="symbol").stack(future_stack=True).rename("close")
    volume_long = panels.volume.rename_axis(index="date", columns="symbol").stack(
        future_stack=True
    ).rename("volume")
    daily_return_long = panels.returns.rename_axis(index="date", columns="symbol").stack(
        future_stack=True
    ).rename("daily_return")
    listing_age_long = (
        panels.close.notna()
        .cumsum()
        .rename_axis(index="date", columns="symbol")
        .stack(future_stack=True)
        .rename("listing_age_days")
    )
    parts = [factor_long, close_long, volume_long, daily_return_long, listing_age_long]
    if panels.industry is not None:
        parts.append(
            panels.industry.rename_axis(index="date", columns="symbol")
            .stack(future_stack=True)
            .rename("industry")
        )
    if panels.cap is not None:
        parts.append(
            panels.cap.rename_axis(index="date", columns="symbol")
            .stack(future_stack=True)
            .rename("market_cap")
        )
    if panels.is_st is not None:
        parts.append(
            panels.is_st.rename_axis(index="date", columns="symbol")
            .stack(future_stack=True)
            .rename("is_st")
        )
    if panels.turnover_value is not None:
        parts.append(
            panels.turnover_value.rename_axis(index="date", columns="symbol")
            .stack(future_stack=True)
            .rename("turnover_value")
        )
    result = pd.concat(parts, axis=1).reset_index()
    return result
