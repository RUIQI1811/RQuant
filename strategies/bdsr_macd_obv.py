"""BDSR, above-zero MACD, and OBV-trend confluence buy strategy.

The repository has no canonical BDSR formula.  This strategy defines the BDSR
pair explicitly as a fast and slow Rank Correlation Index (RCI).  A buy signal
is emitted only when all three causal conditions are true on the same bar:

* fast RCI crosses above slow RCI;
* MACD DIF crosses above DEA while both are above zero;
* OBV is above a rising OBV moving average.
"""
from __future__ import annotations

import pandas as pd

try:
    from strategies.selector import PipelineSelector
    from strategies.mbdsr import calc_obv, calc_rci
except ImportError:  # pragma: no cover - direct script fallback
    from Selector import PipelineSelector
    from mbdsr import calc_obv, calc_rci


REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def _validate_periods(
    *,
    bdsr_fast_window: int,
    bdsr_slow_window: int,
    macd_fast_period: int,
    macd_slow_period: int,
    macd_signal_period: int,
    obv_ma_window: int,
    obv_trend_lookback: int,
) -> None:
    values = {
        "bdsr_fast_window": bdsr_fast_window,
        "bdsr_slow_window": bdsr_slow_window,
        "macd_fast_period": macd_fast_period,
        "macd_slow_period": macd_slow_period,
        "macd_signal_period": macd_signal_period,
        "obv_ma_window": obv_ma_window,
        "obv_trend_lookback": obv_trend_lookback,
    }
    invalid = [name for name, value in values.items() if int(value) <= 0]
    if invalid:
        raise ValueError("strategy periods must be positive: " + ", ".join(invalid))
    if int(bdsr_fast_window) < 2:
        raise ValueError("bdsr_fast_window must be at least 2")
    if int(bdsr_slow_window) <= int(bdsr_fast_window):
        raise ValueError("bdsr_slow_window must be greater than bdsr_fast_window")
    if int(macd_slow_period) <= int(macd_fast_period):
        raise ValueError("macd_slow_period must be greater than macd_fast_period")


def add_bdsr_macd_obv_features(
    df: pd.DataFrame,
    *,
    bdsr_fast_window: int = 9,
    bdsr_slow_window: int = 26,
    macd_fast_period: int = 12,
    macd_slow_period: int = 26,
    macd_signal_period: int = 9,
    obv_ma_window: int = 20,
    obv_trend_lookback: int = 3,
) -> pd.DataFrame:
    """Return indicators, component conditions, and the exact buy signal."""
    _validate_periods(
        bdsr_fast_window=bdsr_fast_window,
        bdsr_slow_window=bdsr_slow_window,
        macd_fast_period=macd_fast_period,
        macd_slow_period=macd_slow_period,
        macd_signal_period=macd_signal_period,
        obv_ma_window=obv_ma_window,
        obv_trend_lookback=obv_trend_lookback,
    )
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(
            "bdsr_macd_obv requires columns: " + ", ".join(missing)
        )

    result = df.copy()
    for column in REQUIRED_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    fast_window = int(bdsr_fast_window)
    slow_window = int(bdsr_slow_window)
    result["BDSR_FAST"] = calc_rci(result["close"], fast_window)
    result["BDSR_SLOW"] = calc_rci(result["close"], slow_window)
    result["bdsr_golden_cross"] = (
        (result["BDSR_FAST"].shift(1) <= result["BDSR_SLOW"].shift(1))
        & (result["BDSR_FAST"] > result["BDSR_SLOW"])
    )

    fast_ema = result["close"].ewm(
        span=int(macd_fast_period),
        adjust=False,
        min_periods=int(macd_fast_period),
    ).mean()
    slow_ema = result["close"].ewm(
        span=int(macd_slow_period),
        adjust=False,
        min_periods=int(macd_slow_period),
    ).mean()
    result["MACD_DIF"] = fast_ema - slow_ema
    result["MACD_DEA"] = result["MACD_DIF"].ewm(
        span=int(macd_signal_period),
        adjust=False,
        min_periods=int(macd_signal_period),
    ).mean()
    result["MACD_HIST"] = 2.0 * (result["MACD_DIF"] - result["MACD_DEA"])
    result["macd_above_zero_golden_cross"] = (
        (result["MACD_DIF"].shift(1) <= result["MACD_DEA"].shift(1))
        & (result["MACD_DIF"] > result["MACD_DEA"])
        & (result["MACD_DIF"] > 0.0)
        & (result["MACD_DEA"] > 0.0)
    )

    result["OBV"] = calc_obv(result["close"], result["volume"])
    result["OBV_MA"] = result["OBV"].rolling(
        int(obv_ma_window),
        min_periods=int(obv_ma_window),
    ).mean()
    result["obv_uptrend"] = (
        (result["OBV"] > result["OBV_MA"])
        & (result["OBV_MA"] > result["OBV_MA"].shift(int(obv_trend_lookback)))
    )

    conditions = [
        "bdsr_golden_cross",
        "macd_above_zero_golden_cross",
        "obv_uptrend",
    ]
    result["bdsr_macd_obv_buy_signal"] = result[conditions].all(axis=1)
    result[conditions + ["bdsr_macd_obv_buy_signal"]] = (
        result[conditions + ["bdsr_macd_obv_buy_signal"]]
        .fillna(False)
        .astype(bool)
    )
    return result


class BDSRMACDOBVSelector(PipelineSelector):
    """Selector for same-bar BDSR, MACD, and OBV confluence."""

    def __init__(
        self,
        *,
        bdsr_fast_window: int = 9,
        bdsr_slow_window: int = 26,
        macd_fast_period: int = 12,
        macd_slow_period: int = 26,
        macd_signal_period: int = 9,
        obv_ma_window: int = 20,
        obv_trend_lookback: int = 3,
        date_col: str = "date",
        extra_bars_buffer: int = 10,
    ) -> None:
        _validate_periods(
            bdsr_fast_window=bdsr_fast_window,
            bdsr_slow_window=bdsr_slow_window,
            macd_fast_period=macd_fast_period,
            macd_slow_period=macd_slow_period,
            macd_signal_period=macd_signal_period,
            obv_ma_window=obv_ma_window,
            obv_trend_lookback=obv_trend_lookback,
        )
        min_bars = max(
            int(bdsr_slow_window) + 1,
            int(macd_slow_period) + int(macd_signal_period),
            int(obv_ma_window) + int(obv_trend_lookback),
        )
        super().__init__(
            filters=(),
            date_col=date_col,
            min_bars=min_bars,
            extra_bars_buffer=extra_bars_buffer,
        )
        self.feature_kwargs = {
            "bdsr_fast_window": int(bdsr_fast_window),
            "bdsr_slow_window": int(bdsr_slow_window),
            "macd_fast_period": int(macd_fast_period),
            "macd_slow_period": int(macd_slow_period),
            "macd_signal_period": int(macd_signal_period),
            "obv_ma_window": int(obv_ma_window),
            "obv_trend_lookback": int(obv_trend_lookback),
        }

    @property
    def signal_column(self) -> str:
        return "bdsr_macd_obv_buy_signal"

    def prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        result = add_bdsr_macd_obv_features(df, **self.feature_kwargs)
        result["_vec_pick"] = result[self.signal_column]
        return result

    def passes_hist(self, hist: pd.DataFrame) -> bool:
        if hist is None or len(hist) < self.min_bars + self.extra_bars_buffer:
            return False
        prepared = self.prepare_df(hist)
        return bool(prepared[self.signal_column].iloc[-1])


__all__ = [
    "BDSRMACDOBVSelector",
    "add_bdsr_macd_obv_features",
]
