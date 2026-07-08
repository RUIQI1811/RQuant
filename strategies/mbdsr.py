"""mBDSR-style trend-pullback buy strategy.

This module implements a transparent rule set inspired by the general idea of
buying an oversold pullback inside an intact uptrend.  It does not reproduce or
depend on any proprietary indicator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from strategies.selector import PipelineSelector
except ImportError:  # pragma: no cover - direct script fallback
    from Selector import PipelineSelector


REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def calc_rci(series: pd.Series, window: int) -> pd.Series:
    """Calculate Rank Correlation Index without TA-Lib.

    Time is ranked from oldest (1) to newest (``window``), while prices use
    ascending average ranks.  Average ranks make tied prices deterministic.
    A fully rising window therefore has RCI +100 and a falling window -100.
    """
    if window < 2:
        raise ValueError("window must be at least 2")

    values = pd.to_numeric(series, errors="coerce")
    output = np.full(len(values), np.nan, dtype=float)
    if len(values) < window:
        return pd.Series(output, index=series.index, name=series.name)

    time_rank = np.arange(1.0, window + 1.0)
    denominator = float(window * (window**2 - 1))
    price_windows = np.lib.stride_tricks.sliding_window_view(
        values.to_numpy(dtype=float), window_shape=window
    )
    valid = np.isfinite(price_windows).all(axis=1)

    # Average ranks, including ties, expressed only with NumPy comparisons:
    # rank(x) = 1 + count(y < x) + (count(y == x) - 1) / 2.
    less_counts = (price_windows[:, :, None] > price_windows[:, None, :]).sum(axis=2)
    equal_counts = (price_windows[:, :, None] == price_windows[:, None, :]).sum(axis=2)
    price_ranks = 1.0 + less_counts + (equal_counts - 1.0) / 2.0
    rank_diff = time_rank - price_ranks
    rolling_rci = (1.0 - 6.0 * np.square(rank_diff).sum(axis=1) / denominator) * 100.0
    rolling_rci[~valid] = np.nan
    output[window - 1 :] = rolling_rci
    return pd.Series(output, index=series.index, name=series.name)


def _calc_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Return a portable simple-moving-average ATR of true range."""
    previous_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window=window, min_periods=window).mean()


def calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Return on-balance volume, starting from zero on the first bar."""
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def add_mbdsr_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate mBDSR indicators, component conditions, and buy signals.

    The returned frame is a copy.  Signal columns are always strict booleans;
    bars without enough lookback history evaluate to ``False``.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f"mBDSR requires columns: {', '.join(missing)}")

    result = df.copy()
    for column in REQUIRED_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result["MA20"] = result["close"].rolling(20, min_periods=20).mean()
    result["MA60"] = result["close"].rolling(60, min_periods=60).mean()
    result["ATR14"] = _calc_atr(result, 14)
    result["OBV"] = calc_obv(result["close"], result["volume"])
    result["VOL20"] = result["volume"].rolling(20, min_periods=20).mean()
    result["RCI9"] = calc_rci(result["close"], 9)
    result["RCI26"] = calc_rci(result["close"], 26)
    result["RCI52"] = calc_rci(result["close"], 52)

    result["trend_filter"] = (
        (result["MA60"] > result["MA60"].shift(5))
        & (result["close"] > result["MA60"] * 0.98)
        & (result["RCI26"] > 0)
        & (result["RCI52"] > 0)
    )
    result["rci_pullback"] = (
        (result["RCI9"].shift(1) < -80)
        & (result["RCI9"] > result["RCI9"].shift(1))
    )
    result["support_touch"] = (
        ((result["close"] - result["MA20"]).abs() / result["MA20"] < 0.03)
        | ((result["close"] - result["MA60"]).abs() / result["MA60"] < 0.04)
    )
    result["volume_filter"] = result["volume"] < result["VOL20"] * 1.2
    result["obv_filter"] = result["OBV"] > result["OBV"].rolling(20).min()

    recent_high = result["high"].rolling(10).max()
    result["atr_filter"] = recent_high - result["close"] < 2.0 * result["ATR14"]
    result["candle_confirm"] = (
        (result["close"] > result["open"])
        & (result["close"] > result["close"].shift(1))
    )

    condition_columns = [
        "trend_filter",
        "rci_pullback",
        "support_touch",
        "volume_filter",
        "obv_filter",
        "atr_filter",
        "candle_confirm",
    ]
    result["mBDSR_buy_signal"] = result[condition_columns].all(axis=1)
    result["mBDSR_buy_next_confirm"] = (
        result["mBDSR_buy_signal"].shift(1, fill_value=False)
        & (result["close"] > result["high"].shift(1))
    )

    bool_columns = condition_columns + ["mBDSR_buy_signal", "mBDSR_buy_next_confirm"]
    result[bool_columns] = result[bool_columns].fillna(False).astype(bool)
    return result


class MBDSRSelector(PipelineSelector):
    """Selector adapter for standard or next-day-confirmed mBDSR signals."""

    def __init__(
        self,
        *,
        use_next_confirm: bool = False,
        date_col: str = "date",
        extra_bars_buffer: int = 10,
    ) -> None:
        super().__init__(
            filters=(),
            date_col=date_col,
            min_bars=65,
            extra_bars_buffer=extra_bars_buffer,
        )
        self.use_next_confirm = bool(use_next_confirm)

    @property
    def signal_column(self) -> str:
        return "mBDSR_buy_next_confirm" if self.use_next_confirm else "mBDSR_buy_signal"

    def prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        result = add_mbdsr_features(df)
        result["_vec_pick"] = result[self.signal_column]
        return result

    def passes_hist(self, hist: pd.DataFrame) -> bool:
        if hist is None or len(hist) < self.min_bars + self.extra_bars_buffer:
            return False
        prepared = add_mbdsr_features(hist)
        return bool(prepared[self.signal_column].iloc[-1])
