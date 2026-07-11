from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from domain.signals import SignalBook
from signals.schema import Signal, frame_to_signals, signals_to_frame


@dataclass(frozen=True)
class FactorSignalConfig:
    """Settings for converting factor values into buy signals."""

    date_col: str = "date"
    symbol_col: str = "symbol"
    factor_col: str = "factor_value"
    source: str = "factor"
    top_n: Optional[int] = None
    top_quantile: Optional[float] = 0.1
    ascending: bool = False
    default_weight: Optional[float] = None


def factor_frame_to_signal_frame(
    factor_data: pd.DataFrame,
    *,
    config: FactorSignalConfig,
) -> pd.DataFrame:
    """Convert long-format factor values into unified buy signals.

    Higher factor values are selected by default. Set ascending=True when lower
    values are better. Use either top_n or top_quantile; top_n takes precedence.
    """
    if factor_data is None or factor_data.empty:
        return signals_to_frame([])
    required = [config.date_col, config.symbol_col, config.factor_col]
    missing = [col for col in required if col not in factor_data.columns]
    if missing:
        raise ValueError(f"missing factor signal columns: {', '.join(missing)}")
    if config.top_n is None and config.top_quantile is None:
        raise ValueError("either top_n or top_quantile must be provided")
    if config.top_quantile is not None and not (0 < config.top_quantile <= 1):
        raise ValueError("top_quantile must be in (0, 1]")

    df = factor_data.copy()
    df[config.date_col] = pd.to_datetime(df[config.date_col])
    df[config.symbol_col] = df[config.symbol_col].astype(str).str.zfill(6)
    df[config.factor_col] = pd.to_numeric(df[config.factor_col], errors="coerce")
    df = df.dropna(subset=[config.factor_col])

    signals: list[Signal] = []
    for date, daily in df.groupby(config.date_col):
        ranked = daily.sort_values(config.factor_col, ascending=config.ascending)
        if config.top_n is not None:
            selected = ranked.head(config.top_n)
        else:
            count = max(1, int(len(ranked) * float(config.top_quantile)))
            selected = ranked.head(count)
        for _, row in selected.iterrows():
            score = float(row[config.factor_col])
            signals.append(
                Signal(
                    date=pd.to_datetime(date).strftime("%Y-%m-%d"),
                    symbol=str(row[config.symbol_col]).zfill(6),
                    signal_type="buy",
                    source=config.source,
                    score=score,
                    weight=config.default_weight,
                    metadata={"factor_value": score},
                )
            )
    return signals_to_frame(signals)


class SimpleFactorSignalEngine:
    """Minimal factor signal engine that selects top factor names each day."""

    def __init__(self, config: FactorSignalConfig) -> None:
        self.config = config
        self.source = config.source

    def generate_signals(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        return factor_frame_to_signal_frame(factor_data, config=self.config)

    def generate_signal_book(self, factor_data: pd.DataFrame) -> SignalBook:
        return SignalBook(frame_to_signals(self.generate_signals(factor_data)))
