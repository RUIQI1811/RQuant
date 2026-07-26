from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import polars as pl

from domain.signals import SignalBook
from domain.tabular import to_polars
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
    factor_data: Any,
    *,
    config: FactorSignalConfig,
) -> pl.DataFrame:
    """Convert long-format factor values into unified buy signals.

    Higher factor values are selected by default. Set ascending=True when lower
    values are better. Use either top_n or top_quantile; top_n takes precedence.
    """
    df = to_polars(factor_data)
    if df.is_empty():
        return signals_to_frame([])
    required = [config.date_col, config.symbol_col, config.factor_col]
    missing = [col for col in required if col not in factor_data.columns]
    if missing:
        raise ValueError(f"missing factor signal columns: {', '.join(missing)}")
    if config.top_n is None and config.top_quantile is None:
        raise ValueError("either top_n or top_quantile must be provided")
    if config.top_quantile is not None and not (0 < config.top_quantile <= 1):
        raise ValueError("top_quantile must be in (0, 1]")

    df = (
        df.with_columns(
            pl.col(config.date_col)
            .cast(pl.String)
            .str.slice(0, 10)
            .str.to_date("%Y-%m-%d", strict=False),
            pl.col(config.symbol_col).cast(pl.String).str.pad_start(6, "0"),
            pl.col(config.factor_col).cast(pl.Float64, strict=False),
        )
        .drop_nulls([config.date_col, config.factor_col])
        .sort(
            [config.date_col, config.factor_col, config.symbol_col],
            descending=[False, not config.ascending, False],
            maintain_order=True,
        )
    )

    signals: list[Signal] = []
    for daily in df.partition_by(config.date_col, maintain_order=True):
        date = daily.item(0, config.date_col)
        if config.top_n is not None:
            selected = daily.head(config.top_n)
        else:
            count = max(1, int(len(daily) * float(config.top_quantile)))
            selected = daily.head(count)
        for row in selected.iter_rows(named=True):
            score = float(row[config.factor_col])
            signals.append(
                Signal(
                    date=date.isoformat(),
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

    def generate_signals(self, factor_data: Any) -> pl.DataFrame:
        return factor_frame_to_signal_frame(factor_data, config=self.config)

    def generate_signal_book(self, factor_data: Any) -> SignalBook:
        return SignalBook(frame_to_signals(self.generate_signals(factor_data)))
