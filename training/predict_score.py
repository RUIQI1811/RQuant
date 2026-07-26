"""Convert model prediction scores into unified signals."""

from __future__ import annotations

import math
from typing import Any

import polars as pl

from domain.tabular import to_polars
from signals.schema import Signal, signals_to_frame


def scores_to_signals(
    scores: Any,
    source: str,
    date_col: str = "date",
    symbol_col: str = "symbol",
    score_col: str = "score",
    top_n: int | None = None,
    top_quantile: float | None = None,
) -> pl.DataFrame:
    frame = to_polars(scores)
    required = {date_col, symbol_col, score_col}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if top_n is not None and top_n <= 0:
        raise ValueError("top_n must be positive")
    if top_quantile is not None and not 0 < top_quantile <= 1:
        raise ValueError("top_quantile must be in (0, 1]")
    frame = frame.select([date_col, symbol_col, score_col]).with_columns(
        pl.col(date_col)
        .cast(pl.String)
        .str.slice(0, 10)
        .str.to_date("%Y-%m-%d", strict=False),
        pl.col(symbol_col).cast(pl.String).str.pad_start(6, "0"),
        pl.col(score_col).cast(pl.Float64, strict=False),
    )
    if frame.select([date_col, symbol_col]).is_duplicated().any():
        raise ValueError("duplicate date/symbol score rows are not allowed")
    invalid_score = pl.col(score_col).is_null() | pl.col(score_col).is_nan() | pl.col(score_col).is_infinite()
    if frame.select(invalid_score.any()).item():
        raise ValueError("scores must contain only finite values")

    frame = frame.sort(
        [date_col, score_col, symbol_col],
        descending=[False, True, False],
        maintain_order=True,
    )

    signals: list[Signal] = []
    for ranked in frame.partition_by(date_col, maintain_order=True):
        date = ranked.item(0, date_col)
        if top_n is not None:
            selected = ranked.head(top_n)
        elif top_quantile is not None:
            selected = ranked.head(max(1, math.ceil(len(ranked) * top_quantile)))
        else:
            selected = ranked
        weight = 1.0 / len(selected) if len(selected) else None
        for rank_position, row in enumerate(selected.iter_rows(named=True), start=1):
            score = float(row[score_col])
            signals.append(
                Signal(
                    date=date.isoformat(),
                    symbol=str(row[symbol_col]).zfill(6),
                    signal_type="buy",
                    source=f"model_{source}",
                    score=score,
                    weight=weight,
                    metadata={
                        "model_score": score,
                        "rank_position": rank_position,
                        "daily_universe_count": len(ranked),
                    },
                )
            )
    return signals_to_frame(signals)
