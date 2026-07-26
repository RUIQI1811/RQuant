from __future__ import annotations

import json
from typing import Any, Iterable

import polars as pl

from domain.signals import Signal
from domain.tabular import to_polars


SIGNAL_COLUMNS = [
    "date",
    "symbol",
    "signal_type",
    "source",
    "score",
    "weight",
    "metadata",
]


def signals_to_frame(signals: Iterable[Signal]) -> pl.DataFrame:
    """Convert Signal objects to a stable long-format DataFrame."""
    rows = [signal.to_dict() for signal in signals]
    if not rows:
        return pl.DataFrame(
            schema={
                "date": pl.String,
                "symbol": pl.String,
                "signal_type": pl.String,
                "source": pl.String,
                "score": pl.Float64,
                "weight": pl.Float64,
                "metadata": pl.Object,
            }
        )
    return pl.DataFrame(
        {
            "date": [row["date"] for row in rows],
            "symbol": [row["symbol"] for row in rows],
            "signal_type": [row["signal_type"] for row in rows],
            "source": [row["source"] for row in rows],
            "score": [row["score"] for row in rows],
            "weight": [row["weight"] for row in rows],
            "metadata": pl.Series(
                "metadata", [row["metadata"] for row in rows], dtype=pl.Object
            ),
        },
        strict=False,
    ).select(SIGNAL_COLUMNS)


def frame_to_signals(frame: Any) -> list[Signal]:
    """Convert a signal DataFrame back to Signal objects."""
    polars_frame = to_polars(frame)
    if polars_frame.is_empty():
        return []
    signals: list[Signal] = []
    for row in polars_frame.to_dicts():
        metadata = row.get("metadata")
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        signals.append(
            Signal(
                date=str(row["date"]),
                symbol=str(row["symbol"]).zfill(6),
                signal_type=str(row.get("signal_type") or "buy"),
                source=str(row.get("source") or ""),
                score=row.get("score"),
                weight=row.get("weight"),
                metadata=metadata,
            )
        )
    return signals
