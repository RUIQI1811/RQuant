from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional

import pandas as pd


SIGNAL_COLUMNS = [
    "date",
    "symbol",
    "signal_type",
    "source",
    "score",
    "weight",
    "metadata",
]


@dataclass(frozen=True)
class Signal:
    """Unified signal record emitted by factor and custom strategy tracks."""

    date: str
    symbol: str
    signal_type: str = "buy"
    source: str = ""
    score: Optional[float] = None
    weight: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["symbol"] = str(data["symbol"]).zfill(6)
        return data


def signals_to_frame(signals: Iterable[Signal]) -> pd.DataFrame:
    """Convert Signal objects to a stable long-format DataFrame."""
    rows = [signal.to_dict() for signal in signals]
    if not rows:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)
    frame = pd.DataFrame(rows)
    for col in SIGNAL_COLUMNS:
        if col not in frame.columns:
            frame[col] = None
    return frame[SIGNAL_COLUMNS]


def frame_to_signals(frame: pd.DataFrame) -> list[Signal]:
    """Convert a signal DataFrame back to Signal objects."""
    if frame is None or frame.empty:
        return []
    signals: list[Signal] = []
    for row in frame.to_dict("records"):
        metadata = row.get("metadata")
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
