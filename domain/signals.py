"""Canonical research intent carried from generators into execution."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import pandas as pd

from .values import SourceId, Symbol, TradingDate


@dataclass(frozen=True)
class Signal:
    date: str
    symbol: str
    signal_type: str = "buy"
    source: str = ""
    score: float | None = None
    weight: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "date", str(TradingDate(self.date)))
        object.__setattr__(self, "symbol", str(Symbol(self.symbol)))
        object.__setattr__(self, "source", str(SourceId(self.source)))
        signal_type = str(self.signal_type).strip().lower() or "buy"
        if signal_type not in {"buy", "sell", "hold"}:
            raise ValueError(f"unsupported signal type: {self.signal_type!r}")
        object.__setattr__(self, "signal_type", signal_type)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SignalBook(Mapping[pd.Timestamp, list[str]]):
    """Date-indexed signals with a legacy code-list mapping view."""

    def __init__(self, signals: Iterable[Signal] = ()) -> None:
        grouped: dict[pd.Timestamp, list[Signal]] = {}
        for signal in signals:
            item = signal if isinstance(signal, Signal) else Signal(**signal)
            grouped.setdefault(pd.Timestamp(item.date), []).append(item)
        self._signals = grouped

    def __getitem__(self, key: pd.Timestamp) -> list[str]:
        return [signal.symbol for signal in self._signals[pd.Timestamp(key)]]

    def __iter__(self) -> Iterator[pd.Timestamp]:
        return iter(self._signals)

    def __len__(self) -> int:
        return len(self._signals)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return dict(self.items()) == dict(other.items())
        return False

    def signals_for(self, date: Any) -> tuple[Signal, ...]:
        return tuple(self._signals.get(pd.Timestamp(date), ()))

    @property
    def signals(self) -> tuple[Signal, ...]:
        return tuple(signal for daily in self._signals.values() for signal in daily)

    def limited(self, max_positions: int) -> "SignalBook":
        return SignalBook(
            signal
            for daily in self._signals.values()
            for signal in daily[:max_positions]
        )
