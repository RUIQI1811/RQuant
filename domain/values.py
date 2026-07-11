"""Validated primitive value objects shared across research paths."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, order=True)
class Symbol:
    value: str

    def __post_init__(self) -> None:
        normalized = str(self.value).strip().split(".", 1)[0]
        if normalized.endswith(".0") and normalized[:-2].isdigit():
            normalized = normalized[:-2]
        normalized = normalized.zfill(6)
        if len(normalized) != 6 or not normalized.isdigit():
            raise ValueError(f"symbol must be a six-digit code: {self.value!r}")
        object.__setattr__(self, "value", normalized)

    def __str__(self) -> str:
        return self.value


def normalize_symbol(value: Any) -> str:
    return str(Symbol(value))


@dataclass(frozen=True, order=True)
class TradingDate:
    value: dt.date

    def __init__(self, value: Any) -> None:
        if isinstance(value, dt.datetime):
            resolved = value.date()
        elif isinstance(value, dt.date):
            resolved = value
        else:
            text = str(value).strip()
            if not text:
                raise ValueError("trading date cannot be empty")
            resolved = dt.date.fromisoformat(text[:10])
        object.__setattr__(self, "value", resolved)

    def __str__(self) -> str:
        return self.value.isoformat()


@dataclass(frozen=True, order=True)
class SourceId:
    value: str

    def __post_init__(self) -> None:
        normalized = str(self.value).strip()
        object.__setattr__(self, "value", normalized)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class DateRange:
    start: TradingDate | None = None
    end: TradingDate | None = None

    def __post_init__(self) -> None:
        if self.start is not None and not isinstance(self.start, TradingDate):
            object.__setattr__(self, "start", TradingDate(self.start))
        if self.end is not None and not isinstance(self.end, TradingDate):
            object.__setattr__(self, "end", TradingDate(self.end))
        if self.start and self.end and self.start > self.end:
            raise ValueError("date range start must not be after end")

    def contains(self, value: Any) -> bool:
        date = value if isinstance(value, TradingDate) else TradingDate(value)
        return (self.start is None or date >= self.start) and (
            self.end is None or date <= self.end
        )
