from __future__ import annotations

from typing import Protocol

from domain.signals import SignalBook


class StrategySignalEngine(Protocol):
    """Protocol for custom buy-strategy signal engines."""

    source: str

    def generate_signal_book(self) -> SignalBook:
        """Return canonical strategy signals."""
        ...
