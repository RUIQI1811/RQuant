from __future__ import annotations

from typing import Protocol

import pandas as pd

from domain.signals import SignalBook


class FactorSignalEngine(Protocol):
    """Protocol for factor-based signal engines."""

    source: str

    def generate_signal_book(self, factor_data: pd.DataFrame) -> SignalBook:
        """Return canonical signals; DataFrames are persistence adapters only."""
        ...
