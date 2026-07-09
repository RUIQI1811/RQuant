from __future__ import annotations

from typing import Protocol

import pandas as pd


class FactorSignalEngine(Protocol):
    """Protocol for factor-based signal engines."""

    source: str

    def generate_signals(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """Return unified signal DataFrame columns defined in signals.schema."""
        ...
