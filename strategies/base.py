from __future__ import annotations

from typing import Protocol

import pandas as pd


class StrategySignalEngine(Protocol):
    """Protocol for custom buy-strategy signal engines."""

    source: str

    def generate_signals(self) -> pd.DataFrame:
        """Return unified signal DataFrame columns defined in signals.schema."""
        ...
