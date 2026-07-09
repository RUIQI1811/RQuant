"""Performance metrics for portfolio equity curves."""

from __future__ import annotations

import math
from typing import Sequence


def annualized_return(total_return: float, trading_days: int) -> float | None:
    if trading_days <= 0:
        return None
    return float((1.0 + total_return) ** (252.0 / trading_days) - 1.0)


def max_drawdown(equity_values: Sequence[float]) -> float:
    peak = None
    worst = 0.0
    for value in equity_values:
        current = float(value)
        peak = current if peak is None else max(peak, current)
        if peak > 0:
            worst = min(worst, current / peak - 1.0)
    return float(worst)


def sharpe_ratio(daily_returns: Sequence[float]) -> float | None:
    values = [float(v) for v in daily_returns]
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    if variance <= 0:
        return None
    return float(mean / math.sqrt(variance) * math.sqrt(252.0))
