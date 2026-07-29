"""Performance metrics for portfolio equity curves."""

from __future__ import annotations

import math
from datetime import date
from typing import Any, Mapping, Sequence


def annualized_return(total_return: float, trading_days: int) -> float | None:
    if trading_days <= 0 or not math.isfinite(float(total_return)) or total_return < -1.0:
        return None
    return float((1.0 + total_return) ** (252.0 / trading_days) - 1.0)


def yearly_return_rows(
    equity_rows: Sequence[Mapping[str, Any]],
    *,
    initial_cash: float,
) -> list[dict[str, Any]]:
    """Calculate calendar-year returns from an auditable daily equity curve.

    The first year's baseline is the configured initial cash. Later years start
    from the prior year's last observed equity, so no overnight boundary return
    is dropped. Partial first/last years remain visible in the output.
    """

    dated_values: dict[date, float] = {}
    for row in equity_rows:
        raw_date = str(row.get("date", ""))[:10]
        try:
            current_date = date.fromisoformat(raw_date)
            value = float(row.get("total_value", row.get("cash")))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            dated_values[current_date] = value
    if not dated_values or initial_cash <= 0:
        return []

    by_year: dict[int, list[tuple[date, float]]] = {}
    for current_date, value in sorted(dated_values.items()):
        by_year.setdefault(current_date.year, []).append((current_date, value))

    rows: list[dict[str, Any]] = []
    baseline = float(initial_cash)
    for year, observations in by_year.items():
        first_date, _ = observations[0]
        last_date, end_value = observations[-1]
        trading_days = len(observations)
        total = end_value / baseline - 1.0
        rows.append(
            {
                "year": year,
                "period_start_date": first_date.isoformat(),
                "period_end_date": last_date.isoformat(),
                "trading_days": trading_days,
                "start_equity": baseline,
                "end_equity": end_value,
                "total_return": float(total),
                "annualized_return": annualized_return(total, trading_days),
                "is_partial_year": first_date.month != 1 or last_date.month != 12,
            }
        )
        baseline = end_value
    return rows


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
