"""Transaction cost constants and helpers for A-share backtests."""

from __future__ import annotations


def calculate_buy_cost(notional: float, commission_rate: float, min_commission: float) -> float:
    commission = max(notional * commission_rate, min_commission) if notional > 0 else 0.0
    return float(commission)


def calculate_sell_cost(
    notional: float,
    commission_rate: float,
    min_commission: float,
    stamp_tax_rate: float,
    transfer_fee_rate: float,
) -> float:
    if notional <= 0:
        return 0.0
    commission = max(notional * commission_rate, min_commission)
    stamp_tax = notional * stamp_tax_rate
    transfer_fee = notional * transfer_fee_rate
    return float(commission + stamp_tax + transfer_fee)
