"""Compatibility alias for factor portfolio backtests."""

from __future__ import annotations

import sys

from backtest import factor_portfolio as _factor_portfolio

sys.modules[__name__] = _factor_portfolio
