"""Compatibility alias for realistic portfolio backtests."""

from __future__ import annotations

import sys

from backtest import portfolio as _portfolio

sys.modules[__name__] = _portfolio
