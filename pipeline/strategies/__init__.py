"""Compatibility package for legacy pipeline strategy imports."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TOP_LEVEL_STRATEGIES = _PROJECT_ROOT / "strategies"
if str(_PROJECT_ROOT) in sys.path:
    sys.path.remove(str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT))
if __name__ == "strategies":
    __path__ = [str(_TOP_LEVEL_STRATEGIES)]

from strategies.base import StrategySignalEngine  # noqa: E402,F401
from strategies.bdsr_macd_obv import (  # noqa: E402,F401
    BDSRMACDOBVSelector,
    add_bdsr_macd_obv_features,
)
from strategies.mbdsr import MBDSRSelector, add_mbdsr_features, calc_obv, calc_rci  # noqa: E402,F401

__all__ = [
    "BDSRMACDOBVSelector",
    "MBDSRSelector",
    "StrategySignalEngine",
    "add_bdsr_macd_obv_features",
    "add_mbdsr_features",
    "calc_obv",
    "calc_rci",
]
