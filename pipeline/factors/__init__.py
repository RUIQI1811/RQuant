"""Compatibility package exports for moved factor modules."""

from factors.alpha101 import ALPHA101_NAMES
from factors.gtja191 import GTJA191_NAMES
from signals.factor_adapters import FactorSignalConfig, SimpleFactorSignalEngine, factor_frame_to_signal_frame

__all__ = [
    "ALPHA101_NAMES",
    "GTJA191_NAMES",
    "FactorSignalConfig",
    "SimpleFactorSignalEngine",
    "factor_frame_to_signal_frame",
]
