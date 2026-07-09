"""Compatibility alias for signal return reports."""

from __future__ import annotations

import sys

from reports import signal_returns as _signal_returns

sys.modules[__name__] = _signal_returns
