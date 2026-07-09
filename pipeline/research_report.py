"""Compatibility alias for combined research reports."""

from __future__ import annotations

import sys

from reports import research_report as _research_report

sys.modules[__name__] = _research_report
