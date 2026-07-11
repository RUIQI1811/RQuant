"""Canonical cross-boundary domain contracts for RQuant."""

from .artifacts import ArtifactRef, WorkflowResult, WorkflowStatus
from .execution import (
    BacktestResult,
    BacktestSummary,
    EquityPoint,
    Fill,
    OrderIntent,
    OrderResult,
    Position,
    PositionSnapshot,
    Trade,
)
from .signals import Signal, SignalBook
from .selection import Candidate, CandidateRun, SelectionResult
from .market import FetchResult
from .factors import FactorEvaluationResult
from .reports import ResearchReportResult, SignalReturnResult, SystemDoctorResult
from .research import MLDatasetResult, ModelFitResult, MultifactorComparisonResult
from .values import DateRange, SourceId, Symbol, TradingDate, normalize_symbol

__all__ = [
    "ArtifactRef",
    "BacktestResult",
    "BacktestSummary",
    "Candidate",
    "CandidateRun",
    "DateRange",
    "EquityPoint",
    "FetchResult",
    "FactorEvaluationResult",
    "Fill",
    "MLDatasetResult",
    "ModelFitResult",
    "MultifactorComparisonResult",
    "OrderResult",
    "OrderIntent",
    "Position",
    "PositionSnapshot",
    "ResearchReportResult",
    "Signal",
    "SignalBook",
    "SignalReturnResult",
    "SelectionResult",
    "SourceId",
    "SystemDoctorResult",
    "Symbol",
    "Trade",
    "TradingDate",
    "WorkflowResult",
    "WorkflowStatus",
    "normalize_symbol",
]
