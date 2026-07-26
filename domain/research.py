"""Typed results for ML dataset, model fit, and comparison workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .values import DateRange


@dataclass(frozen=True)
class MLDatasetResult:
    factors: tuple[str, ...]
    target_columns: tuple[str, ...]
    feature_rows: int
    label_rows: int
    date_range: DateRange
    factor_missing_rows: dict[str, int] = field(default_factory=dict)
    label_missing_rows: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelFitResult:
    model: str
    feature_columns: tuple[str, ...]
    target_column: str
    window_count: int
    prediction_count: int
    signal_count: int
    out_of_sample_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MultifactorComparisonResult:
    models: tuple[str, ...]
    factors: tuple[str, ...]
    target_column: str
    leaderboard: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class FactorResearchPipelineResult:
    """Summary of one complete factor-library research workflow."""

    factor_family: str
    evaluated_factors: tuple[str, ...]
    deduplicated_factors: tuple[str, ...]
    ml_candidate_factors: tuple[str, ...]
    models: tuple[str, ...]
    ml_status: str
