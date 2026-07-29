"""Cross-sectional correlation diagnostics for built-in or external factors."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from factors.alpha101 import Alpha101, Alpha101Panels, normalize_alpha_name
from factors.external import normalize_external_factor_name
from factors.gtja191 import GTJA191, GTJA191Panels, normalize_gtja_name


@dataclass(frozen=True)
class FactorCorrelationConfig:
    """Settings for daily cross-sectional factor correlations."""

    start_date: str | None = None
    end_date: str | None = None
    factor_lag_days: int = 1
    min_observations: int = 20
    min_dates: int = 20
    high_correlation_threshold: float = 0.8

    def __post_init__(self) -> None:
        if self.factor_lag_days < 0:
            raise ValueError("factor_lag_days must be non-negative")
        if self.min_observations < 2:
            raise ValueError("min_observations must be at least 2")
        if self.min_dates < 1:
            raise ValueError("min_dates must be positive")
        if not 0.0 <= self.high_correlation_threshold <= 1.0:
            raise ValueError("high_correlation_threshold must be between 0 and 1")
        if self.start_date and self.end_date:
            if pd.Timestamp(self.start_date) > pd.Timestamp(self.end_date):
                raise ValueError("start_date must not be after end_date")

    def to_dict(self) -> dict[str, object]:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "factor_lag_days": self.factor_lag_days,
            "min_observations": self.min_observations,
            "min_dates": self.min_dates,
            "high_correlation_threshold": self.high_correlation_threshold,
        }


@dataclass(frozen=True)
class FactorCorrelationResult:
    """In-memory correlation reports and factor-calculation status."""

    spearman: pd.DataFrame
    pearson: pd.DataFrame
    valid_dates: pd.DataFrame
    pairs: pd.DataFrame
    deduplication: pd.DataFrame
    deduplicated_factors: pd.DataFrame
    status: pd.DataFrame
    evaluation_dates: pd.DatetimeIndex

    @property
    def successful_factors(self) -> tuple[str, ...]:
        if self.status.empty:
            return ()
        return tuple(self.status.loc[self.status["status"].eq("success"), "factor"])

    @property
    def failed_factors(self) -> tuple[str, ...]:
        if self.status.empty:
            return ()
        return tuple(self.status.loc[self.status["status"].eq("failed"), "factor"])


def calculate_factor_correlations(
    panels: Alpha101Panels,
    factors: Sequence[str],
    *,
    config: FactorCorrelationConfig | None = None,
    factor_statuses: Mapping[str, str] | None = None,
    priority_scores: Mapping[str, float] | None = None,
) -> FactorCorrelationResult:
    """Calculate mean daily cross-sectional factor-correlation matrices.

    Each factor is lagged before evaluation. Correlations are computed within
    every trading-date cross-section and then averaged across valid dates. This
    avoids mixing a factor's time-series level changes with stock-selection
    similarity.
    """

    selected = tuple(dict.fromkeys(normalize_alpha_name(name) for name in factors))
    return _calculate_panel_factor_correlations(
        panels.close,
        selected,
        calculate=Alpha101(panels).calculate,
        config=config,
        factor_statuses=factor_statuses,
        priority_scores=priority_scores,
    )


def calculate_gtja_factor_correlations(
    panels: GTJA191Panels,
    factors: Sequence[str | int],
    *,
    config: FactorCorrelationConfig | None = None,
    factor_statuses: Mapping[str, str] | None = None,
    factor_directions: Mapping[str, int] | None = None,
    priority_scores: Mapping[str, float] | None = None,
) -> FactorCorrelationResult:
    """Calculate point-in-time cross-sectional correlations for GTJA191."""

    selected = tuple(dict.fromkeys(normalize_gtja_name(name) for name in factors))
    directions = factor_directions or {}
    invalid_directions = set(directions.values()).difference((-1, 1))
    if invalid_directions:
        raise ValueError("GTJA191 factor directions must be -1 or 1")
    calculator = GTJA191(panels)

    def calculate_directed(name: str) -> pd.DataFrame:
        return calculator.calculate(name) * int(directions.get(name, 1))

    return _calculate_panel_factor_correlations(
        panels.close,
        selected,
        calculate=calculate_directed,
        config=config,
        factor_statuses=factor_statuses,
        priority_scores=priority_scores,
    )


def _calculate_panel_factor_correlations(
    close: pd.DataFrame,
    selected: Sequence[str],
    *,
    calculate: Callable[[str], pd.DataFrame],
    config: FactorCorrelationConfig | None,
    factor_statuses: Mapping[str, str] | None,
    priority_scores: Mapping[str, float] | None,
) -> FactorCorrelationResult:
    settings = config or FactorCorrelationConfig()
    if len(selected) < 2:
        raise ValueError("at least two factors are required")

    close = close.sort_index()
    evaluation_dates = close.index
    if settings.start_date:
        evaluation_dates = evaluation_dates[evaluation_dates >= pd.Timestamp(settings.start_date)]
    if settings.end_date:
        evaluation_dates = evaluation_dates[evaluation_dates <= pd.Timestamp(settings.end_date)]
    if evaluation_dates.empty:
        raise ValueError("no trading dates remain after applying start_date/end_date")

    values = np.full(
        (len(evaluation_dates), len(close.columns), len(selected)),
        np.nan,
        dtype=np.float32,
    )
    statuses: list[dict[str, object]] = []
    successful_indices: list[int] = []
    supplied_statuses = factor_statuses or {}
    for factor_index, factor_name in enumerate(selected):
        started = time.perf_counter()
        try:
            factor_panel = (
                calculate(factor_name)
                .reindex(index=close.index, columns=close.columns)
                .shift(settings.factor_lag_days)
                .loc[evaluation_dates]
                .replace([np.inf, -np.inf], np.nan)
            )
            factor_array = factor_panel.to_numpy(dtype=np.float32, na_value=np.nan)
            values[:, :, factor_index] = factor_array
            observation_count = int(np.isfinite(factor_array).sum())
            successful_indices.append(factor_index)
            statuses.append(
                _status_row(
                    factor_name,
                    "success",
                    time.perf_counter() - started,
                    observation_count=observation_count,
                    factor_status=supplied_statuses.get(factor_name, "active"),
                )
            )
        except Exception as exc:
            statuses.append(
                _status_row(
                    factor_name,
                    "failed",
                    time.perf_counter() - started,
                    observation_count=0,
                    factor_status=supplied_statuses.get(factor_name, "active"),
                    message=f"{type(exc).__name__}: {exc}",
                )
            )

    return _correlation_result_from_values(
        values,
        selected=selected,
        successful_indices=successful_indices,
        statuses=statuses,
        evaluation_dates=evaluation_dates,
        settings=settings,
        priority_scores=priority_scores,
    )


def calculate_external_factor_correlations(
    frame: pd.DataFrame,
    factors: Sequence[str],
    *,
    config: FactorCorrelationConfig | None = None,
    factor_statuses: Mapping[str, str] | None = None,
    priority_scores: Mapping[str, float] | None = None,
    date_col: str = "date",
    symbol_col: str = "symbol",
) -> FactorCorrelationResult:
    """Calculate correlations from an unlagged external factor wide frame."""

    settings = config or FactorCorrelationConfig()
    selected = tuple(
        dict.fromkeys(normalize_external_factor_name(name) for name in factors)
    )
    if len(selected) < 2:
        raise ValueError("at least two factors are required")
    missing = {date_col, symbol_col, *selected}.difference(frame.columns)
    if missing:
        raise ValueError(
            "external factor frame missing columns: " + ", ".join(sorted(missing))
        )

    work = frame[[date_col, symbol_col, *selected]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    if work[date_col].isna().any():
        raise ValueError("external factor frame contains invalid dates")
    work[symbol_col] = work[symbol_col].astype(str).str.zfill(6)
    if work.duplicated([date_col, symbol_col]).any():
        raise ValueError("external factor frame contains duplicate date/symbol rows")
    evaluation_dates = pd.DatetimeIndex(sorted(work[date_col].unique()))
    if settings.start_date:
        evaluation_dates = evaluation_dates[
            evaluation_dates >= pd.Timestamp(settings.start_date)
        ]
    if settings.end_date:
        evaluation_dates = evaluation_dates[
            evaluation_dates <= pd.Timestamp(settings.end_date)
        ]
    if evaluation_dates.empty:
        raise ValueError("no trading dates remain after applying start_date/end_date")
    symbols = pd.Index(sorted(work[symbol_col].unique()), name="symbol")
    values = np.full(
        (len(evaluation_dates), len(symbols), len(selected)),
        np.nan,
        dtype=np.float32,
    )
    statuses: list[dict[str, object]] = []
    successful_indices: list[int] = []
    supplied_statuses = factor_statuses or {}
    for factor_index, factor_name in enumerate(selected):
        started = time.perf_counter()
        try:
            numeric = pd.to_numeric(work[factor_name], errors="coerce")
            panel = (
                work.assign(__value=numeric)
                .pivot(index=date_col, columns=symbol_col, values="__value")
                .reindex(index=evaluation_dates, columns=symbols)
                .shift(settings.factor_lag_days)
                .replace([np.inf, -np.inf], np.nan)
            )
            factor_array = panel.to_numpy(dtype=np.float32, na_value=np.nan)
            values[:, :, factor_index] = factor_array
            observation_count = int(np.isfinite(factor_array).sum())
            successful_indices.append(factor_index)
            statuses.append(
                _status_row(
                    factor_name,
                    "success",
                    time.perf_counter() - started,
                    observation_count=observation_count,
                    factor_status=supplied_statuses.get(factor_name, "active"),
                )
            )
        except Exception as exc:
            statuses.append(
                _status_row(
                    factor_name,
                    "failed",
                    time.perf_counter() - started,
                    observation_count=0,
                    factor_status=supplied_statuses.get(factor_name, "active"),
                    message=f"{type(exc).__name__}: {exc}",
                )
            )
    return _correlation_result_from_values(
        values,
        selected=selected,
        successful_indices=successful_indices,
        statuses=statuses,
        evaluation_dates=evaluation_dates,
        settings=settings,
        priority_scores=priority_scores,
    )


def _correlation_result_from_values(
    values: np.ndarray,
    *,
    selected: Sequence[str],
    successful_indices: Sequence[int],
    statuses: Sequence[dict[str, object]],
    evaluation_dates: pd.DatetimeIndex,
    settings: FactorCorrelationConfig,
    priority_scores: Mapping[str, float] | None,
) -> FactorCorrelationResult:
    if len(successful_indices) < 2:
        failures = "; ".join(
            f"{row['factor']}: {row['message']}"
            for row in statuses
            if row["status"] == "failed"
        )
        raise ValueError(f"fewer than two factors were calculated successfully: {failures}")

    successful_factors = tuple(selected[index] for index in successful_indices)
    size = len(successful_factors)
    sums = {
        "spearman": np.zeros((size, size), dtype=np.float64),
        "pearson": np.zeros((size, size), dtype=np.float64),
    }
    counts = {
        "spearman": np.zeros((size, size), dtype=np.int64),
        "pearson": np.zeros((size, size), dtype=np.int64),
    }

    for date_index in range(len(evaluation_dates)):
        daily = pd.DataFrame(
            values[date_index][:, successful_indices],
            columns=successful_factors,
        )
        for method in ("spearman", "pearson"):
            correlation = daily.corr(
                method=method,
                min_periods=settings.min_observations,
            ).to_numpy(dtype=np.float64)
            valid = np.isfinite(correlation)
            sums[method] += np.where(valid, correlation, 0.0)
            counts[method] += valid

    matrices: dict[str, pd.DataFrame] = {}
    for method in ("spearman", "pearson"):
        matrix = np.full((size, size), np.nan, dtype=np.float64)
        eligible = counts[method] >= settings.min_dates
        np.divide(sums[method], counts[method], out=matrix, where=eligible)
        matrices[method] = pd.DataFrame(
            matrix,
            index=pd.Index(successful_factors, name="factor"),
            columns=successful_factors,
        )

    valid_dates = pd.DataFrame(
        counts["spearman"],
        index=pd.Index(successful_factors, name="factor"),
        columns=successful_factors,
    )
    pairs = _build_pair_report(
        matrices["spearman"],
        matrices["pearson"],
        valid_dates,
        threshold=settings.high_correlation_threshold,
    )
    deduplication, deduplicated_factors = _build_deduplication_report(
        successful_factors,
        pairs,
        matrices["spearman"],
        priority_scores=priority_scores,
    )
    return FactorCorrelationResult(
        spearman=matrices["spearman"],
        pearson=matrices["pearson"],
        valid_dates=valid_dates,
        pairs=pairs,
        deduplication=deduplication,
        deduplicated_factors=deduplicated_factors,
        status=pd.DataFrame(statuses),
        evaluation_dates=evaluation_dates,
    )


def write_factor_correlation_reports(
    result: FactorCorrelationResult,
    output_dir: str | Path,
    *,
    config: FactorCorrelationConfig | None = None,
    eligible_factors: Sequence[str] | None = None,
    eligibility_settings: Mapping[str, object] | None = None,
) -> Path:
    """Write CSV, HTML, and manifest outputs atomically where practical."""

    settings = config or FactorCorrelationConfig()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(destination / "spearman_matrix.csv", result.spearman, index=True)
    _atomic_write_csv(destination / "pearson_matrix.csv", result.pearson, index=True)
    _atomic_write_csv(destination / "valid_date_count_matrix.csv", result.valid_dates, index=True)
    _atomic_write_csv(destination / "correlation_pairs.csv", result.pairs, index=False)
    _atomic_write_csv(destination / "deduplication.csv", result.deduplication, index=False)
    _atomic_write_csv(
        destination / "deduplicated_factors.csv",
        result.deduplicated_factors,
        index=False,
    )
    _atomic_write_csv(destination / "factor_status.csv", result.status, index=False)
    ml_candidate_factors: pd.DataFrame | None = None
    if eligible_factors is not None:
        eligible = {str(factor) for factor in eligible_factors}
        ml_candidate_factors = result.deduplicated_factors.loc[
            result.deduplicated_factors["factor"].astype(str).isin(eligible)
        ].reset_index(drop=True)
        _atomic_write_csv(
            destination / "ml_candidate_factors.csv",
            ml_candidate_factors,
            index=False,
        )
    _write_heatmap(result.spearman, destination / "spearman_heatmap.html")

    manifest = {
        "method": "mean_daily_cross_sectional_correlation",
        "primary_matrix": "spearman",
        "successful_factors": list(result.successful_factors),
        "failed_factors": list(result.failed_factors),
        "evaluation_start": str(result.evaluation_dates.min().date()),
        "evaluation_end": str(result.evaluation_dates.max().date()),
        "evaluation_date_count": len(result.evaluation_dates),
        "settings": settings.to_dict(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "eligibility_settings": dict(eligibility_settings or {}),
        "eligible_factor_count": (
            len(set(str(factor) for factor in eligible_factors))
            if eligible_factors is not None
            else None
        ),
        "ml_candidate_factor_count": (
            len(ml_candidate_factors) if ml_candidate_factors is not None else None
        ),
        "outputs": {
            "spearman": "spearman_matrix.csv",
            "pearson": "pearson_matrix.csv",
            "valid_dates": "valid_date_count_matrix.csv",
            "pairs": "correlation_pairs.csv",
            "deduplication": "deduplication.csv",
            "deduplicated_factors": "deduplicated_factors.csv",
            "status": "factor_status.csv",
            "heatmap": "spearman_heatmap.html",
            **(
                {"ml_candidate_factors": "ml_candidate_factors.csv"}
                if ml_candidate_factors is not None
                else {}
            ),
        },
    }
    _atomic_write_json(destination / "manifest.json", manifest)
    return destination


def _build_pair_report(
    spearman: pd.DataFrame,
    pearson: pd.DataFrame,
    valid_dates: pd.DataFrame,
    *,
    threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    factors = list(spearman.columns)
    for left_index, left in enumerate(factors):
        for right in factors[left_index + 1 :]:
            rank_correlation = spearman.at[left, right]
            rows.append(
                {
                    "factor_a": left,
                    "factor_b": right,
                    "spearman": rank_correlation,
                    "pearson": pearson.at[left, right],
                    "abs_spearman": abs(rank_correlation) if pd.notna(rank_correlation) else np.nan,
                    "valid_dates": int(valid_dates.at[left, right]),
                    "high_correlation": bool(
                        pd.notna(rank_correlation) and abs(rank_correlation) >= threshold
                    ),
                }
            )
    columns = [
        "factor_a",
        "factor_b",
        "spearman",
        "pearson",
        "abs_spearman",
        "valid_dates",
        "high_correlation",
    ]
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["abs_spearman", "factor_a", "factor_b"], ascending=[False, True, True])
        .reset_index(drop=True)
    )


def _build_deduplication_report(
    factors: Sequence[str],
    pairs: pd.DataFrame,
    spearman: pd.DataFrame,
    *,
    priority_scores: Mapping[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster high-correlation factors and retain one auditable representative."""
    adjacency = {factor: set() for factor in factors}
    if not pairs.empty:
        for row in pairs.loc[pairs["high_correlation"].astype(bool)].itertuples():
            adjacency[str(row.factor_a)].add(str(row.factor_b))
            adjacency[str(row.factor_b)].add(str(row.factor_a))

    supplied_scores = priority_scores or {}
    scores: dict[str, float] = {}
    for factor in factors:
        raw = supplied_scores.get(factor, np.nan)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = np.nan
        scores[factor] = value if np.isfinite(value) else np.nan

    rows: list[dict[str, object]] = []
    visited: set[str] = set()
    cluster_id = 0
    for seed in sorted(factors):
        if seed in visited:
            continue
        cluster_id += 1
        stack = [seed]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(sorted(adjacency[current] - component, reverse=True))
        visited.update(component)
        has_priority = any(np.isfinite(scores[factor]) for factor in component)
        representative = sorted(
            component,
            key=lambda factor: (
                -scores[factor] if np.isfinite(scores[factor]) else np.inf,
                factor,
            ),
        )[0]
        reason = "highest_priority_score" if has_priority else "name_order_fallback"
        for factor in sorted(component):
            correlation = (
                float(spearman.at[factor, representative])
                if factor in spearman.index
                and representative in spearman.columns
                and pd.notna(spearman.at[factor, representative])
                else np.nan
            )
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "factor": factor,
                    "representative": representative,
                    "kept": factor == representative,
                    "priority_score": scores[factor],
                    "spearman_to_representative": correlation,
                    "selection_reason": reason,
                }
            )
    deduplication = pd.DataFrame(rows)
    kept = (
        deduplication.loc[deduplication["kept"], ["factor", "cluster_id", "priority_score"]]
        .sort_values(["cluster_id", "factor"])
        .reset_index(drop=True)
    )
    return deduplication, kept


def _write_heatmap(matrix: pd.DataFrame, path: Path) -> None:
    import plotly.graph_objects as go

    factor_count = len(matrix)
    heatmap_kwargs: dict[str, object] = {}
    if factor_count <= 30:
        heatmap_kwargs["texttemplate"] = "%{z:.2f}"
    figure = go.Figure(
        data=go.Heatmap(
            z=matrix.to_numpy(dtype=float),
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale="RdBu_r",
            colorbar={"title": "Spearman"},
            hovertemplate="%{y} vs %{x}<br>Spearman=%{z:.4f}<extra></extra>",
            **heatmap_kwargs,
        )
    )
    figure.update_layout(
        title="Factor correlation matrix (mean daily cross-sectional Spearman)",
        template="plotly_white",
        width=max(800, factor_count * 55),
        height=max(800, factor_count * 55),
        xaxis={"side": "top"},
        yaxis={"autorange": "reversed"},
    )
    temp = path.with_name(f".{path.name}.tmp")
    figure.write_html(str(temp), include_plotlyjs="cdn")
    os.replace(temp, path)


def _status_row(
    factor: str,
    status: str,
    duration: float,
    *,
    observation_count: int,
    factor_status: str,
    message: str = "",
) -> dict[str, object]:
    return {
        "factor": factor,
        "status": status,
        "factor_status": factor_status,
        "observation_count": observation_count,
        "duration_seconds": round(float(duration), 3),
        "message": message,
    }


def _atomic_write_csv(path: Path, frame: pd.DataFrame, *, index: bool) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temp, index=index)
    os.replace(temp, path)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)
