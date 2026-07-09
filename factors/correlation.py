"""Cross-sectional correlation diagnostics for Alpha101 factors."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from factors.alpha101 import Alpha101, Alpha101Panels, normalize_alpha_name


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
) -> FactorCorrelationResult:
    """Calculate mean daily cross-sectional factor-correlation matrices.

    Each factor is lagged before evaluation. Correlations are computed within
    every trading-date cross-section and then averaged across valid dates. This
    avoids mixing a factor's time-series level changes with stock-selection
    similarity.
    """

    settings = config or FactorCorrelationConfig()
    selected = tuple(dict.fromkeys(normalize_alpha_name(name) for name in factors))
    if len(selected) < 2:
        raise ValueError("at least two factors are required")

    close = panels.close.sort_index()
    evaluation_dates = close.index
    if settings.start_date:
        evaluation_dates = evaluation_dates[evaluation_dates >= pd.Timestamp(settings.start_date)]
    if settings.end_date:
        evaluation_dates = evaluation_dates[evaluation_dates <= pd.Timestamp(settings.end_date)]
    if evaluation_dates.empty:
        raise ValueError("no trading dates remain after applying start_date/end_date")

    calculator = Alpha101(panels)
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
                calculator.calculate(factor_name)
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
    return FactorCorrelationResult(
        spearman=matrices["spearman"],
        pearson=matrices["pearson"],
        valid_dates=valid_dates,
        pairs=pairs,
        status=pd.DataFrame(statuses),
        evaluation_dates=evaluation_dates,
    )


def write_factor_correlation_reports(
    result: FactorCorrelationResult,
    output_dir: str | Path,
    *,
    config: FactorCorrelationConfig | None = None,
) -> Path:
    """Write CSV, HTML, and manifest outputs atomically where practical."""

    settings = config or FactorCorrelationConfig()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(destination / "spearman_matrix.csv", result.spearman, index=True)
    _atomic_write_csv(destination / "pearson_matrix.csv", result.pearson, index=True)
    _atomic_write_csv(destination / "valid_date_count_matrix.csv", result.valid_dates, index=True)
    _atomic_write_csv(destination / "correlation_pairs.csv", result.pairs, index=False)
    _atomic_write_csv(destination / "factor_status.csv", result.status, index=False)
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
        "outputs": {
            "spearman": "spearman_matrix.csv",
            "pearson": "pearson_matrix.csv",
            "valid_dates": "valid_date_count_matrix.csv",
            "pairs": "correlation_pairs.csv",
            "status": "factor_status.csv",
            "heatmap": "spearman_heatmap.html",
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
