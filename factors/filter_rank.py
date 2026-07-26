"""Two-stage factor signals: filter on one factor, rank on another."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import polars as pl

from domain.tabular import metadata_to_json
from signals.schema import Signal, signals_to_frame

from factors.alpha101 import Alpha101, Alpha101Panels, normalize_alpha_name


@dataclass(frozen=True)
class FilterRankConfig:
    """Settings for a filter-then-rank cross-sectional factor signal."""

    filter_factor: str = "alpha_077"
    rank_factor: str = "alpha_040"
    filter_top_quantile: float = 0.5
    top_n: int = 10
    rank_start: int = 1
    rank_end: int | None = None
    filter_ascending: bool = False
    rank_ascending: bool = False
    factor_lag_days: int = 1
    min_universe: int = 20
    min_listing_days: int = 60
    liquidity_lookback_days: int = 20
    min_liquidity: float = 0.0
    source: str = "alpha077_filter_alpha040_rank"

    def __post_init__(self) -> None:
        object.__setattr__(self, "filter_factor", normalize_alpha_name(self.filter_factor))
        object.__setattr__(self, "rank_factor", normalize_alpha_name(self.rank_factor))
        if self.filter_factor == self.rank_factor:
            raise ValueError("filter_factor and rank_factor must be different")
        if not 0.0 < self.filter_top_quantile <= 1.0:
            raise ValueError("filter_top_quantile must be in (0, 1]")
        if self.top_n <= 0:
            raise ValueError("top_n must be positive")
        if self.rank_start <= 0:
            raise ValueError("rank_start must be positive")
        if self.rank_end is not None and self.rank_end < self.rank_start:
            raise ValueError("rank_end must be greater than or equal to rank_start")
        if self.factor_lag_days < 0:
            raise ValueError("factor_lag_days must be non-negative")
        if self.min_universe < 2:
            raise ValueError("min_universe must be at least 2")
        if self.min_listing_days < 0:
            raise ValueError("min_listing_days must be non-negative")
        if self.liquidity_lookback_days <= 0:
            raise ValueError("liquidity_lookback_days must be positive")
        if self.min_liquidity < 0:
            raise ValueError("min_liquidity must be non-negative")

    @property
    def resolved_rank_end(self) -> int:
        """Inclusive final rank; top_n is the fallback for compatibility."""

        return self.rank_end if self.rank_end is not None else self.rank_start + self.top_n - 1

    @property
    def selected_rank_count(self) -> int:
        return self.resolved_rank_end - self.rank_start + 1


@dataclass(frozen=True)
class FilterRankResult:
    signals: pl.DataFrame
    selections: pd.DataFrame
    daily_summary: pd.DataFrame
    filter_status: pd.DataFrame


def build_filter_rank_frame(
    panels: Alpha101Panels,
    *,
    config: FilterRankConfig | None = None,
    dates: Sequence[pd.Timestamp | str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate lagged factor values and point-in-time eligibility fields."""

    settings = config or FilterRankConfig()
    calculator = Alpha101(panels)
    close = panels.close.sort_index()
    evaluation_dates = _resolve_dates(close.index, dates)
    lag = settings.factor_lag_days

    filter_panel = (
        calculator.calculate(settings.filter_factor)
        .reindex(index=close.index, columns=close.columns)
        .shift(lag)
        .loc[evaluation_dates]
    )
    rank_panel = (
        calculator.calculate(settings.rank_factor)
        .reindex(index=close.index, columns=close.columns)
        .shift(lag)
        .loc[evaluation_dates]
    )
    reference_close = close.shift(lag).loc[evaluation_dates]
    listing_age = close.notna().cumsum().shift(lag).loc[evaluation_dates]
    turnover = (
        panels.turnover_value.reindex(index=close.index, columns=close.columns)
        if panels.turnover_value is not None
        else close * panels.volume.reindex(index=close.index, columns=close.columns)
    )
    average_turnover = (
        turnover.rolling(settings.liquidity_lookback_days, min_periods=1)
        .mean()
        .shift(lag)
        .loc[evaluation_dates]
    )

    is_st_available = panels.is_st is not None and not panels.is_st.isna().all().all()
    if is_st_available:
        is_st = panels.is_st.reindex(index=close.index, columns=close.columns).shift(lag).loc[
            evaluation_dates
        ]
    else:
        is_st = pd.DataFrame(False, index=evaluation_dates, columns=close.columns)

    eligible = reference_close.notna()
    eligible &= listing_age >= settings.min_listing_days
    if settings.min_liquidity > 0:
        eligible &= average_turnover >= settings.min_liquidity
    if is_st_available:
        eligible &= ~is_st.fillna(False).astype(bool)

    frame = pd.concat(
        [
            _to_long(filter_panel, "filter_value"),
            _to_long(rank_panel, "rank_value"),
            _to_long(reference_close, "reference_close"),
            _to_long(listing_age, "listing_age_days"),
            _to_long(average_turnover, "avg_turnover_lagged"),
            _to_long(is_st, "is_st"),
            _to_long(eligible, "eligible"),
        ],
        axis=1,
    ).reset_index()
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["eligible"] = frame["eligible"].fillna(False).astype(bool)
    status = pd.DataFrame(
        [
            {
                "filter": "listing_age",
                "status": "applied",
                "detail": f"minimum {settings.min_listing_days} prior trading observations",
            },
            {
                "filter": "liquidity",
                "status": "applied" if settings.min_liquidity > 0 else "not_requested",
                "detail": (
                    f"{settings.liquidity_lookback_days}-day lagged average turnover >= "
                    f"{settings.min_liquidity}"
                ),
            },
            {
                "filter": "is_st",
                "status": "applied" if is_st_available else "missing_input",
                "detail": "point-in-time is_st field" if is_st_available else "is_st not present",
            },
        ]
    )
    return frame, status


def filter_then_rank(
    factor_frame: pd.DataFrame,
    *,
    config: FilterRankConfig | None = None,
    filter_status: pd.DataFrame | None = None,
) -> FilterRankResult:
    """Filter the daily universe, then rank only the surviving stocks."""

    settings = config or FilterRankConfig()
    required = {"date", "symbol", "filter_value", "rank_value"}
    missing = required.difference(factor_frame.columns)
    if missing:
        raise ValueError(f"missing filter-rank columns: {', '.join(sorted(missing))}")
    if factor_frame.duplicated(["date", "symbol"]).any():
        raise ValueError("factor_frame contains duplicate date/symbol rows")

    frame = factor_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["filter_value"] = pd.to_numeric(frame["filter_value"], errors="coerce")
    frame["rank_value"] = pd.to_numeric(frame["rank_value"], errors="coerce")
    if "eligible" not in frame.columns:
        frame["eligible"] = True
    else:
        frame["eligible"] = frame["eligible"].fillna(False).astype(bool)

    signal_rows: list[Signal] = []
    selection_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    percentile_cutoff = 1.0 - settings.filter_top_quantile
    for date, daily in frame.groupby("date", sort=True):
        complete = daily.dropna(subset=["filter_value", "rank_value"])
        eligible = complete.loc[complete["eligible"]].copy()
        summary: dict[str, object] = {
            "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
            "complete_factor_count": len(complete),
            "eligible_count": len(eligible),
            "filtered_count": 0,
            "selected_count": 0,
            "filter_percentile_cutoff": percentile_cutoff,
            "filter_value_cutoff": np.nan,
            "rank_start": settings.rank_start,
            "rank_end": settings.resolved_rank_end,
            "status": "success",
            "message": "",
        }
        if len(eligible) < settings.min_universe:
            summary["status"] = "skipped"
            summary["message"] = (
                f"eligible universe {len(eligible)} is below min_universe "
                f"{settings.min_universe}"
            )
            summaries.append(summary)
            continue

        eligible["filter_percentile"] = _higher_is_better_percentile(
            eligible["filter_value"],
            lower_is_better=settings.filter_ascending,
        )
        retain_count = max(1, math.ceil(len(eligible) * settings.filter_top_quantile))
        passed = (
            eligible.sort_values(
                ["filter_value", "symbol"],
                ascending=[settings.filter_ascending, True],
                kind="mergesort",
            )
            .head(retain_count)
            .copy()
        )
        summary["filtered_count"] = len(passed)
        if passed.empty:
            summary["status"] = "skipped"
            summary["message"] = "no stocks passed the filter percentile"
            summaries.append(summary)
            continue

        summary["filter_value_cutoff"] = (
            passed["filter_value"].max()
            if settings.filter_ascending
            else passed["filter_value"].min()
        )
        passed["rank_percentile"] = _higher_is_better_percentile(
            passed["rank_value"],
            lower_is_better=settings.rank_ascending,
        )
        ranked = passed.sort_values(
            ["rank_value", "symbol"],
            ascending=[settings.rank_ascending, True],
            kind="mergesort",
        )
        ranked["rank_position"] = np.arange(1, len(ranked) + 1)
        selected = ranked.iloc[
            settings.rank_start - 1 : settings.resolved_rank_end
        ].copy()
        if selected.empty:
            summary["status"] = "skipped"
            summary["message"] = (
                f"filtered universe {len(passed)} does not reach rank "
                f"{settings.rank_start}"
            )
            summaries.append(summary)
            continue
        selected["weight"] = 1.0 / len(selected)
        selected["filter_factor"] = settings.filter_factor
        selected["rank_factor"] = settings.rank_factor
        selected["universe_count"] = len(eligible)
        selected["filtered_count"] = len(passed)
        selected["factor_lag_days"] = settings.factor_lag_days
        summary["selected_count"] = len(selected)
        summaries.append(summary)
        selection_frames.append(selected)

        for _, row in selected.iterrows():
            metadata = {
                "filter_factor": settings.filter_factor,
                "filter_value": float(row["filter_value"]),
                "filter_percentile": float(row["filter_percentile"]),
                "rank_factor": settings.rank_factor,
                "rank_value": float(row["rank_value"]),
                "rank_percentile": float(row["rank_percentile"]),
                "rank_position": int(row["rank_position"]),
                "filtered_count": len(passed),
                "eligible_count": len(eligible),
                "factor_lag_days": settings.factor_lag_days,
            }
            signal_rows.append(
                Signal(
                    date=pd.Timestamp(date).strftime("%Y-%m-%d"),
                    symbol=str(row["symbol"]).zfill(6),
                    signal_type="buy",
                    source=settings.source,
                    score=float(row["rank_percentile"]),
                    weight=float(row["weight"]),
                    metadata=metadata,
                )
            )

    selections = (
        pd.concat(selection_frames, ignore_index=True)
        if selection_frames
        else pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "filter_value",
                "filter_percentile",
                "rank_value",
                "rank_percentile",
                "rank_position",
                "weight",
            ]
        )
    )
    return FilterRankResult(
        signals=signals_to_frame(signal_rows),
        selections=selections,
        daily_summary=pd.DataFrame(summaries),
        filter_status=(filter_status.copy() if filter_status is not None else pd.DataFrame()),
    )


def write_filter_rank_reports(
    result: FilterRankResult,
    output_dir: str | Path,
    *,
    config: FilterRankConfig | None = None,
) -> Path:
    """Write unified signals and selection diagnostics."""

    settings = config or FilterRankConfig()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    signals_csv = metadata_to_json(result.signals)
    _atomic_write_csv(destination / "signals.csv", signals_csv)
    _atomic_write_csv(destination / "selections.csv", result.selections)
    _atomic_write_csv(destination / "daily_summary.csv", result.daily_summary)
    _atomic_write_csv(destination / "filter_status.csv", result.filter_status)
    _atomic_write_json(
        destination / "signals.json",
        {"signals": result.signals.to_dicts()},
    )
    manifest = {
        "strategy": "filter_then_rank",
        "settings": asdict(settings),
        "signal_count": len(result.signals),
        "date_count": int(result.daily_summary["date"].nunique())
        if not result.daily_summary.empty
        else 0,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "outputs": {
            "signals_csv": "signals.csv",
            "signals_json": "signals.json",
            "selections": "selections.csv",
            "daily_summary": "daily_summary.csv",
            "filter_status": "filter_status.csv",
        },
    }
    _atomic_write_json(destination / "manifest.json", manifest)
    return destination


def _resolve_dates(
    available: pd.DatetimeIndex,
    requested: Sequence[pd.Timestamp | str] | None,
) -> pd.DatetimeIndex:
    if requested is None:
        return available
    normalized = pd.DatetimeIndex(pd.to_datetime(list(requested)))
    missing = normalized.difference(available)
    if not missing.empty:
        raise ValueError(
            "requested selection dates are not trading dates: "
            + ", ".join(str(value.date()) for value in missing)
        )
    return available[available.isin(normalized)]


def _to_long(panel: pd.DataFrame, name: str) -> pd.Series:
    return panel.rename_axis(index="date", columns="symbol").stack(future_stack=True).rename(name)


def _higher_is_better_percentile(
    values: pd.Series,
    *,
    lower_is_better: bool,
) -> pd.Series:
    return values.rank(method="average", pct=True, ascending=not lower_is_better)


def _atomic_write_csv(path: Path, frame: pd.DataFrame | pl.DataFrame) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    if isinstance(frame, pl.DataFrame):
        frame.write_csv(temp)
    else:
        frame.to_csv(temp, index=False)
    os.replace(temp, path)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    os.replace(temp, path)


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    raise TypeError(f"not JSON serializable: {type(value).__name__}")
