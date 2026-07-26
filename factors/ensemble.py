"""Auditable cross-sectional rank ensembles for factor research signals."""

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
from factors.alpha101 import Alpha101, Alpha101Panels, normalize_alpha_name
from signals.schema import Signal, signals_to_frame


@dataclass(frozen=True)
class RankEnsembleConfig:
    """Settings for a weighted ensemble of daily cross-sectional factor ranks."""

    factors: tuple[str, ...] = ("alpha_040",)
    weights: tuple[float, ...] | None = None
    ascending_factors: tuple[str, ...] = ()
    min_factor_coverage: float = 1.0
    top_n: int = 10
    rank_start: int = 1
    rank_end: int | None = None
    factor_lag_days: int = 1
    min_universe: int = 20
    min_listing_days: int = 60
    liquidity_lookback_days: int = 20
    min_liquidity: float = 0.0
    source: str = "factor_rank_ensemble"

    def __post_init__(self) -> None:
        factors = tuple(_canonical_factor_name(value) for value in self.factors)
        if not factors:
            raise ValueError("factors must not be empty")
        if len(set(factors)) != len(factors):
            raise ValueError("factors must be unique")
        object.__setattr__(self, "factors", factors)

        raw_weights = self.weights
        if raw_weights is None:
            weights = tuple(1.0 for _ in factors)
        else:
            weights = tuple(float(value) for value in raw_weights)
        if len(weights) != len(factors):
            raise ValueError("weights must contain exactly one value per factor")
        if any(not math.isfinite(value) or value <= 0 for value in weights):
            raise ValueError("weights must be finite positive numbers")
        object.__setattr__(self, "weights", weights)

        ascending = tuple(_canonical_factor_name(value) for value in self.ascending_factors)
        if len(set(ascending)) != len(ascending):
            raise ValueError("ascending_factors must be unique")
        unknown_ascending = set(ascending).difference(factors)
        if unknown_ascending:
            raise ValueError(
                "ascending_factors must be included in factors: "
                + ", ".join(sorted(unknown_ascending))
            )
        object.__setattr__(self, "ascending_factors", ascending)

        if not 0.0 < self.min_factor_coverage <= 1.0:
            raise ValueError("min_factor_coverage must be in (0, 1]")
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
    def normalized_weights(self) -> tuple[float, ...]:
        total = sum(self.weights or ())
        return tuple(value / total for value in (self.weights or ()))

    @property
    def resolved_rank_end(self) -> int:
        return self.rank_end if self.rank_end is not None else self.rank_start + self.top_n - 1

    @property
    def selected_rank_count(self) -> int:
        return self.resolved_rank_end - self.rank_start + 1


@dataclass(frozen=True)
class RankEnsembleResult:
    signals: pl.DataFrame
    selections: pd.DataFrame
    daily_summary: pd.DataFrame
    filter_status: pd.DataFrame


def build_alpha101_rank_ensemble_frame(
    panels: Alpha101Panels,
    *,
    config: RankEnsembleConfig,
    dates: Sequence[pd.Timestamp | str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate lagged Alpha101 components and point-in-time eligibility fields."""

    for factor in config.factors:
        normalize_alpha_name(factor)

    calculator = Alpha101(panels)
    close = panels.close.sort_index()
    evaluation_dates = _resolve_dates(close.index, dates)
    lag = config.factor_lag_days

    reference_close = close.shift(lag).loc[evaluation_dates]
    listing_age = close.notna().cumsum().shift(lag).loc[evaluation_dates]
    turnover = (
        panels.turnover_value.reindex(index=close.index, columns=close.columns)
        if panels.turnover_value is not None
        else close * panels.volume.reindex(index=close.index, columns=close.columns)
    )
    average_turnover = (
        turnover.rolling(config.liquidity_lookback_days, min_periods=1)
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
    eligible &= listing_age >= config.min_listing_days
    if config.min_liquidity > 0:
        eligible &= average_turnover >= config.min_liquidity
    if is_st_available:
        eligible &= ~is_st.fillna(False).astype(bool)

    index = pd.MultiIndex.from_product(
        [evaluation_dates, close.columns],
        names=["date", "symbol"],
    )
    base = pd.DataFrame(index=index)
    for panel, column in (
        (reference_close, "reference_close"),
        (listing_age, "listing_age_days"),
        (average_turnover, "avg_turnover_lagged"),
        (is_st, "is_st"),
        (eligible, "eligible"),
    ):
        base[column] = _to_long(panel, column).reindex(index)

    frames: list[pd.DataFrame] = []
    for factor in config.factors:
        values = (
            calculator.calculate(factor)
            .reindex(index=close.index, columns=close.columns)
            .shift(lag)
            .loc[evaluation_dates]
        )
        component = base.copy()
        component["factor"] = factor
        component["factor_value"] = _to_long(values, "factor_value").reindex(index)
        frames.append(component.reset_index())

    frame = pd.concat(frames, ignore_index=True)
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["eligible"] = frame["eligible"].fillna(False).astype(bool)
    status = pd.DataFrame(
        [
            {
                "filter": "listing_age",
                "status": "applied",
                "detail": f"minimum {config.min_listing_days} prior trading observations",
            },
            {
                "filter": "liquidity",
                "status": "applied" if config.min_liquidity > 0 else "not_requested",
                "detail": (
                    f"{config.liquidity_lookback_days}-day lagged average turnover >= "
                    f"{config.min_liquidity}"
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


def rank_factor_ensemble(
    factor_frame: pd.DataFrame,
    *,
    config: RankEnsembleConfig,
    filter_status: pd.DataFrame | None = None,
) -> RankEnsembleResult:
    """Combine daily factor percentiles, then select an inclusive rank interval."""

    required = {"date", "symbol", "factor", "factor_value"}
    missing = required.difference(factor_frame.columns)
    if missing:
        raise ValueError(f"missing ensemble columns: {', '.join(sorted(missing))}")

    frame = factor_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["factor"] = frame["factor"].map(_canonical_factor_name)
    frame["factor_value"] = pd.to_numeric(frame["factor_value"], errors="coerce")
    frame = frame.loc[frame["factor"].isin(config.factors)].copy()
    if frame.duplicated(["date", "symbol", "factor"]).any():
        raise ValueError("factor_frame contains duplicate date/symbol/factor rows")
    if "eligible" not in frame.columns:
        frame["eligible"] = True
    else:
        frame["eligible"] = frame["eligible"].fillna(False).astype(bool)

    normalized_weights = pd.Series(
        config.normalized_weights,
        index=config.factors,
        dtype=float,
    )
    ascending = set(config.ascending_factors)
    signal_rows: list[Signal] = []
    selection_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []

    for date, daily in frame.groupby("date", sort=True):
        values = daily.pivot(index="symbol", columns="factor", values="factor_value").reindex(
            columns=config.factors
        )
        eligibility = daily.groupby("symbol", sort=False)["eligible"].all().reindex(values.index)
        eligible_symbols = eligibility.loc[eligibility].index
        percentiles = pd.DataFrame(index=values.index, columns=config.factors, dtype=float)
        for factor in config.factors:
            factor_values = values.loc[eligible_symbols, factor]
            percentiles.loc[eligible_symbols, factor] = factor_values.rank(
                method="average",
                pct=True,
                ascending=factor not in ascending,
            )

        available = percentiles.notna()
        available_weight = available.mul(normalized_weights, axis=1).sum(axis=1)
        numerator = percentiles.mul(normalized_weights, axis=1).sum(axis=1, min_count=1)
        ensemble_score = numerator.div(available_weight.where(available_weight > 0))
        factor_coverage = available_weight
        available_count = available.sum(axis=1)
        scored_mask = (
            eligibility
            & ensemble_score.notna()
            & factor_coverage.ge(config.min_factor_coverage - 1e-12)
        )
        scored = pd.DataFrame(
            {
                "ensemble_score": ensemble_score.loc[scored_mask],
                "factor_coverage": factor_coverage.loc[scored_mask],
                "available_factor_count": available_count.loc[scored_mask].astype(int),
            }
        )
        scored.index.name = "symbol"
        summary: dict[str, object] = {
            "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
            "raw_symbol_count": len(values),
            "eligible_count": int(eligibility.sum()),
            "scored_count": len(scored),
            "selected_count": 0,
            "factor_count": len(config.factors),
            "min_factor_coverage": config.min_factor_coverage,
            "rank_start": config.rank_start,
            "rank_end": config.resolved_rank_end,
            "status": "success",
            "message": "",
        }
        if len(scored) < config.min_universe:
            summary["status"] = "skipped"
            summary["message"] = (
                f"scored universe {len(scored)} is below min_universe {config.min_universe}"
            )
            summaries.append(summary)
            continue

        ranked = scored.reset_index().sort_values(
            ["ensemble_score", "symbol"],
            ascending=[False, True],
            kind="mergesort",
        )
        ranked["rank_position"] = np.arange(1, len(ranked) + 1)
        selected = ranked.iloc[config.rank_start - 1 : config.resolved_rank_end].copy()
        if selected.empty:
            summary["status"] = "skipped"
            summary["message"] = (
                f"scored universe {len(scored)} does not reach rank {config.rank_start}"
            )
            summaries.append(summary)
            continue

        selected["weight"] = 1.0 / len(selected)
        selected.insert(0, "date", pd.Timestamp(date).strftime("%Y-%m-%d"))
        selected["factor_lag_days"] = config.factor_lag_days
        selected["factor_values"] = selected["symbol"].map(
            lambda symbol: _finite_mapping(values.loc[symbol])
        )
        selected["factor_percentiles"] = selected["symbol"].map(
            lambda symbol: _finite_mapping(percentiles.loc[symbol])
        )
        summary["selected_count"] = len(selected)
        summaries.append(summary)
        selection_frames.append(selected)

        weight_map = dict(zip(config.factors, config.normalized_weights))
        for _, row in selected.iterrows():
            metadata = {
                "factors": list(config.factors),
                "factor_weights": weight_map,
                "ascending_factors": list(config.ascending_factors),
                "factor_values": row["factor_values"],
                "factor_percentiles": row["factor_percentiles"],
                "factor_coverage": float(row["factor_coverage"]),
                "available_factor_count": int(row["available_factor_count"]),
                "ensemble_score": float(row["ensemble_score"]),
                "rank_position": int(row["rank_position"]),
                "eligible_count": int(eligibility.sum()),
                "factor_lag_days": config.factor_lag_days,
            }
            signal_rows.append(
                Signal(
                    date=pd.Timestamp(date).strftime("%Y-%m-%d"),
                    symbol=str(row["symbol"]).zfill(6),
                    signal_type="buy",
                    source=config.source,
                    score=float(row["ensemble_score"]),
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
                "ensemble_score",
                "factor_coverage",
                "available_factor_count",
                "rank_position",
                "weight",
                "factor_lag_days",
                "factor_values",
                "factor_percentiles",
            ]
        )
    )
    return RankEnsembleResult(
        signals=signals_to_frame(signal_rows),
        selections=selections,
        daily_summary=pd.DataFrame(summaries),
        filter_status=filter_status.copy() if filter_status is not None else pd.DataFrame(),
    )


def write_rank_ensemble_reports(
    result: RankEnsembleResult,
    output_dir: str | Path,
    *,
    config: RankEnsembleConfig,
) -> Path:
    """Write unified signals and component-level ensemble diagnostics."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    signals_csv = metadata_to_json(result.signals)
    selections_csv = result.selections.copy()
    for column in ("factor_values", "factor_percentiles"):
        if column in selections_csv.columns:
            selections_csv[column] = selections_csv[column].map(
                lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, dict)
                else value
            )
    _atomic_write_csv(destination / "signals.csv", signals_csv)
    _atomic_write_csv(destination / "selections.csv", selections_csv)
    _atomic_write_csv(destination / "daily_summary.csv", result.daily_summary)
    _atomic_write_csv(destination / "filter_status.csv", result.filter_status)
    _atomic_write_json(
        destination / "signals.json",
        {"signals": result.signals.to_dicts()},
    )
    manifest = {
        "strategy": "rank_ensemble",
        "settings": asdict(config),
        "normalized_weights": dict(zip(config.factors, config.normalized_weights)),
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


def _canonical_factor_name(value: object) -> str:
    name = str(value).strip().lower().replace("-", "_")
    if not name:
        raise ValueError("factor names must not be blank")
    if name.startswith("alpha") or name.isdigit():
        try:
            return normalize_alpha_name(name)
        except KeyError:
            pass
    return name


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


def _finite_mapping(values: pd.Series) -> dict[str, float]:
    output: dict[str, float] = {}
    for key, value in values.items():
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.notna(numeric) and np.isfinite(float(numeric)):
            output[str(key)] = float(numeric)
    return output


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
