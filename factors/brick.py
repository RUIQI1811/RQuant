"""BrickChart-derived factors for the standalone factor research track.

The existing ``BrickChartSelector`` remains the source of truth for custom
strategy picks.  This module only adapts its causal features into
``FactorTester``'s long format; it does not replace or change strategy
execution.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

import numpy as np
import pandas as pd

from strategies.selector import BrickChartSelector, BrickComputeParams


BRICK_FACTOR_NAMES = ("brick", "brick_chart", "brick_growth")
LISTED_BRICK_FACTORS = ("brick", "brick_growth")


def normalize_brick_factor_name(factor_name: str) -> str:
    """Normalize supported BrickChart factor names.

    ``brick``/``brick_chart`` test the complete configured strategy gate.
    ``brick_growth`` tests the dense continuous brick-strength feature.
    """
    name = str(factor_name).strip().lower().replace("-", "_")
    if name == "brick_chart":
        return "brick"
    if name not in LISTED_BRICK_FACTORS:
        raise KeyError(f"unknown BrickChart factor: {factor_name}")
    return name


def is_brick_factor(factor_name: str) -> bool:
    """Return whether ``factor_name`` is handled by this module."""
    try:
        normalize_brick_factor_name(factor_name)
    except KeyError:
        return False
    return True


def _symbol(value: object) -> str:
    match = re.search(r"(\d{6})", str(value))
    return match.group(1) if match else str(value).zfill(6)


def _brick_compute_params(config: Mapping[str, object]) -> BrickComputeParams:
    return BrickComputeParams(
        n=int(config.get("n", 4)),
        m1=int(config.get("m1", 4)),
        m2=int(config.get("m2", 6)),
        m3=int(config.get("m3", 6)),
        t=float(config.get("t", 4.0)),
        shift1=float(config.get("shift1", 90.0)),
        shift2=float(config.get("shift2", 100.0)),
        sma_w1=int(config.get("sma_w1", 1)),
        sma_w2=int(config.get("sma_w2", 1)),
        sma_w3=int(config.get("sma_w3", 1)),
    )


def _brick_selector(config: Mapping[str, object]) -> BrickChartSelector:
    zxdq_ratio = config.get("zxdq_ratio")
    return BrickChartSelector(
        daily_return_threshold=float(config.get("daily_return_threshold", 0.05)),
        brick_growth_ratio=float(config.get("brick_growth_ratio", 1.0)),
        min_prior_green_bars=int(config.get("min_prior_green_bars", 2)),
        zxdq_ratio=None if zxdq_ratio is None else float(zxdq_ratio),
        zxdq_span=int(config.get("zxdq_span", 10)),
        require_zxdq_gt_zxdkx=bool(config.get("require_zxdq_gt_zxdkx", True)),
        zxdkx_m1=int(config.get("zxdkx_m1", 14)),
        zxdkx_m2=int(config.get("zxdkx_m2", 28)),
        zxdkx_m3=int(config.get("zxdkx_m3", 57)),
        zxdkx_m4=int(config.get("zxdkx_m4", 114)),
        require_weekly_ma_bull=bool(config.get("require_weekly_ma_bull", True)),
        wma_short=int(config.get("wma_short", 20)),
        wma_mid=int(config.get("wma_mid", 60)),
        wma_long=int(config.get("wma_long", 120)),
        n=int(config.get("n", 4)),
        m1=int(config.get("m1", 4)),
        m2=int(config.get("m2", 6)),
        m3=int(config.get("m3", 6)),
        t=float(config.get("t", 4.0)),
        shift1=float(config.get("shift1", 90.0)),
        shift2=float(config.get("shift2", 100.0)),
        sma_w1=int(config.get("sma_w1", 1)),
        sma_w2=int(config.get("sma_w2", 1)),
        sma_w3=int(config.get("sma_w3", 1)),
    )


def _growth_from_brick(brick: np.ndarray) -> np.ndarray:
    previous = np.empty_like(brick, dtype=float)
    previous[0] = np.nan
    previous[1:] = brick[:-1]
    denominator = np.abs(previous)
    growth = np.divide(
        brick,
        denominator,
        out=brick.astype(float, copy=True),
        where=denominator > 0,
    )
    growth[~np.isfinite(growth)] = np.nan
    return growth


def _merge_metadata(
    result: pd.DataFrame,
    metadata: pd.DataFrame | None,
) -> pd.DataFrame:
    if metadata is None or metadata.empty or result.empty:
        return result
    meta = metadata.copy()
    meta.columns = [str(column).lower() for column in meta.columns]
    meta_symbol = "symbol" if "symbol" in meta.columns else "ts_code"
    if meta_symbol not in meta.columns:
        return result
    meta["symbol"] = meta[meta_symbol].map(_symbol)
    available = [
        column for column in ("industry", "sector", "subindustry") if column in meta.columns
    ]
    if not available:
        return result
    return result.merge(
        meta[["symbol", *available]].drop_duplicates("symbol"),
        on="symbol",
        how="left",
    )


def brick_factor_to_long(
    raw_data: Mapping[str, pd.DataFrame],
    factor_name: str,
    *,
    config: Mapping[str, object] | None = None,
    metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Calculate a BrickChart-derived factor in FactorTester's long schema.

    For ``brick``, ``factor_value`` is ``brick_growth`` only where the full
    configured selector emits a pick; all other rows are ``NaN``.  For
    ``brick_growth``, every finite continuous growth value is retained.

    The config's ``enabled`` switch is intentionally ignored: explicitly
    running a research factor must work even when live preselection is off.
    """
    name = normalize_brick_factor_name(factor_name)
    cfg = dict(config or {})
    selector = _brick_selector(cfg) if name == "brick" else None
    compute_params = _brick_compute_params(cfg)
    rows: list[pd.DataFrame] = []

    for raw_symbol, raw_frame in raw_data.items():
        if raw_frame is None or raw_frame.empty:
            continue
        frame = raw_frame.copy()
        frame.columns = [str(column).lower() for column in frame.columns]
        required = {"date", "high", "low", "close"}
        if not required.issubset(frame.columns):
            continue
        frame["date"] = pd.to_datetime(frame["date"])
        for column in ("high", "low", "close"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = (
            frame.sort_values("date")
            .drop_duplicates("date", keep="last")
            .set_index("date", drop=False)
        )

        if selector is not None:
            prepared = selector.prepare_df(frame)
            growth = pd.to_numeric(prepared["brick_growth"], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            signal = prepared["_vec_pick"].fillna(False).astype(bool)
            factor_value = growth.where(signal)
        else:
            prepared = frame.copy()
            brick = compute_params.compute_arr(prepared)
            growth = pd.Series(_growth_from_brick(brick), index=prepared.index)
            signal = pd.Series(False, index=prepared.index, dtype=bool)
            factor_value = growth

        prepared["symbol"] = _symbol(raw_symbol)
        prepared["factor_value"] = factor_value.to_numpy()
        prepared["brick_growth"] = growth.to_numpy()
        prepared["brick_signal"] = signal.to_numpy()
        keep = ["date", "symbol", "factor_value", "close", "brick_growth", "brick_signal"]
        keep.extend(
            column
            for column in (
                "volume",
                "amount",
                "turnover_value",
                "daily_return",
                "is_tradeable",
                "is_suspended",
                "is_limit_up",
                "is_limit_down",
                "is_st",
                "listing_age_days",
                "market_cap",
                "total_mv",
                "cap",
            )
            if column in prepared.columns and column not in keep
        )
        rows.append(prepared[keep].reset_index(drop=True))

    columns = ["date", "symbol", "factor_value", "close", "brick_growth", "brick_signal"]
    if not rows:
        return pd.DataFrame(columns=columns)
    result = pd.concat(rows, ignore_index=True)
    result = _merge_metadata(result, metadata)
    if "market_cap" not in result.columns:
        for candidate in ("total_mv", "cap"):
            if candidate in result.columns:
                result["market_cap"] = pd.to_numeric(result[candidate], errors="coerce")
                break
    return result.sort_values(["date", "symbol"]).reset_index(drop=True)


__all__ = [
    "BRICK_FACTOR_NAMES",
    "LISTED_BRICK_FACTORS",
    "brick_factor_to_long",
    "is_brick_factor",
    "normalize_brick_factor_name",
]
