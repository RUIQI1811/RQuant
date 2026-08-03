"""Validated adapters for user-supplied factor panels.

The canonical external format is one row per ``date``/``symbol`` with one
numeric column per factor.  A long ``factor``/``factor_value`` layout is also
accepted and is pivoted to the canonical wide representation.  Values remain
unlagged here: research and ML consumers apply their own mandatory one-day
lag so the source file has one unambiguous time meaning.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_CONTEXT_COLUMNS = frozenset(
    {
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "up_limit",
        "down_limit",
        "turnover_value",
        "daily_return",
        "industry",
        "sector",
        "subindustry",
        "market_cap",
        "circulating_market_cap",
        "turnover_rate",
        "volume_ratio",
        "pb",
        "book_to_market",
        "total_mv",
        "cap",
        "market_regime",
        "is_tradeable",
        "is_suspended",
        "is_limit_up",
        "is_limit_down",
        "is_st",
        "listing_age_days",
    }
)


@dataclass(frozen=True)
class ExternalFactorFrame:
    """Canonical external factor values plus their selected factor columns."""

    frame: pd.DataFrame
    factors: tuple[str, ...]
    source_path: Path
    source_layout: str


def load_research_context_file(
    path: str | Path,
    *,
    date_col: str = "date",
    symbol_col: str = "symbol",
) -> pd.DataFrame:
    """Load point-in-time cap/classification/regime fields by date and symbol."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"research context file not found: {source}")
    frame = _read_context_source(source, symbol_col=symbol_col)
    missing = {date_col, symbol_col}.difference(frame.columns)
    if missing:
        raise ValueError(
            "research context file missing key columns: "
            + ", ".join(sorted(missing))
        )
    context_renames = {
        column: str(column).strip().lower()
        for column in frame.columns
        if column not in {date_col, symbol_col}
    }
    if len(set(context_renames.values())) != len(context_renames):
        raise ValueError("research context file has duplicate normalized column names")
    frame = frame.rename(columns=context_renames)
    available = [column for column in frame.columns if column in DEFAULT_CONTEXT_COLUMNS]
    if not available:
        raise ValueError("research context file contains no supported context columns")
    result = frame[[date_col, symbol_col, *available]].copy()
    result[date_col] = pd.to_datetime(result[date_col], errors="coerce")
    if result[date_col].isna().any():
        raise ValueError("research context file contains invalid dates")
    result[symbol_col] = _normalize_symbols(result[symbol_col])
    if result[symbol_col].isna().any():
        raise ValueError("research context file contains invalid symbols")
    if result.duplicated([date_col, symbol_col]).any():
        raise ValueError("research context file contains duplicate date/symbol rows")
    numeric_columns = {
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "up_limit",
        "down_limit",
        "turnover_value",
        "daily_return",
        "market_cap",
        "circulating_market_cap",
        "turnover_rate",
        "volume_ratio",
        "pb",
        "book_to_market",
        "total_mv",
        "cap",
        "listing_age_days",
    }
    for column in numeric_columns.intersection(available):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if "market_cap" not in result.columns:
        for alias in ("total_mv", "cap"):
            if alias in result.columns:
                result["market_cap"] = result[alias]
                break
    return (
        result.rename(columns={date_col: "date", symbol_col: "symbol"})
        .sort_values(["date", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )


def research_context_signature(path: str | Path) -> str:
    """Hash a single context CSV or all persisted partitions in a directory."""

    source = Path(path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"research context path not found: {source}")
    paths = [source] if source.is_file() else sorted(source.rglob("*.csv"))
    manifest = source / "_context_manifest.json" if source.is_dir() else None
    if manifest is not None and manifest.exists():
        paths.append(manifest)
    if not paths:
        raise ValueError(f"research context directory contains no partitions: {source}")
    digest = hashlib.sha256()
    for item in paths:
        stat = item.stat()
        digest.update(str(item.relative_to(source) if source.is_dir() else item).encode())
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(str(getattr(stat, "st_flags", 0)).encode("ascii"))
    return digest.hexdigest()


def _read_context_source(source: Path, *, symbol_col: str) -> pd.DataFrame:
    if source.is_file():
        return pd.read_csv(source, dtype={symbol_col: str})
    if not source.is_dir():
        raise ValueError(f"research context path is neither file nor directory: {source}")
    manifest_path = source / "_context_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("research context manifest is unreadable") from exc
        if manifest.get("status") != "complete":
            raise ValueError("research context manifest is partial; resume fetch-context first")
    paths = sorted(source.rglob("*.csv"))
    if not paths:
        raise ValueError(f"research context directory contains no CSV partitions: {source}")
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError("partitioned research context requires polars") from exc
    scans = [
        pl.scan_csv(str(path), schema_overrides={symbol_col: pl.String})
        for path in paths
    ]
    collected = pl.concat(scans, how="vertical_relaxed").collect()
    return pd.DataFrame(collected.to_dict(as_series=False))


def merge_context_with_raw_data(
    raw_data: Mapping[str, pd.DataFrame],
    context: pd.DataFrame | None,
) -> dict[str, pd.DataFrame]:
    """Merge explicit point-in-time context into per-symbol market frames."""

    if context is None:
        return {str(symbol).zfill(6): frame for symbol, frame in raw_data.items()}
    required = {"date", "symbol"}
    if not required.issubset(context.columns):
        raise ValueError("research context must contain date and symbol")
    by_symbol = {
        str(symbol).zfill(6): part.drop(columns="symbol").copy()
        for symbol, part in context.groupby("symbol", sort=False)
    }
    result: dict[str, pd.DataFrame] = {}
    for raw_symbol, raw_frame in raw_data.items():
        symbol = str(raw_symbol).zfill(6)
        frame = raw_frame.copy()
        frame.columns = [str(column).strip().lower() for column in frame.columns]
        if "date" not in frame.columns:
            result[symbol] = frame
            continue
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        supplied = by_symbol.get(symbol)
        if supplied is None:
            result[symbol] = frame
            continue
        merged = frame.merge(
            supplied,
            on="date",
            how="left",
            suffixes=("__market", ""),
            validate="one_to_one",
        )
        for column in DEFAULT_CONTEXT_COLUMNS:
            market_column = f"{column}__market"
            if market_column not in merged.columns:
                continue
            if column in merged.columns:
                merged[column] = merged[column].combine_first(merged[market_column])
            else:
                merged[column] = merged[market_column]
            merged = merged.drop(columns=market_column)
        result[symbol] = merged
    return result


def merge_context_with_research_frame(
    frame: pd.DataFrame,
    context: pd.DataFrame | None,
    *,
    date_col: str = "date",
    symbol_col: str = "symbol",
) -> pd.DataFrame:
    """Attach canonical context to an existing long research frame."""

    if context is None:
        return frame.copy()
    missing = {date_col, symbol_col}.difference(frame.columns)
    if missing:
        raise ValueError(
            "research frame missing context merge keys: " + ", ".join(sorted(missing))
        )
    base = frame.copy()
    base[date_col] = pd.to_datetime(base[date_col], errors="coerce")
    base[symbol_col] = _normalize_symbols(base[symbol_col])
    supplied = context.rename(columns={"date": date_col, "symbol": symbol_col})
    merged = base.merge(
        supplied,
        on=[date_col, symbol_col],
        how="left",
        suffixes=("__base", ""),
        validate="many_to_one",
    )
    for column in DEFAULT_CONTEXT_COLUMNS:
        base_column = f"{column}__base"
        if base_column not in merged.columns:
            continue
        if column in merged.columns:
            merged[column] = merged[column].combine_first(merged[base_column])
        else:
            merged[column] = merged[base_column]
        merged = merged.drop(columns=base_column)
    return merged


def normalize_external_factor_name(value: object) -> str:
    """Return a stable external column name without imposing a factor family."""

    name = str(value).strip()
    if not name:
        raise ValueError("external factor names must not be empty")
    if name.lower() in {"date", "symbol", *DEFAULT_CONTEXT_COLUMNS}:
        raise ValueError(f"reserved external factor name: {name}")
    return name


def load_external_factor_file(
    path: str | Path,
    *,
    factors: Sequence[str] | None = None,
    date_col: str = "date",
    symbol_col: str = "symbol",
    layout: str = "auto",
    factor_name_col: str = "factor",
    factor_value_col: str = "factor_value",
) -> ExternalFactorFrame:
    """Load a wide or long external factor CSV and validate its primary key."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"external factor file not found: {source}")
    frame = pd.read_csv(source, dtype={symbol_col: str})
    requested_layout = str(layout).strip().lower()
    if requested_layout not in {"auto", "wide", "long"}:
        raise ValueError("external factor layout must be auto, wide, or long")
    missing_key = {date_col, symbol_col}.difference(frame.columns)
    if missing_key:
        raise ValueError(
            "external factor file missing key columns: "
            + ", ".join(sorted(missing_key))
        )

    resolved_layout = requested_layout
    if resolved_layout == "auto":
        resolved_layout = (
            "long"
            if factor_name_col in frame.columns and factor_value_col in frame.columns
            else "wide"
        )

    frame = frame.copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    if frame[date_col].isna().any():
        raise ValueError("external factor file contains invalid dates")
    frame[symbol_col] = _normalize_symbols(frame[symbol_col])
    if frame[symbol_col].isna().any():
        raise ValueError("external factor file contains invalid symbols")

    requested = _normalize_requested_factors(factors)
    if resolved_layout == "long":
        canonical, selected = _pivot_long_factor_frame(
            frame,
            requested=requested,
            date_col=date_col,
            symbol_col=symbol_col,
            factor_name_col=factor_name_col,
            factor_value_col=factor_value_col,
        )
    else:
        canonical, selected = _select_wide_factor_frame(
            frame,
            requested=requested,
            date_col=date_col,
            symbol_col=symbol_col,
            factor_name_col=factor_name_col,
            factor_value_col=factor_value_col,
        )

    canonical = canonical.rename(columns={date_col: "date", symbol_col: "symbol"})
    canonical["date"] = pd.to_datetime(canonical["date"])
    canonical["symbol"] = _normalize_symbols(canonical["symbol"])
    canonical = canonical.sort_values(["date", "symbol"], kind="mergesort").reset_index(
        drop=True
    )
    return ExternalFactorFrame(
        frame=canonical,
        factors=selected,
        source_path=source.resolve(),
        source_layout=resolved_layout,
    )


def merge_external_with_market_data(
    external: ExternalFactorFrame,
    raw_data: Mapping[str, pd.DataFrame],
    *,
    metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach point-in-time market fields needed by ``FactorTester``.

    External values win when the source already provides a context field.
    Missing context is filled from the repository's daily per-symbol bars.
    Static metadata is used only for classification columns, never for prices
    or market capitalisation.
    """

    market = _raw_market_frame(raw_data)
    result = external.frame.merge(
        market,
        on=["date", "symbol"],
        how="left",
        suffixes=("", "__market"),
        validate="one_to_one",
    )
    for column in DEFAULT_CONTEXT_COLUMNS:
        market_column = f"{column}__market"
        if market_column not in result.columns:
            continue
        if column in result.columns:
            result[column] = result[column].combine_first(result[market_column])
        else:
            result[column] = result[market_column]
        result = result.drop(columns=market_column)

    if metadata is not None and not metadata.empty:
        result = _attach_static_classifications(result, metadata)
    return result.sort_values(["date", "symbol"], kind="mergesort").reset_index(
        drop=True
    )


def _normalize_requested_factors(
    factors: Sequence[str] | None,
) -> tuple[str, ...] | None:
    if factors is None:
        return None
    normalized = tuple(
        dict.fromkeys(
            normalize_external_factor_name(value)
            for value in factors
            if str(value).strip().lower() != "all"
        )
    )
    return normalized or None


def _pivot_long_factor_frame(
    frame: pd.DataFrame,
    *,
    requested: tuple[str, ...] | None,
    date_col: str,
    symbol_col: str,
    factor_name_col: str,
    factor_value_col: str,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    missing = {factor_name_col, factor_value_col}.difference(frame.columns)
    if missing:
        raise ValueError(
            "long external factor file missing columns: "
            + ", ".join(sorted(missing))
        )
    work = frame[[date_col, symbol_col, factor_name_col, factor_value_col]].copy()
    work[factor_name_col] = work[factor_name_col].map(normalize_external_factor_name)
    if requested is not None:
        missing_factors = set(requested).difference(work[factor_name_col].unique())
        if missing_factors:
            raise ValueError(
                "external factor file missing requested factors: "
                + ", ".join(sorted(missing_factors))
            )
        work = work.loc[work[factor_name_col].isin(requested)]
        selected = requested
    else:
        selected = tuple(dict.fromkeys(work[factor_name_col].tolist()))
    duplicate = work.duplicated([date_col, symbol_col, factor_name_col], keep=False)
    if duplicate.any():
        sample = work.loc[duplicate, [date_col, symbol_col, factor_name_col]].iloc[0]
        raise ValueError(
            "duplicate external factor observation: "
            f"{sample[date_col].date()} {sample[symbol_col]} {sample[factor_name_col]}"
        )
    work[factor_value_col] = pd.to_numeric(work[factor_value_col], errors="coerce")
    wide = work.pivot(
        index=[date_col, symbol_col],
        columns=factor_name_col,
        values=factor_value_col,
    ).reset_index()
    wide.columns.name = None
    return wide[[date_col, symbol_col, *selected]], selected


def _select_wide_factor_frame(
    frame: pd.DataFrame,
    *,
    requested: tuple[str, ...] | None,
    date_col: str,
    symbol_col: str,
    factor_name_col: str,
    factor_value_col: str,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    duplicate = frame.duplicated([date_col, symbol_col], keep=False)
    if duplicate.any():
        sample = frame.loc[duplicate, [date_col, symbol_col]].iloc[0]
        raise ValueError(
            "duplicate external factor row: "
            f"{sample[date_col].date()} {sample[symbol_col]}"
        )
    if requested is None:
        excluded = {
            date_col,
            symbol_col,
            factor_name_col,
            factor_value_col,
            *DEFAULT_CONTEXT_COLUMNS,
        }
        candidates = [str(column) for column in frame.columns if column not in excluded]
        selected = tuple(
            column
            for column in candidates
            if pd.to_numeric(frame[column], errors="coerce").notna().any()
        )
    else:
        selected = requested
    if not selected:
        raise ValueError("external factor file contains no numeric factor columns")
    missing = set(selected).difference(frame.columns)
    if missing:
        raise ValueError(
            "external factor file missing requested factors: "
            + ", ".join(sorted(missing))
        )
    context = [
        column
        for column in frame.columns
        if column in DEFAULT_CONTEXT_COLUMNS and column not in selected
    ]
    result = frame[[date_col, symbol_col, *selected, *context]].copy()
    for factor in selected:
        result[factor] = pd.to_numeric(result[factor], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    return result, selected


def _normalize_symbols(values: pd.Series) -> pd.Series:
    extracted = values.astype(str).str.extract(r"(\d{1,6})", expand=False)
    return extracted.str.zfill(6)


def _raw_market_frame(raw_data: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for raw_symbol, raw_frame in raw_data.items():
        if raw_frame is None or raw_frame.empty:
            continue
        frame = raw_frame.copy()
        frame.columns = [str(column).strip().lower() for column in frame.columns]
        if "date" not in frame.columns:
            continue
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["symbol"] = str(raw_symbol).zfill(6)
        columns = [
            "date",
            "symbol",
            *[
                column
                for column in DEFAULT_CONTEXT_COLUMNS
                if column in frame.columns
            ],
        ]
        rows.append(frame[columns])
    if not rows:
        raise ValueError("raw market data contains no dated rows")
    result = pd.concat(rows, ignore_index=True)
    result = result.dropna(subset=["date", "symbol"])
    duplicate = result.duplicated(["date", "symbol"], keep=False)
    if duplicate.any():
        raise ValueError("raw market data contains duplicate date/symbol rows")
    return result


def _attach_static_classifications(
    frame: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    meta = metadata.copy()
    meta.columns = [str(column).strip().lower() for column in meta.columns]
    symbol_column = "symbol" if "symbol" in meta.columns else "ts_code"
    if symbol_column not in meta.columns:
        return frame
    meta["symbol"] = _normalize_symbols(meta[symbol_column])
    classifications = [
        column for column in ("industry", "sector", "subindustry") if column in meta
    ]
    if not classifications:
        return frame
    meta = meta[["symbol", *classifications]].drop_duplicates("symbol")
    result = frame.merge(meta, on="symbol", how="left", suffixes=("", "__meta"))
    for column in classifications:
        metadata_column = f"{column}__meta"
        if metadata_column not in result.columns:
            continue
        if column in result.columns:
            result[column] = result[column].combine_first(result[metadata_column])
        else:
            result[column] = result[metadata_column]
        result = result.drop(columns=metadata_column)
    return result
