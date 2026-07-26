"""Shared Polars table boundary helpers.

Business modules use :class:`polars.DataFrame` as their long-format table
contract.  ``to_polars`` deliberately accepts legacy dataframe-like inputs at
the boundary so callers can migrate independently without importing pandas in
the new implementation.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import polars as pl


DataFrame = pl.DataFrame


def to_polars(frame: Any, *, columns: list[str] | None = None) -> pl.DataFrame:
    """Return a Polars frame without requiring pandas as an adapter dependency."""

    if frame is None:
        return pl.DataFrame({name: [] for name in (columns or [])})
    if isinstance(frame, pl.LazyFrame):
        result = frame.collect()
    elif isinstance(frame, pl.DataFrame):
        result = frame.clone()
    elif isinstance(frame, Mapping):
        result = pl.DataFrame(frame, strict=False)
    elif hasattr(frame, "to_dict"):
        try:
            data = frame.to_dict(orient="list")
        except TypeError:
            data = frame.to_dict()
        result = pl.DataFrame(data, strict=False)
    else:
        result = pl.DataFrame(frame, strict=False)
    if columns is not None:
        missing = [name for name in columns if name not in result.columns]
        if missing:
            result = result.with_columns(
                [pl.lit(None).alias(name) for name in missing]
            )
        result = result.select(columns)
    return result


def metadata_to_json(frame: pl.DataFrame, column: str = "metadata") -> pl.DataFrame:
    """Convert an object/struct metadata column to stable JSON for CSV output."""

    if column not in frame.columns:
        return frame
    return frame.with_columns(
        pl.col(column)
        .map_elements(
            lambda value: json.dumps(value or {}, ensure_ascii=False, sort_keys=True),
            return_dtype=pl.String,
            skip_nulls=False,
        )
        .alias(column)
    )


def atomic_write_csv(path: str | Path, frame: Any) -> None:
    """Atomically write a Polars-compatible dataframe to CSV."""

    destination = Path(path)
    temp = destination.with_name(f".{destination.name}.tmp")
    to_polars(frame).write_csv(temp)
    os.replace(temp, destination)
