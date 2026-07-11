"""Walk-forward validation helpers for model training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    purge_start: pd.Timestamp | None = None
    purge_end: pd.Timestamp | None = None


def validate_feature_label_frame(
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    date_col: str = "date",
    symbol_col: str = "symbol",
) -> pd.DataFrame:
    required = set(feature_cols) | {target_col, date_col, symbol_col}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    result = frame.copy()
    result[symbol_col] = result[symbol_col].astype(str).str.zfill(6)
    result[date_col] = pd.to_datetime(result[date_col])
    if result.duplicated([date_col, symbol_col]).any():
        raise ValueError("duplicate date/symbol rows are not allowed")
    numeric_columns = list(feature_cols) + [target_col]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[numeric_columns].isna().any().any():
        raise ValueError("features and target must not contain missing values")
    if not np.isfinite(result[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("features and target must contain only finite values")
    return result.sort_values([date_col, symbol_col]).reset_index(drop=True)


def build_walk_forward_windows(
    dates: Iterable[pd.Timestamp],
    train_size: int,
    test_size: int,
    purge_size: int = 0,
) -> list[WalkForwardWindow]:
    ordered = sorted(pd.Timestamp(date) for date in set(dates))
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")
    if purge_size < 0:
        raise ValueError("purge_size must be non-negative")
    windows: list[WalkForwardWindow] = []
    start = 0
    while start + train_size + purge_size + test_size <= len(ordered):
        train = ordered[start : start + train_size]
        purge = ordered[start + train_size : start + train_size + purge_size]
        test_start_index = start + train_size + purge_size
        test = ordered[test_start_index : test_start_index + test_size]
        windows.append(
            WalkForwardWindow(
                train_start=train[0],
                train_end=train[-1],
                test_start=test[0],
                test_end=test[-1],
                purge_start=purge[0] if purge else None,
                purge_end=purge[-1] if purge else None,
            )
        )
        start += test_size
    return windows
