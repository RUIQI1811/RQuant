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


def build_calendar_year_walk_forward_windows(
    dates: Iterable[pd.Timestamp],
    *,
    train_years: int = 3,
    test_years: int = 1,
    purge_size: int = 0,
) -> list[WalkForwardWindow]:
    """Build strict prior-calendar-years training and next-year OOS windows.

    For a 2024 test window with ``train_years=3``, only observations from
    calendar years 2021-2023 are eligible for training. The final
    ``purge_size`` trading dates are removed from that training slice so a
    forward-return label cannot overlap the first test observation.
    """
    ordered = sorted(pd.Timestamp(date) for date in set(dates))
    if train_years <= 0 or test_years <= 0:
        raise ValueError("train_years and test_years must be positive")
    if purge_size < 0:
        raise ValueError("purge_size must be non-negative")
    if not ordered:
        return []

    available_years = {date.year for date in ordered}
    first_year = min(available_years)
    last_year = max(available_years)
    windows: list[WalkForwardWindow] = []
    test_start_year = first_year + train_years
    while test_start_year + test_years - 1 <= last_year:
        required_train_years = set(
            range(test_start_year - train_years, test_start_year)
        )
        required_test_years = set(
            range(test_start_year, test_start_year + test_years)
        )
        if required_train_years.issubset(available_years) and required_test_years.issubset(
            available_years
        ):
            train_all = [
                date
                for date in ordered
                if date.year in required_train_years
            ]
            test = [date for date in ordered if date.year in required_test_years]
            if len(train_all) > purge_size and test:
                train = train_all[:-purge_size] if purge_size else train_all
                purge = train_all[-purge_size:] if purge_size else []
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
        test_start_year += test_years
    return windows
