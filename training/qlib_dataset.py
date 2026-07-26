"""Build in-memory Qlib datasets from RQuant's aligned walk-forward rows."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QlibDatasetBundle:
    """Qlib dataset plus the exact point-in-time boundaries used to build it."""

    dataset: Any
    train_index: pd.MultiIndex
    valid_index: pd.MultiIndex
    test_index: pd.MultiIndex
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def audit(self) -> dict[str, object]:
        return {
            "index_names": list(self.test_index.names),
            "train_start": self.train_start.strftime("%Y-%m-%d"),
            "train_end": self.train_end.strftime("%Y-%m-%d"),
            "valid_start": self.valid_start.strftime("%Y-%m-%d"),
            "valid_end": self.valid_end.strftime("%Y-%m-%d"),
            "test_start": self.test_start.strftime("%Y-%m-%d"),
            "test_end": self.test_end.strftime("%Y-%m-%d"),
            "train_rows": len(self.train_index),
            "valid_rows": len(self.valid_index),
            "test_rows": len(self.test_index),
        }


def build_qlib_dataset(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    date_col: str = "date",
    symbol_col: str = "symbol",
    valid_ratio: float = 0.2,
) -> QlibDatasetBundle:
    """Convert one RQuant window to Qlib's datetime/instrument DatasetH contract.

    The validation segment is the chronologically last part of the existing
    training window.  The test segment remains wholly out of sample and must
    begin strictly after validation.  Purge dates are excluded by the caller's
    walk-forward window before this adapter is invoked.
    """

    if not 0 < float(valid_ratio) < 1:
        raise ValueError("valid_ratio must be in (0, 1)")
    features = tuple(str(value) for value in feature_cols)
    if not features:
        raise ValueError("feature_cols must not be empty")
    required = {date_col, symbol_col, target_col, *features}
    for name, frame in (("train", train), ("test", test)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} frame missing Qlib columns: {', '.join(missing)}")

    prepared_train = _prepare_frame(
        train,
        feature_cols=features,
        target_col=target_col,
        date_col=date_col,
        symbol_col=symbol_col,
    )
    prepared_test = _prepare_frame(
        test,
        feature_cols=features,
        target_col=target_col,
        date_col=date_col,
        symbol_col=symbol_col,
    )
    train_dates = pd.DatetimeIndex(prepared_train[date_col].drop_duplicates().sort_values())
    if len(train_dates) < 2:
        raise ValueError("Qlib training requires at least two dates for train/valid split")
    valid_date_count = max(1, int(math.ceil(len(train_dates) * float(valid_ratio))))
    valid_date_count = min(valid_date_count, len(train_dates) - 1)
    valid_start = train_dates[-valid_date_count]
    learn = prepared_train.loc[prepared_train[date_col] < valid_start].copy()
    valid = prepared_train.loc[prepared_train[date_col] >= valid_start].copy()
    if learn.empty or valid.empty or prepared_test.empty:
        raise ValueError("Qlib train, valid, and test segments must all contain rows")

    train_end = pd.Timestamp(learn[date_col].max())
    valid_start = pd.Timestamp(valid[date_col].min())
    valid_end = pd.Timestamp(valid[date_col].max())
    test_start = pd.Timestamp(prepared_test[date_col].min())
    if not train_end < valid_start:
        raise ValueError("Qlib train and valid segments overlap")
    if not valid_end < test_start:
        raise ValueError("Qlib validation must end before the out-of-sample test segment")

    combined = pd.concat([learn, valid, prepared_test], ignore_index=True)
    qlib_frame = _to_qlib_frame(
        combined,
        feature_cols=features,
        target_col=target_col,
        date_col=date_col,
        symbol_col=symbol_col,
    )
    try:
        from qlib.data.dataset import DatasetH
        from qlib.data.dataset.handler import DataHandlerLP
    except (ImportError, OSError) as exc:
        raise ImportError(
            "Qlib models require pyqlib; install requirements-qlib.txt"
        ) from exc

    handler = DataHandlerLP.from_df(qlib_frame)
    segments = {
        "train": (_date_text(learn[date_col].min()), _date_text(learn[date_col].max())),
        "valid": (_date_text(valid[date_col].min()), _date_text(valid[date_col].max())),
        "test": (
            _date_text(prepared_test[date_col].min()),
            _date_text(prepared_test[date_col].max()),
        ),
    }
    dataset = DatasetH(handler=handler, segments=segments)
    return QlibDatasetBundle(
        dataset=dataset,
        train_index=_frame_index(learn, date_col=date_col, symbol_col=symbol_col),
        valid_index=_frame_index(valid, date_col=date_col, symbol_col=symbol_col),
        test_index=_frame_index(prepared_test, date_col=date_col, symbol_col=symbol_col),
        train_start=pd.Timestamp(learn[date_col].min()),
        train_end=train_end,
        valid_start=valid_start,
        valid_end=valid_end,
        test_start=test_start,
        test_end=pd.Timestamp(prepared_test[date_col].max()),
    )


def normalize_qlib_scores(
    values: pd.Series,
    *,
    expected_index: pd.MultiIndex,
) -> pd.Series:
    """Return finite Qlib predictions in the exact RQuant test-row order."""

    if not isinstance(values, pd.Series):
        values = pd.Series(values)
    if values.index.duplicated().any():
        raise ValueError("Qlib prediction index contains duplicates")
    scores = pd.to_numeric(values, errors="coerce").reindex(expected_index)
    if scores.isna().any() or not np.isfinite(scores.to_numpy(dtype=float)).all():
        raise ValueError("Qlib predictions do not cover the complete test index")
    scores.name = "score"
    return scores


def _prepare_frame(
    frame: pd.DataFrame,
    *,
    feature_cols: tuple[str, ...],
    target_col: str,
    date_col: str,
    symbol_col: str,
) -> pd.DataFrame:
    result = frame[[date_col, symbol_col, *feature_cols, target_col]].copy()
    result[date_col] = pd.to_datetime(result[date_col], errors="raise").dt.normalize()
    result[symbol_col] = result[symbol_col].astype(str).str.zfill(6)
    if result.duplicated([date_col, symbol_col]).any():
        raise ValueError("duplicate date/symbol rows are not allowed in a Qlib dataset")
    numeric_cols = [*feature_cols, target_col]
    result[numeric_cols] = result[numeric_cols].apply(pd.to_numeric, errors="coerce")
    numeric = result[numeric_cols].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Qlib feature and label values must be finite")
    return result.sort_values([date_col, symbol_col], kind="mergesort").reset_index(drop=True)


def _to_qlib_frame(
    frame: pd.DataFrame,
    *,
    feature_cols: tuple[str, ...],
    target_col: str,
    date_col: str,
    symbol_col: str,
) -> pd.DataFrame:
    index = _frame_index(frame, date_col=date_col, symbol_col=symbol_col)
    feature_values = frame.loc[:, list(feature_cols)].copy()
    feature_values.index = index
    label_values = frame.loc[:, [target_col]].copy()
    label_values.index = index
    result = pd.concat({"feature": feature_values, "label": label_values}, axis=1)
    return result.sort_index()


def _frame_index(
    frame: pd.DataFrame,
    *,
    date_col: str,
    symbol_col: str,
) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays(
        [
            pd.to_datetime(frame[date_col]).to_numpy(),
            frame[symbol_col].astype(str).str.zfill(6).to_numpy(),
        ],
        names=["datetime", "instrument"],
    )


def _date_text(value: object) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")
