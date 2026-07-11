"""Convert model prediction scores into unified signals."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from signals.schema import Signal, signals_to_frame


def scores_to_signals(
    scores: pd.DataFrame,
    source: str,
    date_col: str = "date",
    symbol_col: str = "symbol",
    score_col: str = "score",
    top_n: int | None = None,
    top_quantile: float | None = None,
) -> pd.DataFrame:
    required = {date_col, symbol_col, score_col}
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if top_n is not None and top_n <= 0:
        raise ValueError("top_n must be positive")
    if top_quantile is not None and not 0 < top_quantile <= 1:
        raise ValueError("top_quantile must be in (0, 1]")
    frame = scores[[date_col, symbol_col, score_col]].copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame[symbol_col] = frame[symbol_col].astype(str).str.zfill(6)
    frame[score_col] = pd.to_numeric(frame[score_col], errors="coerce")
    if frame.duplicated([date_col, symbol_col]).any():
        raise ValueError("duplicate date/symbol score rows are not allowed")
    if frame[score_col].isna().any() or not np.isfinite(frame[score_col]).all():
        raise ValueError("scores must contain only finite values")

    signals: list[Signal] = []
    for date, daily in frame.groupby(date_col, sort=True):
        ranked = daily.sort_values(
            [score_col, symbol_col],
            ascending=[False, True],
            kind="mergesort",
        ).copy()
        if top_n is not None:
            selected = ranked.head(top_n).copy()
        elif top_quantile is not None:
            selected = ranked.head(max(1, math.ceil(len(ranked) * top_quantile))).copy()
        else:
            selected = ranked
        selected["rank_position"] = np.arange(1, len(selected) + 1)
        weight = 1.0 / len(selected) if len(selected) else None
        for _, row in selected.iterrows():
            score = float(row[score_col])
            signals.append(
                Signal(
                    date=pd.Timestamp(date).strftime("%Y-%m-%d"),
                    symbol=str(row[symbol_col]).zfill(6),
                    signal_type="buy",
                    source=f"model_{source}",
                    score=score,
                    weight=weight,
                    metadata={
                        "model_score": score,
                        "rank_position": int(row["rank_position"]),
                        "daily_universe_count": len(ranked),
                    },
                )
            )
    return signals_to_frame(signals)
