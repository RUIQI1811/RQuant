"""Convert model prediction scores into unified signals."""

from __future__ import annotations

import pandas as pd


def scores_to_signals(
    scores: pd.DataFrame,
    source: str,
    date_col: str = "date",
    symbol_col: str = "symbol",
    score_col: str = "score",
) -> pd.DataFrame:
    required = {date_col, symbol_col, score_col}
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    frame = scores[[date_col, symbol_col, score_col]].copy()
    frame[date_col] = pd.to_datetime(frame[date_col]).dt.strftime("%Y-%m-%d")
    frame[symbol_col] = frame[symbol_col].astype(str).str.zfill(6)
    frame["signal_type"] = "buy"
    frame["source"] = f"model_{source}"
    frame["weight"] = 0.0
    frame["metadata"] = "{}"
    return frame.rename(columns={score_col: "score"})[
        [date_col, symbol_col, "signal_type", "source", "score", "weight", "metadata"]
    ]
