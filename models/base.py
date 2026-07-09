"""Shared model interface for supervised stock-score models."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class ScoreModel(Protocol):
    def fit(self, features: pd.DataFrame, target: pd.Series) -> "ScoreModel":
        ...

    def predict(self, features: pd.DataFrame) -> pd.Series:
        ...
