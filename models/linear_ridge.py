"""Ridge regression score model."""

from __future__ import annotations

import numpy as np
import pandas as pd


class RidgeModel:
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.coef_: np.ndarray | None = None
        try:
            from sklearn.linear_model import Ridge
        except ImportError:
            self.model = None
        else:
            self.model = Ridge(alpha=alpha)

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "RidgeModel":
        if self.model is not None:
            self.model.fit(features, target)
            return self
        x = np.asarray(features, dtype=float)
        y = np.asarray(target, dtype=float)
        design = np.column_stack([np.ones(len(x)), x])
        penalty = np.eye(design.shape[1]) * self.alpha
        penalty[0, 0] = 0.0
        self.coef_ = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if self.model is not None:
            values = self.model.predict(features)
            return pd.Series(values, index=features.index, name="score")
        if self.coef_ is None:
            raise ValueError("RidgeModel must be fitted before predict")
        x = np.asarray(features, dtype=float)
        design = np.column_stack([np.ones(len(x)), x])
        values = design @ self.coef_
        return pd.Series(values, index=features.index, name="score")
