"""ElasticNet regression score model."""

from __future__ import annotations

import numpy as np
import pandas as pd


class ElasticNetModel:
    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5) -> None:
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.coef_: np.ndarray | None = None
        try:
            from sklearn.linear_model import ElasticNet
        except ImportError:
            self.model = None
        else:
            self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "ElasticNetModel":
        if self.model is not None:
            self.model.fit(features, target)
            return self
        x = np.asarray(features, dtype=float)
        y = np.asarray(target, dtype=float)
        design = np.column_stack([np.ones(len(x)), x])
        ridge_alpha = self.alpha * max(1.0 - self.l1_ratio, 0.0)
        penalty = np.eye(design.shape[1]) * ridge_alpha
        penalty[0, 0] = 0.0
        self.coef_ = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if self.model is not None:
            values = self.model.predict(features)
            return pd.Series(values, index=features.index, name="score")
        if self.coef_ is None:
            raise ValueError("ElasticNetModel must be fitted before predict")
        x = np.asarray(features, dtype=float)
        design = np.column_stack([np.ones(len(x)), x])
        values = design @ self.coef_
        return pd.Series(values, index=features.index, name="score")
