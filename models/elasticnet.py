"""ElasticNet regression score model."""

from __future__ import annotations

import numpy as np
import pandas as pd


class ElasticNetModel:
    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        *,
        max_iter: int = 10000,
        tol: float = 1e-6,
    ) -> None:
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if not 0 <= self.l1_ratio <= 1:
            raise ValueError("l1_ratio must be in [0, 1]")
        if self.max_iter <= 0 or self.tol <= 0:
            raise ValueError("max_iter and tol must be positive")
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.n_iter_: int = 0
        try:
            from sklearn.linear_model import ElasticNet
        except ImportError:
            self.model = None
        else:
            self.model = ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                max_iter=max_iter,
                tol=tol,
            )

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "ElasticNetModel":
        if self.model is not None:
            self.model.fit(features, target)
            return self
        x = np.asarray(features, dtype=float)
        y = np.asarray(target, dtype=float)
        if x.ndim != 2 or len(x) != len(y) or len(x) == 0:
            raise ValueError("features and target must be non-empty aligned arrays")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("features and target must contain only finite values")
        feature_mean = x.mean(axis=0)
        feature_scale = x.std(axis=0)
        feature_scale = np.where(feature_scale > 1e-12, feature_scale, 1.0)
        target_mean = float(y.mean())
        standardized = (x - feature_mean) / feature_scale
        centered_target = y - target_mean
        coefficients = np.zeros(standardized.shape[1], dtype=float)
        l1_penalty = self.alpha * self.l1_ratio
        l2_penalty = self.alpha * (1.0 - self.l1_ratio)
        sample_count = len(standardized)
        for iteration in range(1, self.max_iter + 1):
            previous = coefficients.copy()
            for column in range(standardized.shape[1]):
                feature = standardized[:, column]
                residual = centered_target - standardized @ coefficients + feature * coefficients[column]
                rho = float(feature @ residual) / sample_count
                denominator = float(feature @ feature) / sample_count + l2_penalty
                coefficients[column] = (
                    _soft_threshold(rho, l1_penalty) / denominator
                    if denominator > 1e-12
                    else 0.0
                )
            if float(np.max(np.abs(coefficients - previous))) <= self.tol:
                self.n_iter_ = iteration
                break
        else:
            self.n_iter_ = self.max_iter
        self.coef_ = coefficients / feature_scale
        self.intercept_ = target_mean - float(feature_mean @ self.coef_)
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if self.model is not None:
            values = self.model.predict(features)
            return pd.Series(values, index=features.index, name="score")
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("ElasticNetModel must be fitted before predict")
        x = np.asarray(features, dtype=float)
        values = x @ self.coef_ + self.intercept_
        return pd.Series(values, index=features.index, name="score")


def _soft_threshold(value: float, threshold: float) -> float:
    if value > threshold:
        return value - threshold
    if value < -threshold:
        return value + threshold
    return 0.0
