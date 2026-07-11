"""LightGBM score model wrapper."""

from __future__ import annotations

import pandas as pd


class LightGBMModel:
    def __init__(self, **params: object) -> None:
        try:
            import lightgbm as lgb
        except (ImportError, OSError) as exc:
            detail = (
                " and the native libomp runtime"
                if "libomp.dylib" in str(exc).casefold()
                else ""
            )
            raise ImportError(
                f"LightGBMModel requires lightgbm{detail} to be installed and importable"
            ) from exc
        self.model = lgb.LGBMRegressor(**params)

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "LightGBMModel":
        self.model.fit(features, target)
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        values = self.model.predict(features)
        return pd.Series(values, index=features.index, name="score")
