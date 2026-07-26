"""Qlib-native score-model wrappers used by RQuant walk-forward training."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


def _configure_openmp_threads(n_jobs: int) -> None:
    """Bound libomp before LightGBM is imported.

    On macOS, importing Qlib, MLflow and Torch in one research process can make
    libomp's unconstrained default fail while creating its mutex pool.  Qlib's
    own ``num_threads`` parameter is applied after the native runtime loads, so
    mirror an explicit positive worker count into ``OMP_NUM_THREADS`` first.
    An environment value supplied by the caller always wins.
    """

    workers = int(n_jobs)
    if workers > 0:
        os.environ.setdefault("OMP_NUM_THREADS", str(workers))


def _ensure_qlib_initialized() -> None:
    try:
        import qlib
        from qlib.config import C
    except (ImportError, OSError) as exc:
        raise ImportError(
            "Qlib models require pyqlib; install requirements-qlib.txt"
        ) from exc
    if not C.registered:
        qlib.init(provider_uri={})


class LightGBMModel:
    """Use Qlib's LGBModel directly; no native RQuant LightGBM fallback exists."""

    backend = "qlib"

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        n_jobs: int = 1,
        random_state: int = 42,
        **params: object,
    ) -> None:
        _configure_openmp_threads(n_jobs)
        _ensure_qlib_initialized()
        try:
            from qlib.contrib.model.gbdt import LGBModel
        except (ImportError, OSError) as exc:
            raise ImportError(
                "Qlib LightGBM requires pyqlib and an importable LightGBM runtime"
            ) from exc
        qlib_params = {
            "num_threads": int(n_jobs),
            "seed": int(random_state),
            "feature_fraction_seed": int(random_state),
            "bagging_seed": int(random_state),
            "data_random_seed": int(random_state),
            "verbosity": -1,
            **params,
        }
        self.model = LGBModel(
            loss="mse",
            num_boost_round=int(n_estimators),
            **qlib_params,
        )

    def fit(self, dataset: Any) -> "LightGBMModel":
        _ensure_qlib_initialized()
        from qlib.workflow import R

        # Qlib's LGBModel records per-round metrics through QlibRecorder. Keep
        # that implementation detail ephemeral; RQuant's run.json remains the
        # durable lifecycle source of truth.
        with tempfile.TemporaryDirectory(prefix="rquant-qlib-recorder-") as temp_dir:
            root = Path(temp_dir).resolve()
            database_path = root / "mlflow.db"
            artifact_path = root / "artifacts"
            artifact_path.mkdir()
            uri = f"sqlite:///{database_path}"
            from mlflow.tracking import MlflowClient

            experiment_id = MlflowClient(tracking_uri=uri).create_experiment(
                "rquant-lightgbm",
                artifact_location=artifact_path.as_uri(),
            )
            with R.start(experiment_id=str(experiment_id), uri=uri):
                self.model.fit(dataset, verbose_eval=0)
        return self

    def predict(self, dataset: Any, segment: str = "test") -> pd.Series:
        values = self.model.predict(dataset, segment=segment)
        return pd.Series(values, index=values.index, name="score")


class DoubleEnsembleModel:
    """Qlib DoubleEnsemble with LightGBM sub-models."""

    backend = "qlib"

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        n_jobs: int = 1,
        random_state: int = 42,
        num_models: int = 6,
        **params: object,
    ) -> None:
        _configure_openmp_threads(n_jobs)
        _ensure_qlib_initialized()
        try:
            from qlib.contrib.model.double_ensemble import DEnsembleModel
        except (ImportError, OSError) as exc:
            raise ImportError(
                "Qlib DoubleEnsemble requires pyqlib and an importable LightGBM runtime"
            ) from exc
        qlib_params = {"decay": 0.5, **params}
        self.model = DEnsembleModel(
            base_model="gbm",
            loss="mse",
            num_models=int(num_models),
            epochs=int(n_estimators),
            num_threads=int(n_jobs),
            seed=int(random_state),
            feature_fraction_seed=int(random_state),
            bagging_seed=int(random_state),
            data_random_seed=int(random_state),
            verbosity=-1,
            **qlib_params,
        )

    def fit(self, dataset: Any) -> "DoubleEnsembleModel":
        self.model.fit(dataset)
        return self

    def predict(self, dataset: Any, segment: str = "test") -> pd.Series:
        values = self.model.predict(dataset, segment=segment)
        return pd.Series(values, index=values.index, name="score")
