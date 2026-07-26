"""PyTorch MLP regression model for cross-sectional stock scores."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_MLP_EPOCHS = 10


class TorchMLPModel:
    """Small standardized MLP with deterministic training and local persistence."""

    def __init__(
        self,
        *,
        hidden_sizes: Sequence[int] = (64, 32),
        learning_rate: float = 1e-3,
        epochs: int = DEFAULT_MLP_EPOCHS,
        batch_size: int = 256,
        weight_decay: float = 0.0,
        dropout: float = 0.0,
        random_state: int = 42,
        device: str = "auto",
    ) -> None:
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ImportError("TorchMLPModel requires torch to be installed") from exc

        hidden = tuple(int(value) for value in hidden_sizes)
        if not hidden or any(value <= 0 for value in hidden):
            raise ValueError("hidden_sizes must contain positive integers")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.hidden_sizes = hidden
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.weight_decay = float(weight_decay)
        self.dropout = float(dropout)
        self.random_state = int(random_state)
        self.device_name = str(device)
        self._torch = torch
        self._nn = nn
        self.device = self._resolve_device(self.device_name)
        self.network = None
        self.feature_names_: tuple[str, ...] | None = None
        self.feature_mean_: np.ndarray | None = None
        self.feature_scale_: np.ndarray | None = None
        self.target_mean_: float | None = None
        self.target_scale_: float | None = None
        self.loss_history_: list[float] = []

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "TorchMLPModel":
        x, feature_names = self._feature_array(features, fitting=True)
        y = pd.to_numeric(target, errors="coerce").to_numpy(dtype=np.float32)
        if len(y) != len(x):
            raise ValueError("features and target must have the same row count")
        if not target.index.equals(features.index):
            raise ValueError("features and target indices must match")
        if not np.isfinite(y).all():
            raise ValueError("target must contain only finite values")

        self.feature_names_ = feature_names
        self.feature_mean_ = x.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = x.std(axis=0, dtype=np.float64).astype(np.float32)
        self.feature_scale_ = np.where(scale > 1e-12, scale, 1.0).astype(np.float32)
        self.target_mean_ = float(y.mean(dtype=np.float64))
        target_scale = float(y.std(dtype=np.float64))
        self.target_scale_ = target_scale if target_scale > 1e-12 else 1.0

        x_scaled = (x - self.feature_mean_) / self.feature_scale_
        y_scaled = (y - self.target_mean_) / self.target_scale_
        self._torch.manual_seed(self.random_state)
        if self._torch.cuda.is_available():
            self._torch.cuda.manual_seed_all(self.random_state)
        self.network = self._build_network(x.shape[1])

        x_tensor = self._torch.as_tensor(x_scaled, dtype=self._torch.float32)
        y_tensor = self._torch.as_tensor(y_scaled[:, None], dtype=self._torch.float32)
        dataset = self._torch.utils.data.TensorDataset(x_tensor, y_tensor)
        generator = self._torch.Generator()
        generator.manual_seed(self.random_state)
        loader = self._torch.utils.data.DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
            generator=generator,
        )
        optimizer = self._torch.optim.AdamW(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = self._nn.MSELoss()
        self.loss_history_ = []
        self.network.train()
        for _ in range(self.epochs):
            total_loss = 0.0
            seen = 0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                predictions = self.network(batch_x)
                loss = loss_fn(predictions, batch_y)
                loss.backward()
                optimizer.step()
                batch_count = len(batch_x)
                total_loss += float(loss.detach().cpu()) * batch_count
                seen += batch_count
            self.loss_history_.append(total_loss / max(seen, 1))
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        self._require_fitted()
        x, _ = self._feature_array(features, fitting=False)
        x_scaled = (x - self.feature_mean_) / self.feature_scale_
        tensor = self._torch.as_tensor(
            x_scaled,
            dtype=self._torch.float32,
            device=self.device,
        )
        self.network.eval()
        with self._torch.no_grad():
            scaled = self.network(tensor).squeeze(-1).detach().cpu().numpy()
        values = scaled * float(self.target_scale_) + float(self.target_mean_)
        return pd.Series(values, index=features.index, name="score")

    def save(self, path: str | Path) -> Path:
        self._require_fitted()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "hidden_sizes": self.hidden_sizes,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "weight_decay": self.weight_decay,
                "dropout": self.dropout,
                "random_state": self.random_state,
            },
            "feature_names": self.feature_names_,
            "feature_mean": self.feature_mean_,
            "feature_scale": self.feature_scale_,
            "target_mean": self.target_mean_,
            "target_scale": self.target_scale_,
            "loss_history": self.loss_history_,
            "state_dict": self.network.state_dict(),
        }
        temp = destination.with_name(f".{destination.name}.tmp")
        self._torch.save(payload, temp)
        temp.replace(destination)
        return destination

    @classmethod
    def load(cls, path: str | Path, *, device: str = "auto") -> "TorchMLPModel":
        try:
            import torch
        except ImportError as exc:
            raise ImportError("TorchMLPModel requires torch to be installed") from exc
        map_location = cls._resolve_device_static(torch, device)
        try:
            payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        except TypeError:
            payload = torch.load(Path(path), map_location=map_location)
        config = dict(payload["config"])
        model = cls(**config, device=device)
        model.feature_names_ = tuple(payload["feature_names"])
        model.feature_mean_ = np.asarray(payload["feature_mean"], dtype=np.float32)
        model.feature_scale_ = np.asarray(payload["feature_scale"], dtype=np.float32)
        model.target_mean_ = float(payload["target_mean"])
        model.target_scale_ = float(payload["target_scale"])
        model.loss_history_ = [float(value) for value in payload.get("loss_history", [])]
        model.network = model._build_network(len(model.feature_names_))
        model.network.load_state_dict(payload["state_dict"])
        model.network.eval()
        return model

    def _feature_array(
        self,
        features: pd.DataFrame,
        *,
        fitting: bool,
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        if not isinstance(features, pd.DataFrame) or features.empty:
            raise ValueError("features must be a non-empty DataFrame")
        if features.columns.duplicated().any():
            raise ValueError("feature columns must be unique")
        names = tuple(str(column) for column in features.columns)
        if not fitting:
            expected = self.feature_names_ or ()
            missing = set(expected).difference(names)
            extra = set(names).difference(expected)
            if missing or extra:
                raise ValueError(
                    "prediction feature columns do not match training columns; "
                    f"missing={sorted(missing)}, extra={sorted(extra)}"
                )
            features = features.loc[:, list(expected)]
            names = expected
        numeric = features.apply(pd.to_numeric, errors="coerce")
        values = numeric.to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("features must contain only finite numeric values")
        return values, names

    def _build_network(self, input_size: int):
        layers = []
        previous = input_size
        for hidden in self.hidden_sizes:
            layers.append(self._nn.Linear(previous, hidden))
            layers.append(self._nn.ReLU())
            if self.dropout > 0:
                layers.append(self._nn.Dropout(self.dropout))
            previous = hidden
        layers.append(self._nn.Linear(previous, 1))
        return self._nn.Sequential(*layers).to(self.device)

    def _resolve_device(self, requested: str):
        return self._resolve_device_static(self._torch, requested)

    @staticmethod
    def _resolve_device_static(torch, requested: str):
        name = str(requested).strip().lower()
        mps_backend = getattr(torch.backends, "mps", None)
        mps_available = bool(mps_backend and mps_backend.is_available())
        if name == "auto":
            if mps_available:
                name = "mps"
            elif torch.cuda.is_available():
                name = "cuda"
            else:
                name = "cpu"
        if name == "mps" and not mps_available:
            raise ValueError("requested MPS device is not available")
        if name.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("requested CUDA device is not available")
        if name != "cpu" and name != "mps" and not name.startswith("cuda"):
            raise ValueError("device must be auto, cpu, mps, or cuda[:index]")
        return torch.device(name)

    def _require_fitted(self) -> None:
        if (
            self.network is None
            or self.feature_names_ is None
            or self.feature_mean_ is None
            or self.feature_scale_ is None
            or self.target_mean_ is None
            or self.target_scale_ is None
        ):
            raise ValueError("TorchMLPModel must be fitted before predict or save")
