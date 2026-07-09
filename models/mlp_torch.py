"""Minimal PyTorch MLP score model with explicit dependency check."""

from __future__ import annotations


class TorchMLPModel:
    def __init__(self, *args: object, **kwargs: object) -> None:
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ImportError("TorchMLPModel requires torch to be installed") from exc
        raise NotImplementedError("TorchMLPModel training is not wired into the first restructure pass")
