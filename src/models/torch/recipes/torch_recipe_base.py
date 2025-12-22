# src/models/torch/recipes/torch_recipe_base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import torch


class TorchRecipeBase(ABC):
    """
    Strict config contract:

    - train hyperparams live under cfg.torch.*
      (batch_size, lr, epochs, amp, grad_clip, ...)

    - model hyperparams live under cfg.model_args[cfg.model].*
      (embed_dim, n_layers, dropout, cardinality, ...)
    """

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self._validate_cfg_contract()

    def _validate_cfg_contract(self) -> None:
        if not hasattr(self.cfg, "torch"):
            raise KeyError("Missing cfg.torch (train hyperparams must be under cfg.torch.*)")
        if not hasattr(self.cfg, "model"):
            raise KeyError("Missing cfg.model")
        if not hasattr(self.cfg, "model_args"):
            raise KeyError("Missing cfg.model_args")
        if self.cfg.model not in self.cfg.model_args:
            raise KeyError(f"Missing cfg.model_args['{self.cfg.model}']")

    def train_cfg(self):
        # cfg.torch.* only
        return self.cfg.torch

    def model_cfg(self):
        # cfg.model_args[cfg.model].* only
        return self.cfg.model_args[self.cfg.model]

    @abstractmethod
    def build_model(self, cfg, bundle): ...
    @abstractmethod
    def build_optimizer(self, cfg, model): ...
    @abstractmethod
    def build_loaders(self, cfg, bundle): ...
    @abstractmethod
    def train_step(self, cfg, batch, model): ...
    @abstractmethod
    def predict_step(self, cfg, batch, model): ...

    def move_batch_to_device(self, batch, device):
        if torch.is_tensor(batch):
            return batch.to(device)
        if isinstance(batch, tuple):
            return tuple(self.move_batch_to_device(b, device) for b in batch)
        if isinstance(batch, dict):
            return {k: self.move_batch_to_device(v, device) for k, v in batch.items()}
        return batch
