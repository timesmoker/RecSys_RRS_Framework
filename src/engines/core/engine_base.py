# src/engines/core/engine_base.py
from __future__ import annotations

from typing import Any, Optional
from abc import ABC, abstractmethod

from src.data.data_bundle import DataBundle


class EngineBase(ABC):
    def __init__(self, cfg: Any, logger, setting):
        self.cfg = cfg
        self.logger = logger
        self.setting = setting

    @abstractmethod
    def fit(self, data_bundle: DataBundle) -> None:
        ...

    @abstractmethod
    def predict(self, data_bundle: DataBundle, checkpoint: Optional[str] = None):
        ...
