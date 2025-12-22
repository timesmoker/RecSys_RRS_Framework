#src/engines/recbole/registry.py
from __future__ import annotations

from src.engines.registry import ENGINE_REGISTRY
from src.engines.recbole.recbole_engine import RecBoleEngine


def build_recbole_engine(cfg):
    return RecBoleEngine(cfg)


ENGINE_REGISTRY["recbole"] = build_recbole_engine
