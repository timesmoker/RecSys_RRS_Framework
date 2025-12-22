# src/factories/engine_factory.py
from __future__ import annotations
from typing import Any

from src.engines.core.engine_base import EngineBase
from src.engines.registry import ENGINE_REGISTRY

class EngineFactory:
    @classmethod
    def build(cls, cfg: Any, logger, setting) -> EngineBase:
        import src.engines  # noqa: F401  (trigger registrations)

        engine_type = getattr(cfg, "engine", None)
        if engine_type is None:
            raise ValueError("cfg.engine is required")

        # cfg.engine.type 권장
        t = getattr(engine_type, "type", None)
        if t is None:
            # mapping/dict 스타일도 지원
            try:
                t = engine_type.get("type")
            except Exception:
                pass
        if not t:
            raise ValueError("cfg.engine.type is required")

        t = str(t)
        if t not in ENGINE_REGISTRY:
            raise ValueError(f"Unknown engine.type={t}. Available={list(ENGINE_REGISTRY.keys())}")

        return ENGINE_REGISTRY[t](cfg=cfg, logger=logger, setting=setting)
