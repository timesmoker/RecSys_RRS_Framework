# src/data/transforms/registry.py
from __future__ import annotations

from typing import Dict, Type, Callable

from src.data.transforms.base import GlobalTransform, PostTransform

GLOBAL_TRANSFORM_REGISTRY: Dict[str, Type[GlobalTransform]] = {}
POST_TRANSFORM_REGISTRY: Dict[str, Type[PostTransform]] = {}


def register_global(name: str) -> Callable[[Type[GlobalTransform]], Type[GlobalTransform]]:
    def deco(cls: Type[GlobalTransform]) -> Type[GlobalTransform]:
        if name in GLOBAL_TRANSFORM_REGISTRY and GLOBAL_TRANSFORM_REGISTRY[name] is not cls:
            raise KeyError(f"Duplicate global transform: {name}")
        GLOBAL_TRANSFORM_REGISTRY[name] = cls
        return cls
    return deco


def register_post(name: str) -> Callable[[Type[PostTransform]], Type[PostTransform]]:
    def deco(cls: Type[PostTransform]) -> Type[PostTransform]:
        if name in POST_TRANSFORM_REGISTRY and POST_TRANSFORM_REGISTRY[name] is not cls:
            raise KeyError(f"Duplicate post transform: {name}")
        POST_TRANSFORM_REGISTRY[name] = cls
        return cls
    return deco
