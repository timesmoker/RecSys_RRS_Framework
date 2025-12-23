from __future__ import annotations

from typing import Any, Callable, Dict

from .base import RecBoleRecipeBase

Builder = Callable[[Any], RecBoleRecipeBase]
_REG: Dict[str, Builder] = {}


def register_recbole_recipe(model_name: str):
    def deco(fn: Builder):
        if model_name in _REG and _REG[model_name] is not fn:
            raise KeyError(f"Duplicate recbole recipe: {model_name}")
        _REG[model_name] = fn
        return fn
    return deco


def build_recbole_recipe(cfg: Any) -> RecBoleRecipeBase:
    model = getattr(cfg, "model", None)
    if not model:
        raise ValueError("cfg.model is required to build recbole recipe")
    if model not in _REG:
        raise KeyError(f"Unknown recbole recipe: {model}. Available={sorted(_REG.keys())}")
    return _REG[model](cfg)


# explicit registration imports (no autodiscovery)
import src.models.recbole.recipes.lightgcn  # noqa: F401
import src.models.recbole.recipes.recvae  # noqa: F401