from __future__ import annotations
from typing import Any, Callable, Dict

SklearnBuilder = Callable[[Any], Any]   # registry에서는 Any로 둠
SKLEARN_RECIPE_REGISTRY: Dict[str, SklearnBuilder] = {}

def register_sklearn_recipe(name: str):
    def deco(fn: SklearnBuilder):
        prev = SKLEARN_RECIPE_REGISTRY.get(name)
        if prev is not None and prev is not fn:
            raise KeyError(f"Duplicate sklearn recipe: {name}")
        SKLEARN_RECIPE_REGISTRY[name] = fn
        return fn
    return deco


# -------- import triggers --------
from src.models.sklearn.recipes import catboost_recipe  # noqa
from src.models.sklearn.recipes import lgbm_recipe     # noqa