from __future__ import annotations
from typing import Any, Dict, Type, Callable
from src.data.pipelines.base import DataPipelineBase

PIPELINE_REGISTRY: Dict[str, Type[DataPipelineBase]] = {}

def register_pipeline(name: str) -> Callable[[Type[DataPipelineBase]], Type[DataPipelineBase]]:
    def deco(cls: Type[DataPipelineBase]) -> Type[DataPipelineBase]:
        if name in PIPELINE_REGISTRY and PIPELINE_REGISTRY[name] is not cls:
            raise KeyError(f"Duplicate pipeline: {name}")
        PIPELINE_REGISTRY[name] = cls
        return cls
    return deco

# ---- explicit registration imports (NO autodiscover) ----
import src.data.pipelines.books_rating_v1  # noqa: F401
import src.data.pipelines.books_rating_v2  # noqa: F401
import src.data.pipelines.seq_topn_ml_v1   # noqa: F401
