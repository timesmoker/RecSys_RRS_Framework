# src/factories/pipeline_factory.py
from __future__ import annotations

from src.data.pipelines.registry import PIPELINE_REGISTRY

class PipelineFactory:
    @staticmethod
    def build(cfg):
        name = cfg.data.pipeline
        if name not in PIPELINE_REGISTRY:
            raise KeyError(f"Unknown pipeline: {name}. Available={sorted(PIPELINE_REGISTRY.keys())}")
        return PIPELINE_REGISTRY[name](cfg)