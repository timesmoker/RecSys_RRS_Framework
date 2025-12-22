# src/engines/torch/torch_base.py
from __future__ import annotations
from typing import Any, List

import torch
import numpy as np

from src.engines.core.engine_base import EngineBase
from src.engines.core.common import PredsValidator
from src.models.torch.recipes.registry import build_torch_recipe


class TorchBaseEngine(EngineBase):
    """
    Torch 공통 실행 엔진 (단 하나).

    책임:
    - train / predict 루프
    - device / AMP / grad clip
    - checkpoint save/load
    - logger / validator 호출

    책임 아님:
    - 태스크 개념
    - 데이터 의미
    - padding / sampling / candidate
    """

    def __init__(self, cfg: Any, logger, setting):
        super().__init__(cfg, logger, setting)

        device_str = str(getattr(cfg, "device", "cpu"))
        self.device = torch.device(
            device_str if device_str.startswith("cuda") and torch.cuda.is_available() else "cpu"
        )

        self.use_amp = bool(getattr(cfg, "amp", False))
        self.grad_clip = float(getattr(cfg, "grad_clip", 0.0) or 0.0)

        self.recipe = build_torch_recipe(cfg)

        self.model = None
        self.optimizer = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    # ---------------- public API ----------------
    def fit(self, bundle):
        print("BUNDLE_META_KEYS=", sorted(bundle.meta.keys()))

        if bool(getattr(self.cfg, "predict", False)):
            return

        loaders = self.recipe.build_loaders(self.cfg, bundle)
        self._init_train_components(bundle)

        epochs = int(getattr(self.cfg, "epochs", 1))

        for epoch in range(epochs):
            self.model.train()
            for batch in loaders["train"]:
                batch = self.recipe.move_batch_to_device(batch, self.device)

                self.optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    out = self.recipe.train_step(self.cfg, batch, self.model)
                    loss = out["loss"]

                self.scaler.scale(loss).backward()

                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()

            self._save_checkpoint(epoch)

    def predict(self, bundle, checkpoint: str | None = None):
        loaders = self.recipe.build_loaders(self.cfg, bundle)
        self._load_checkpoint(checkpoint)

        self.model.eval()
        preds_chunks: List[Any] = []

        with torch.no_grad():
            for batch in loaders["test"]:
                batch = self.recipe.move_batch_to_device(batch, self.device)
                out = self.recipe.predict_step(self.cfg, batch, self.model)
                preds_chunks.append(out)

        preds = self._merge_preds(preds_chunks)
        PredsValidator.validate(preds, bundle)
        return preds

    # ---------------- internal ----------------
    def _init_train_components(self, bundle):
        self.model = self.recipe.build_model(self.cfg, bundle).to(self.device)
        self.optimizer = self.recipe.build_optimizer(self.cfg, self.model)

    def _save_checkpoint(self, epoch: int):
        path = f"{self.setting.run_dir}/last.pt"
        self.setting.ensure_dir(self.setting.run_dir)
        torch.save({"model": self.model.state_dict()}, path)

    def _load_checkpoint(self, path: str | None):
        if not path:
            path = f"{self.setting.run_dir}/last.pt"
        obj = torch.load(path, map_location="cpu")
        self.model.load_state_dict(obj["model"])
        self.model.to(self.device)

    def _merge_preds(self, chunks):
        # 기본: numpy concat (regression)
        if isinstance(chunks[0], np.ndarray):
            return np.concatenate(chunks, axis=0)
        # list-of-list (topn/seq)
        out = []
        for c in chunks:
            out.extend(c)
        return out
