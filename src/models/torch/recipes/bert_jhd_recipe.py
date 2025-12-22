# src/models/torch/recipes/bert_jhd_recipe.py
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from src.models.torch.recipes.base import TorchRecipeBase
from src.models.torch.datasets.bert_jhd import TorchBertJHDDataset
from src.models.torch.collates.bert_jhd import bert_jhd_collate
from src.models.torch.modules.bert_JHD import bert_rec


class BertJHDRecipe(TorchRecipeBase):

    def build_model(self, cfg, bundle):
        # 모델은 model_args만 신뢰
        mcfg = self.model_cfg()
        return bert_rec(mcfg)

    def build_optimizer(self, cfg, model):
        tcfg = self.train_cfg()
        return torch.optim.Adam(model.parameters(), lr=float(tcfg.lr))

    def build_loaders(self, cfg, bundle):
        tcfg = self.train_cfg()

        train_ds = TorchBertJHDDataset(
            bundle.meta["X_train_idx"], bundle.meta["y_train"]
        )
        test_ds = TorchBertJHDDataset(
            bundle.meta["X_test_idx"], None
        )

        return {
            "train": DataLoader(
                train_ds,
                batch_size=int(tcfg.batch_size),
                shuffle=True,
                collate_fn=bert_jhd_collate,
            ),
            "test": DataLoader(
                test_ds,
                batch_size=int(tcfg.batch_size),
                shuffle=False,
                collate_fn=bert_jhd_collate,
            ),
        }

    def train_step(self, cfg, batch, model):
        X, y = batch
        pred = model(X)
        loss = torch.mean((pred - y) ** 2)
        return {"loss": loss}

    def predict_step(self, cfg, batch, model):
        return model(batch).detach().cpu().numpy()
