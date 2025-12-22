from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset


class TorchBertJHDDataset(Dataset):
    """
    Torch 전용 Dataset for bert_JHD.

    Input contract:
    - X_idx: np.ndarray[int64]
        shape = (N, num_features + 1)
        X_idx[:, 0]   -> summary_index
        X_idx[:, 1:]  -> categorical feature indices
    - y: np.ndarray[float32] | None

    Notes:
    - dtype/의미 보존이 핵심 (tensor화는 collate 책임)
    - padding / sampling / transform 없음 (규율)
    """

    def __init__(
        self,
        X_idx: np.ndarray,
        y: np.ndarray | None,
    ):
        assert X_idx.ndim == 2, "X_idx must be 2D array (N, num_fields+1)"

        self.X = X_idx.astype(np.int64, copy=False)
        self.y = None if y is None else y.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]
