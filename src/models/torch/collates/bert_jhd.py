from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union, Optional
import numpy as np
import torch


def _to_long_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.long()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).long()
    return torch.tensor(x, dtype=torch.long)


def _to_float_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.float()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    return torch.tensor(x, dtype=torch.float)


def bert_jhd_collate(batch: List[Any]) -> Union[torch.Tensor, Tuple[Any, torch.Tensor]]:
    """
    Supports:
      - train:  [(X, y), ...]
      - test :  [X, ...] or [(X, None), ...]
    Where X can be:
      - Tensor/ndarray/list (fixed-length)
      - dict of fields (e.g., input_ids/attention_mask/...)
    """

    # Normalize items
    first = batch[0]
    has_y = False

    if isinstance(first, (tuple, list)) and len(first) == 2:
        has_y = first[1] is not None

    # --- Case 1: X is dict ---
    def collate_dict(xs: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        keys = xs[0].keys()
        for k in keys:
            vals = [x[k] for x in xs]
            # assume token ids / masks are int-like
            out[k] = _to_long_tensor(vals)
        return out

    if isinstance(first, (tuple, list)) and len(first) == 2:
        Xs = [x for x, _ in batch]
        ys = [y for _, y in batch]
    else:
        Xs = batch
        ys = None

    if isinstance(Xs[0], dict):
        X_batch = collate_dict(Xs)  # type: ignore[arg-type]
    else:
        # numeric/index features: stack into [B, ...]
        X_batch = _to_long_tensor(Xs)

    if has_y:
        y_batch = _to_float_tensor(ys)  # type: ignore[arg-type]
        return X_batch, y_batch

    return X_batch
