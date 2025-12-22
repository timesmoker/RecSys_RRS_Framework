from __future__ import annotations

from typing import Any, Dict, Tuple
import pandas as pd

from src.data.transforms.base import PostTransform
from src.data.transforms.registry import register_post

@register_post("user_item_count")
class UserItemCountTransform(PostTransform):
    name = "user_item_count"

    def __init__(self, user_col: str, item_col: str):
        self.user_col = user_col
        self.item_col = item_col

    def fit(self, cfg: Any, raw: Dict[str, pd.DataFrame], train_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        user_cnt = train_df.groupby(self.user_col).size().rename("user_cnt").reset_index()
        item_cnt = train_df.groupby(self.item_col).size().rename("item_cnt").reset_index()
        return {"user_cnt": user_cnt, "item_cnt": item_cnt}

    def transform(self, cfg: Any, raw: Dict[str, pd.DataFrame], df: pd.DataFrame, state: Any) -> pd.DataFrame:
        user_cnt = state["user_cnt"]
        item_cnt = state["item_cnt"]

        out = df.merge(user_cnt, on=self.user_col, how="left").merge(item_cnt, on=self.item_col, how="left")
        out["user_cnt"] = out["user_cnt"].fillna(0).astype(int)
        out["item_cnt"] = out["item_cnt"].fillna(0).astype(int)
        return out
