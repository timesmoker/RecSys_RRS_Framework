from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.data_bundle import DataBundle
from src.data.loaders.books_csv import load_books_csv
from src.data.pipelines.base import DataPipelineBase
from src.data.transforms.base import PostTransform
from src.data.transforms.post_transform.post_counts import UserItemCountTransform
from src.data.pipelines.registry import register_pipeline

@register_pipeline("books_rating_v1")
class BooksRatingV1Pipeline(DataPipelineBase):
    name = "books_rating_v1"
    def __init__(self, cfg: Any):
        super().__init__(cfg)

    def load_raw(self, cfg: Any) -> Dict[str, pd.DataFrame]:
        return load_books_csv(cfg)

    def get_post_transforms(self, cfg: Any) -> List[PostTransform]:
        return [
            UserItemCountTransform(user_col="user_id", item_col="isbn"),
        ]

    def prepare_data(
        self, cfg: Any, raw: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame, Dict[str, Any]]:
        train = raw["train"].copy()
        test = raw["test"].copy()

        valid_ratio = float(cfg.dataset.get("valid_ratio", 0.2))
        seed = int(cfg.get("seed", 42))
        tr, va = train_test_split(train, test_size=valid_ratio, random_state=seed, shuffle=True)

        return tr.reset_index(drop=True), va.reset_index(drop=True), test.reset_index(drop=True), {
            "pipeline": self.name,
        }

    def to_bundle(self, cfg: Any, train_df: pd.DataFrame, valid_df: Optional[pd.DataFrame], test_df: pd.DataFrame, meta: Dict[str, Any]) -> DataBundle:
        target_col = "rating"
        feature_cols = [c for c in train_df.columns if c != target_col]

        schema = {
            "task": "regression",
            "user_col": "user_id",
            "item_col": "isbn",
            "time_col": None,
            "target_col": target_col,
            "feature_cols": feature_cols,
        }
        return DataBundle(train=train_df, valid=valid_df, test=test_df, schema=schema, meta=meta)
