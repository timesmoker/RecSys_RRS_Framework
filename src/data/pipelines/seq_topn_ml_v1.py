# src/data/pipelines/seq_topn_ml_v1.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from src.data.data_bundle import DataBundle
from src.data.pipelines.base import DataPipelineBase
from src.data.loaders.ml_train_dir import load_ml_train_dir
from src.data.pipelines.registry import register_pipeline

@register_pipeline("seq_topn_ml_v1")
class SeqTopNMLV1Pipeline(DataPipelineBase):
    name = "seq_topn_ml_v1"

    def __init__(self, cfg: Any):
        super().__init__(cfg)

    def load_raw(self, cfg: Any) -> Dict[str, Any]:
        # loader contract: {"ratings": df, "item2attributes": ..., ...}
        return load_ml_train_dir(cfg)


    def prepare_data(
        self, cfg: Any, raw: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame, Dict[str, Any]]:
        """
        seq_topn 준비 단계:
          - user별 시퀀스 구성 (time 정렬)
          - submission users order 생성
          - (선택) train/valid split은 향후 holdout 정책으로 확장
        """
        ratings: pd.DataFrame = raw["ratings"].copy()

        user_col = str(cfg.dataset.get("user_col", "user"))
        item_col = str(cfg.dataset.get("item_col", "item"))
        time_col = str(cfg.dataset.get("time_col", "time"))

        # submission users order: train에서 등장한 user unique 순서 고정
        users: List[int] = ratings[user_col].drop_duplicates().tolist()

        # user_seq: user -> list[item] (time 정렬)
        ratings_sorted = ratings.sort_values([user_col, time_col], ascending=True)
        user_seq = ratings_sorted.groupby(user_col)[item_col].apply(list).to_dict()

        tr = ratings.reset_index(drop=True)
        va = None

        # test는 Problem/Engine 계약상 항상 존재해야 하므로
        # 현재 스켈레톤에서는 "빈 DF(컬럼 유지)"로 채워 둠.
        te = ratings.iloc[0:0].copy().reset_index(drop=True)

        meta = {
            "pipeline": self.name,
            "submission": {"users": users},
            "user_seq": user_seq,
            # raw에 포함된 부가 리소스가 필요하면 meta로 넘겨도 됨(선택)
            # "item2attributes": raw.get("item2attributes", None),
        }
        return tr, va, te, meta

    '''
    def get_post_transforms(self, cfg: Any) -> List[PostTransform]:
        # config로 조립하는 게 최종형이지만, 우선은 코드로 박아도 됨
        return [
            UserItemCountTransform(user_col="user_id", item_col="isbn"),
        ]
    '''

    def to_bundle(
        self,
        cfg: Any,
        train_df: pd.DataFrame,
        valid_df: Optional[pd.DataFrame],
        test_df: pd.DataFrame,
        meta: Dict[str, Any],
    ) -> DataBundle:
        user_col = str(cfg.dataset.get("user_col", "user"))
        item_col = str(cfg.dataset.get("item_col", "item"))
        time_col = str(cfg.dataset.get("time_col", "time"))

        schema = {
            "task": "seq_topn",
            "user_col": user_col,
            "item_col": item_col,
            "time_col": time_col,
            "target_col": None,
            "feature_cols": None,
        }
        return DataBundle(
            train=train_df,
            valid=valid_df,
            test=test_df,
            schema=schema,
            meta=meta,
        )
