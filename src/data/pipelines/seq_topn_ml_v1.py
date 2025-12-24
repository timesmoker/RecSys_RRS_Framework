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

        # train users (for building sequences / long_sequence)
        train_users: List[int] = ratings[user_col].drop_duplicates().tolist()

        # user_seq: user -> list[item] (time 정렬)
        ratings_sorted = ratings.sort_values([user_col, time_col], ascending=True)
        user_seq = ratings_sorted.groupby(user_col)[item_col].apply(list).to_dict()
        # long sequence (for pretraining negative segment sampling)
        long_sequence: List[int] = []
        for u in train_users:
            long_sequence.extend(user_seq.get(u, []) or [])

        # submission users order: prefer sample_submission template if available
        users: List[int] = train_users
        k_from_sample = None
        ss = raw.get("sample_submission", None)
        if isinstance(ss, pd.DataFrame) and (user_col in ss.columns):
            users = ss[user_col].drop_duplicates().tolist()
            try:
                vc = ss[user_col].value_counts()
                if len(vc) > 0:
                    k_from_sample = int(vc.iloc[0])
            except Exception:
                k_from_sample = None

        # optional item2attributes (for S3Rec pretraining tasks)
        item2attributes = raw.get("item2attributes", None)
        attribute_size = None
        if isinstance(item2attributes, dict) and item2attributes:
            try:
                mx = 0
                for _, attrs in item2attributes.items():
                    if not attrs:
                        continue
                    mx = max(mx, max(int(a) for a in attrs))
                # +1 for 0 padding
                attribute_size = int(mx) + 1
            except Exception:
                attribute_size = None

        tr = ratings.reset_index(drop=True)
        va = None

        # test는 Problem/Engine 계약상 항상 존재해야 하므로
        # 현재 스켈레톤에서는 "빈 DF(컬럼 유지)"로 채워 둠.
        te = ratings.iloc[0:0].copy().reset_index(drop=True)

        meta = {
            "pipeline": self.name,
            "submission": {"users": users, "k": k_from_sample},
            "user_seq": user_seq,
            "long_sequence": long_sequence,
            # optional: for pretraining
            "item2attributes": item2attributes,
            "attribute_size": attribute_size,
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
