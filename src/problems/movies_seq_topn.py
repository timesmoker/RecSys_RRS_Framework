from __future__ import annotations

from typing import Any, Optional, List
import pandas as pd

from src.data.data_bundle  import DataBundle
from src.problems.base import ProblemBase
from src.factories.pipeline_factory import PipelineFactory
from src.problems.registry import register_problem

@register_problem("movies_seq_topn", pipelines=["seq_topn_v1"])
class MoviesSeqTopNProblem(ProblemBase):
    name = "movies_seq_topn"

    def build_data_bundle(self) -> DataBundle:
        pipeline_name = self.cfg.data.get("pipeline") or self.cfg.problem.get("pipeline")
        if not pipeline_name:
            raise ValueError("cfg.data.pipeline is required for movies_seq_topn")
        pipeline = PipelineFactory.build(self.cfg)
        return pipeline.build(self.cfg)

    def save_submission(
        self,
        preds: List[List[int]],
        cfg: Any,
        setting,
        bundle: DataBundle,
    ) -> Optional[str]:

        # validate_bundle에서 이미 meta['submission']['users']를 강제한다고 가정
        bundle_users = bundle.meta["submission"]["users"]

        if len(preds) != len(bundle_users):
            raise ValueError(
                f"preds length mismatch: preds={len(preds)} users={len(bundle_users)}"
            )

        result = []
        for idx, items in enumerate(preds):
            u = bundle_users[idx]
            for it in items:
                result.append((u, it))

        sub_df = pd.DataFrame(result, columns=["user", "item"])
        train_cfg = cfg.get("train", {})  # train 없으면 {}
        submit_dir = train_cfg.get("submit_dir", "saved/submit")

        out_path = setting.get_submit_path(
            base_dir=submit_dir,
            model=cfg.model,
            run_name=cfg.get("run_name"),
        )
        sub_df.to_csv(out_path, index=False)
        return out_path

    def evaluate_preds(self, preds: List[List[int]], cfg: Any, bundle: DataBundle) -> dict | None:
        """
        Competition-friendly metric for this problem:
        - Recall@K computed as hit-rate of the held-out last item per user.

        NOTE:
        - We don't have public labels for eval in the competition setting.
        - This provides a lightweight, always-available sanity metric using train history.
        """
        meta = bundle.meta or {}
        sub = meta.get("submission", {}) or {}
        users = sub.get("users") or []
        user_seq = meta.get("user_seq") or {}

        if not users or not user_seq:
            return None

        # K: prefer cfg.train.topk, fallback 10
        k = 10
        try:
            k = int(cfg.get("train", {}).get("topk", 10))
        except Exception:
            pass

        hits = 0
        total = 0
        for u, recs in zip(users, preds):
            seq = user_seq.get(u)
            if not seq:
                continue
            gt = seq[-1]
            try:
                gt = int(gt)
            except Exception:
                gt = gt
            total += 1
            if gt in recs[:k]:
                hits += 1

        if total == 0:
            return None

        return {f"Recall@{k}": hits / total}