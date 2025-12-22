from __future__ import annotations

from typing import Any, Optional
import pandas as pd

from src.data.data_bundle import DataBundle
from src.problems.base import ProblemBase
from src.factories.pipeline_factory import PipelineFactory
from src.problems.registry import register_problem

@register_problem("books_rating")
class BooksRatingProblem(ProblemBase):
    name = "books_rating"

    def build_data_bundle(self) -> DataBundle:
        pipeline_name = self.cfg.data.get("pipeline")
        if not pipeline_name:
            raise ValueError("cfg.data.pipeline is required for books_rating")
        pipeline = PipelineFactory.build(self.cfg)
        return pipeline.build(self.cfg)

    def save_submission(self, preds, cfg: Any, setting, bundle: DataBundle) -> Optional[str]:
        """
        sample_submission.csv 포맷을 따른다.

        Contract:
          - len(preds) == len(bundle.test)
          - bundle.test row order == submission row order
        """
        if bundle.test is None:
            raise ValueError("bundle.test is required for submission")

        if len(preds) != len(bundle.test):
            raise ValueError(
                f"pred length mismatch: preds={len(preds)} test={len(bundle.test)}"
            )

        base = cfg.dataset.data_path
        sub = pd.read_csv(base + "sample_submission.csv")

        if len(sub) != len(bundle.test):
            raise ValueError(
                f"sample_submission length mismatch: sub={len(sub)} test={len(bundle.test)}"
            )

        id_like = {"user_id", "isbn", "id"}
        target_cols = [c for c in sub.columns if c not in id_like]
        target_col = target_cols[0] if target_cols else "rating"

        if target_col not in sub.columns:
            sub[target_col] = 0.0

        sub[target_col] = preds

        submit_dir = cfg.train.get("submit_dir", "saved/submit")
        out_path = setting.get_submit_path(
            base_dir=submit_dir,
            model=cfg.model,
            run_name=cfg.get("run_name"),
        )
        sub.to_csv(out_path, index=False)
        return out_path
