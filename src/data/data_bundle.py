from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Literal

import pandas as pd

TaskType = Literal["regression", "topn", "seq_topn"]


@dataclass(frozen=True)
class DataBundle:
    """
    Data → Engine contract (via Problem.run()).

    Required:
      - train/test: DataFrame
      - valid: DataFrame or None
      - schema: task/user_col/item_col/time_col/target_col (+ optional feature_cols)
      - meta: free-form
    """
    train: pd.DataFrame
    valid: Optional[pd.DataFrame]
    test: pd.DataFrame
    schema: Dict[str, Any]
    meta: Dict[str, Any]
