from __future__ import annotations

from typing import Any, Dict, Optional
import json
import os

import pandas as pd


def load_ml_train_dir(cfg: Any) -> Dict[str, Any]:
    """
    MovieLens-like sequential competition loader.

    Expected files under cfg.dataset.data_path:
      - train_ratings.csv  (user,item,time)
      - Ml_item2attributes.json (optional but common)
      - titles.tsv / years.tsv / genres.tsv / directors.tsv / writers.tsv (optional)

    Returns:
      {
        "ratings": pd.DataFrame,
        "item2attributes": dict | None,
        "aux_paths": {...},
        "aux_tables": {...}  # optionally loaded if cfg.dataset.load_aux_tables=True
      }
    """
    base = cfg.dataset.data_path
    if not base.endswith(os.sep):
        base = base + os.sep

    ratings_path = os.path.join(base, "train_ratings.csv")
    if not os.path.exists(ratings_path):
        raise FileNotFoundError(f"Missing file: {ratings_path}")

    ratings = pd.read_csv(ratings_path)
    required = ["user", "item", "time"]
    missing = [c for c in required if c not in ratings.columns]
    if missing:
        raise ValueError(f"train_ratings.csv missing columns: {missing}")

    # optional json
    item2attr_path = os.path.join(base, cfg.dataset.get("item2attributes_file", "Ml_item2attributes.json"))
    item2attributes: Optional[dict] = None
    if os.path.exists(item2attr_path):
        with open(item2attr_path, "r", encoding="utf-8") as f:
            item2attributes = json.load(f)

    aux_paths = {
        "titles": os.path.join(base, "titles.tsv"),
        "years": os.path.join(base, "years.tsv"),
        "genres": os.path.join(base, "genres.tsv"),
        "directors": os.path.join(base, "directors.tsv"),
        "writers": os.path.join(base, "writers.tsv"),
    }

    load_aux = bool(cfg.dataset.get("load_aux_tables", False))
    aux_tables: Dict[str, pd.DataFrame] = {}
    if load_aux:
        # tsv들은 크기가 작으니 필요할 때만 로드
        for k, p in aux_paths.items():
            if os.path.exists(p):
                aux_tables[k] = pd.read_csv(p, sep="\t")

    # optional: sample_submission for competition template (users/order/K)
    sample_path = None
    try:
        sample_path = cfg.dataset.get("sample_submission_path", None)
    except Exception:
        sample_path = None

    # default: sibling eval dir (..../train -> ..../eval/sample_submission.csv)
    if not sample_path:
        base_dir = os.path.abspath(os.path.join(base, os.pardir))
        candidate = os.path.join(base_dir, "eval", "sample_submission.csv")
        if os.path.exists(candidate):
            sample_path = candidate

    sample_submission = None
    if sample_path and os.path.exists(sample_path):
        try:
            sample_submission = pd.read_csv(sample_path)
        except Exception:
            sample_submission = None

    return {
        "ratings": ratings,
        "item2attributes": item2attributes,
        "sample_submission": sample_submission,
        "aux_paths": aux_paths,
        "aux_tables": aux_tables,
        "paths": {
            "ratings_path": ratings_path,
            "item2attributes_path": item2attr_path if os.path.exists(item2attr_path) else None,
            "sample_submission_path": sample_path if (sample_path and os.path.exists(sample_path)) else None,
        },
    }
