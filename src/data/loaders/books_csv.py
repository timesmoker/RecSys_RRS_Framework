from __future__ import annotations

from typing import Any, Dict
import pandas as pd


def load_books_csv(cfg: Any) -> Dict[str, pd.DataFrame]:
    base = cfg.dataset.data_path
    users = pd.read_csv(base + "users.csv")
    books = pd.read_csv(base + "books.csv")
    train = pd.read_csv(base + "train_ratings.csv")
    test = pd.read_csv(base + "test_ratings.csv")
    sub = pd.read_csv(base + "sample_submission.csv")
    return {"users": users, "books": books, "train": train, "test": test, "sub": sub}
