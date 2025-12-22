from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.data.data_bundle import DataBundle


@dataclass(frozen=True)
class RecBoleDatasetSpec:
    data_path: Path
    dataset: str
    inter_path: Path


def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in DataFrame: {missing}")


def export_to_recbole_inter(
    bundle: DataBundle,
    out_dir: str | Path,
    dataset: str,
    user_col: str,
    item_col: str,
    rating_col: Optional[str] = None,
    time_col: Optional[str] = None,
) -> RecBoleDatasetSpec:
    """
    Engine/Adapter 책임:
      DataBundle(DF)를 RecBole .inter 파일로 export.
    Pipeline은 엔진을 모른다.

    최소 계약:
      bundle.train 에 user/item은 존재해야 한다.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # MVP: train(+valid)만 사용. 필요시 valid 포함 여부를 스위치로 바꿀 수 있음.
    if bundle.train is None:
        raise ValueError("bundle.train is None; RecBole requires interactions in bundle.train")

    df = bundle.train.copy()
    if bundle.valid is not None:
        df = pd.concat([df, bundle.valid], axis=0, ignore_index=True)

    needed = [user_col, item_col]
    if rating_col:
        needed.append(rating_col)
    if time_col:
        needed.append(time_col)
    _ensure_cols(df, needed)

    # RecBole은 field type을 config에서 지정하는 패턴이 많음.
    # 여기서는 파일은 '원본 값' 그대로 저장하고, 타입 선언은 config(overrides)에서 처리.
    # 파일 컬럼명은 RecBole config의 *_FIELD와 반드시 일치해야 함.
    columns: Dict[str, str] = {
        user_col: "user_id",
        item_col: "item_id",
    }
    if rating_col:
        columns[rating_col] = "rating"
    if time_col:
        columns[time_col] = "timestamp"

    inter_df = df[list(columns.keys())].rename(columns=columns)

    inter_path = out_dir / f"{dataset}.inter"
    # RecBole 기본은 tab 구분을 많이 씀. field_separator를 config로 맞추면 됨.
    inter_df.to_csv(inter_path, sep="\t", index=False)

    return RecBoleDatasetSpec(
        data_path=out_dir,
        dataset=dataset,
        inter_path=inter_path,
    )
