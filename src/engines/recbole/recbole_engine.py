# src/engines/recbole/recbole_engine.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from src.data.data_bundle import DataBundle
from src.engines.core.engine_base import EngineBase
from src.engines.recbole.adapter import export_to_recbole_inter


class RecBoleEngine(EngineBase):
    """
    RecBole 래퍼 엔진.
    - 우리 프레임워크의 logger/setting을 사용한다 (EngineBase 계약).
    - RecBole의 내부 logger는 보조로만 사용.
    """

    def __init__(self, cfg: Any, logger, setting):
        super().__init__(cfg, logger, setting)

    def fit(self, bundle: DataBundle) -> Dict[str, Any]:
        self.logger.info("[RecBole] fit() start")

        schema = bundle.schema or {}
        user_col = schema.get("user_col", "user_id")
        item_col = schema.get("item_col", "item_id")
        rating_col = schema.get("rating_col") or schema.get("target_col")
        time_col = schema.get("time_col")

        work_dir = Path(getattr(self.cfg.recbole, "work_dir", f"{self.setting.run_dir}/recbole"))
        data_dir = work_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        dataset = self.cfg.recbole.dataset

        ds_spec = export_to_recbole_inter(
            bundle=bundle,
            out_dir=data_dir,
            dataset=dataset,
            user_col=user_col,
            item_col=item_col,
            rating_col=rating_col,
            time_col=time_col,
        )
        self.logger.info(f"[RecBole] exported inter: {ds_spec.inter_path}")

        # --- RecBole run ---
        from recbole.config import Config
        from recbole.data import create_dataset, data_preparation
        from recbole.utils import init_seed, init_logger, get_model, get_trainer

        tcfg = self.cfg.recbole_train
        mcfg = self.cfg.model_args[self.cfg.model]

        overrides = {
            "data_path": str(ds_spec.data_path),
            "dataset": ds_spec.dataset,
            "model": self.cfg.model,

            "field_separator": "\t",
            "USER_ID_FIELD": "user_id",
            "ITEM_ID_FIELD": "item_id",
        }
        if rating_col is not None:
            overrides["RATING_FIELD"] = "rating"
        if time_col is not None:
            overrides["TIME_FIELD"] = "timestamp"

        overrides.update({
            "seed": int(tcfg.seed),
            "epochs": int(tcfg.epochs),
            "train_batch_size": int(tcfg.train_batch_size),
            "eval_batch_size": int(tcfg.eval_batch_size),
            "learning_rate": float(tcfg.learning_rate),
        })

        # 모델 파라미터
        overrides.update(dict(mcfg))

        # 유저 override 최종
        if getattr(self.cfg.recbole, "overrides", None):
            overrides.update(dict(self.cfg.recbole.overrides))

        config = Config(model=self.cfg.model, dataset=ds_spec.dataset, config_dict=overrides)

        # RecBole 내부 로거 초기화(보조). 메인 로깅은 self.logger로 계속 남긴다.
        init_seed(config["seed"], config.get("reproducibility", True))
        init_logger(config)

        self.logger.info(f"[RecBole] config summary: model={config['model']} dataset={config['dataset']} device={config['device']}")

        dataset_obj = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset_obj)

        model = get_model(config["model"])(config, train_data.dataset).to(config["device"])
        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

        best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, saved=True, show_progress=True)
        self.logger.info(f"[RecBole] best_valid_score={best_valid_score} best_valid_result={best_valid_result}")

        test_result = trainer.evaluate(test_data, load_best_model=True)
        self.logger.info(f"[RecBole] test_result={test_result}")

        return {
            "best_valid_score": best_valid_score,
            "best_valid_result": best_valid_result,
            "test_result": test_result,
        }

    def predict(self, bundle: DataBundle, checkpoint: Optional[str] = None):
        # MVP: 필요해지면 CheckpointPolicy 붙여서 torch와 동일 규칙으로 가져가면 됨
        self.logger.info("[RecBole] predict() is not implemented in MVP (use fit/evaluate first).")
        raise NotImplementedError
