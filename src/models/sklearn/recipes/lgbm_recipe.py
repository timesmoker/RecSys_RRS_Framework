# src/models/sklearn/recipes/lgbm_recipe.py

from __future__ import annotations

from typing import Any, Dict
from lightgbm import LGBMRegressor

from src.models.sklearn.recipes.base import SklearnRegressionRecipeSpec
from src.models.sklearn.recipes.registry import register_sklearn_recipe


@register_sklearn_recipe("LightGBM")
def build(cfg: Any) -> SklearnRegressionRecipeSpec:
    return LGBMRegressionRecipe(cfg)


class LGBMRegressionRecipe(SklearnRegressionRecipeSpec):
    name = "LightGBM"

    def build_estimator(self, cfg: Any, bundle):
        params = {}
        if hasattr(cfg, "model_args") and "LightGBM" in cfg.model_args:
            params = dict(cfg.model_args["LightGBM"])
        else:
            m = getattr(cfg, "model", None)
            params = dict(getattr(m, "args", {}) or {})

        params.pop("datatype", None)
        params.setdefault("objective", "regression")
        return LGBMRegressor(**params)

    def fit_params(self, cfg: Any, bundle) -> Dict[str, Any]:
        """
        sklearn Pipeline(step name='estimator') prefix 필요.
        """
        if bundle.valid is None or bundle.valid.empty:
            return {}

        feature_cols, target_col = self.prepare_schema(bundle)
        X_val = bundle.valid[feature_cols]
        y_val = bundle.valid[target_col].values

        params: Dict[str, Any] = {
            "estimator__eval_set": [(X_val, y_val)],
            "estimator__eval_metric": "rmse",
        }

        # 필요하면 model_args에 넣어서 제어할 수 있게 해둠(선택)
        # LightGBM sklearn API는 callbacks도 받음. 여기서는 MVP로 생략.

        return params
