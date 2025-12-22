# src/models/sklearn/recipes/catboost_recipe.py

from __future__ import annotations

from typing import Any, List, Dict

from src.models.sklearn.recipes.base import SklearnRegressionRecipeSpec
from src.models.sklearn.recipes.registry import register_sklearn_recipe


@register_sklearn_recipe("CatBoost")
def build(cfg: Any) -> SklearnRegressionRecipeSpec:
    return CatBoostRegressionRecipe(cfg)


class CatBoostRegressionRecipe(SklearnRegressionRecipeSpec):
    name = "CatBoost"

    def build_preprocessor(self, bundle, feature_cols: List[str]):
        return None

    def build_estimator(self, cfg: Any, bundle):
        import catboost

        params = {}
        if hasattr(cfg, "model_args") and "CatBoost" in cfg.model_args:
            params = dict(cfg.model_args["CatBoost"])
        else:
            m = getattr(cfg, "model", None)
            params = dict(getattr(m, "args", {}) or {})

        return catboost.CatBoostRegressor(**params)

    def configure_estimator(self, bundle, estimator, feature_cols):
        params = estimator.get_params(deep=False)
        if params.get("cat_features") is not None:
            return estimator

        meta = bundle.meta or {}
        numeric = meta.get("numeric_features")
        if not numeric:
            sample = bundle.train[feature_cols]
            numeric = [c for c in feature_cols if sample[c].dtype != "object"]

        numeric_set = set(numeric)
        cat_features = [c for c in feature_cols if c not in numeric_set]
        estimator.set_params(cat_features=cat_features)
        return estimator

    def fit_params(self, cfg: Any, bundle) -> Dict[str, Any]:
        """
        IMPORTANT:
          sklearn Pipeline(step name='estimator')에 전달되므로 prefix가 필요함.
        """
        if bundle.valid is None or bundle.valid.empty:
            return {}

        feature_cols, target_col = self.prepare_schema(bundle)
        X_val = bundle.valid[feature_cols]
        y_val = bundle.valid[target_col].values

        params: Dict[str, Any] = {
            "estimator__eval_set": (X_val, y_val),
            "estimator__use_best_model": True,
        }

        try:
            margs = dict(cfg.model_args.get("CatBoost", {}))
            if "early_stopping_rounds" in margs:
                params["estimator__early_stopping_rounds"] = int(margs["early_stopping_rounds"])
        except Exception:
            pass

        return params
