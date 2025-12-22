# src/engines/sklearn/sklearn_regression_engine.py
from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.pipeline import Pipeline

from src.data.data_bundle import DataBundle
from src.engines.sklearn.sklearn_base import SklearnEngineBase
from src.factories.sklearn_recipe_factory import SklearnRecipeFactory

class SklearnRegressionEngine(SklearnEngineBase):
    """
    Task-family engine for regression using sklearn-style estimators.
    Model-specific differences live ONLY in recipes.
    """

    def fit(self, bundle: DataBundle) -> None:
        self._guard_regression(bundle)

        recipe = SklearnRecipeFactory.build(self.cfg)
        feature_cols, target_col = recipe.prepare_schema(bundle)

        X_train = bundle.train[feature_cols]
        y_train = bundle.train[target_col].values

        preprocessor = recipe.build_preprocessor(bundle, feature_cols)
        estimator = recipe.build_estimator(self.cfg, bundle)
        estimator = recipe.configure_estimator(bundle, estimator, feature_cols)

        steps = []
        if preprocessor is not None:
            steps.append(("preprocess", preprocessor))
        steps.append(("estimator", estimator))

        pipe = Pipeline(steps=steps)
        fit_params = recipe.fit_params(self.cfg, bundle)
        pipe.fit(X_train, y_train, **fit_params)

        model_name = self._model_name()
        ckpt_path = self._default_ckpt_path(f"{model_name}.joblib")
        self._save_checkpoint(
            {
                "pipeline": pipe,
                "feature_cols": feature_cols,
                "target_col": target_col,
                "recipe": getattr(recipe, "name", model_name),
                "schema": dict(bundle.schema),
            },
            ckpt_path,
        )

        self._log_train(
            {
                "engine_family": "sklearn_regression",
                "recipe": getattr(recipe, "name", model_name),
                "checkpoint_saved": ckpt_path,
                "n_train": int(len(bundle.train)),
                "n_valid": int(len(bundle.valid)) if bundle.valid is not None else 0,
                "n_features": int(len(feature_cols)),
            }
        )

    def predict(self, bundle: DataBundle, checkpoint: Optional[str] = None):
        self._guard_regression(bundle)

        if bundle.test is None:
            raise ValueError("Regression prediction requires DataBundle.test (DataFrame).")

        model_name = self._model_name()
        ckpt_path = self._resolve_checkpoint(checkpoint, f"{model_name}.joblib")
        obj = self._load_checkpoint(ckpt_path)

        pipe: Pipeline = obj["pipeline"]
        feature_cols: List[str] = obj["feature_cols"]

        X_test = bundle.test[feature_cols]
        preds = pipe.predict(X_test)
        preds = np.asarray(preds).reshape(-1)

        self._validate_preds(preds, bundle)

        self._log_predict(
            {
                "engine_family": "sklearn_regression",
                "recipe": obj.get("recipe", model_name),
                "checkpoint_used": ckpt_path,
                "n_test": int(len(bundle.test)),
            }
        )
        return preds

    # ---------- helpers ----------
    @staticmethod
    def _guard_regression(bundle: DataBundle) -> None:
        if bundle.schema.get("task") != "regression":
            raise ValueError("SklearnRegressionEngine supports only schema.task == 'regression'.")

    def _model_name(self) -> str:
        m = getattr(self.cfg, "model", None)
        if m is None:
            return "model"
        name = getattr(m, "name", None)
        return str(name) if name is not None else str(m)
