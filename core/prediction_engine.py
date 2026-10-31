from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from core.input_processor import prepare_inputs_for_both


class PredictionError(RuntimeError):
    """Raised when a fitted model returns invalid probability output."""


def _top_k(
    model: Any,
    X,
    class_names: list[str],
    *,
    k: int = 3,
) -> list[dict[str, Any]]:
    if k < 1:
        raise ValueError("k must be at least 1.")
    if X.shape[0] != 1:
        raise ValueError(f"Expected one patient input row; got {X.shape[0]}.")

    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    if probabilities.shape != (1, len(class_names)):
        raise PredictionError(
            f"predict_proba returned shape {probabilities.shape}; "
            f"expected (1, {len(class_names)})."
        )
    if not np.isfinite(probabilities).all():
        raise PredictionError("Model returned NaN or infinite probability values.")
    if (probabilities < -1e-10).any() or (probabilities > 1 + 1e-10).any():
        raise PredictionError("Model returned values outside the probability range [0, 1].")

    row = probabilities[0]
    row_sum = float(row.sum())
    if not np.isclose(row_sum, 1.0, atol=1e-5):
        raise PredictionError(f"Model probability row sums to {row_sum:.8f}, not 1.0.")

    # Stable sorting keeps the original model class order as the tie-breaker.
    order = np.argsort(-row, kind="stable")[: min(k, len(class_names))]
    return [
        {
            "rank": rank,
            "disease": class_names[int(index)],
            "model_score": float(row[int(index)]),
            "model_score_percent": float(row[int(index)] * 100.0),
        }
        for rank, index in enumerate(order, start=1)
    ]


def predict_one(
    loaded_models: dict[str, Any],
    selected_symptoms: Iterable[str],
    *,
    approach: str,
    model_name: str,
    top_k: int = 3,
) -> dict[str, Any]:
    """Predict with one approach/model using the exact audited input columns."""
    approach = approach.lower()
    if approach not in {"stable_fci", "star_ci"}:
        raise ValueError("approach must be 'stable_fci' or 'star_ci'.")
    if model_name not in {"logistic_regression", "random_forest", "xgboost"}:
        raise ValueError("model_name must be logistic_regression, random_forest, or xgboost.")

    symptoms = list(selected_symptoms)
    configs = loaded_models["configurations"]
    inputs = prepare_inputs_for_both(symptoms, configs)
    model_input = inputs[approach]
    model = loaded_models["models"][approach][model_name]
    class_names = loaded_models["class_names"][approach][model_name]
    predictions = _top_k(model, model_input, class_names, k=top_k)

    feature_config = configs[f"{approach}_features"]
    model_features = set(feature_config["feature_names"])
    selected_set = set(symptoms)
    selected_features_present = [
        name for name in feature_config["feature_names"] if name in selected_set
    ]
    selected_symptoms_not_used = [
        name for name in symptoms if name not in model_features
    ]
    return {
        "approach": "Stable-FCI" if approach == "stable_fci" else "STAR CI-selected",
        "model": model_name,
        "selected_symptom_count": len(symptoms),
        "selected_model_feature_count": len(selected_features_present),
        "ignored_selected_symptom_count": len(selected_symptoms_not_used),
        "ignored_selected_symptoms": selected_symptoms_not_used,
        "model_feature_count": feature_config["feature_count"],
        "predictions": predictions,
        "probability_sum_check": "PASS",
        "score_note": "Model-estimated predict_proba score; not a calibrated clinical probability.",
    }


def predict_comparison(
    loaded_models: dict[str, Any],
    selected_symptoms: Iterable[str],
    *,
    stable_model: str = "random_forest",
    star_model: str = "logistic_regression",
    top_k: int = 3,
) -> dict[str, Any]:
    """Run independent predictions; no ensembling or averaging is performed."""
    symptoms = list(selected_symptoms)
    return {
        "selected_symptoms": symptoms,
        "stable_fci": predict_one(
            loaded_models, symptoms,
            approach="stable_fci", model_name=stable_model, top_k=top_k,
        ),
        "star_ci": predict_one(
            loaded_models, symptoms,
            approach="star_ci", model_name=star_model, top_k=top_k,
        ),
    }
