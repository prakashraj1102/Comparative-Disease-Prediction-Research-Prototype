from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.prediction_engine import predict_comparison

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configurations"


class FixedProbabilityModel:
    def __init__(self, top_indices):
        self.top_indices = top_indices

    def predict_proba(self, X):
        assert X.shape[0] == 1
        probs = np.full((1, 41), 0.001, dtype=float)
        # Assign descending, deterministic values to three distinct output classes.
        probs[0, self.top_indices[0]] = 0.70
        probs[0, self.top_indices[1]] = 0.20
        probs[0, self.top_indices[2]] = 0.06
        probs /= probs.sum(axis=1, keepdims=True)
        return probs


def read(name):
    return json.loads((CONFIG / name).read_text(encoding="utf-8"))


def test_comparison_runs_two_independent_top_three_predictions():
    configs = {
        "master_symptoms": read("master_symptoms.json"),
        "stable_fci_features": read("stable_fci_features.json"),
        "star_ci_features": read("star_ci_features.json"),
        "disease_labels": read("disease_labels.json"),
    }
    names = [item["display_name"] for item in configs["disease_labels"]["classes"]]
    loaded = {
        "configurations": configs,
        "models": {
            "stable_fci": {
                "random_forest": FixedProbabilityModel([4, 2, 0]),
            },
            "star_ci": {
                "logistic_regression": FixedProbabilityModel([2, 5, 1]),
            },
        },
        "class_names": {
            "stable_fci": {"random_forest": names},
            "star_ci": {"logistic_regression": names},
        },
    }

    result = predict_comparison(
        loaded,
        ["itching", "fatigue"],
        stable_model="random_forest",
        star_model="logistic_regression",
        top_k=3,
    )

    assert result["stable_fci"]["model_feature_count"] == 37
    assert result["star_ci"]["model_feature_count"] == 106
    assert [x["rank"] for x in result["stable_fci"]["predictions"]] == [1, 2, 3]
    assert [x["disease"] for x in result["stable_fci"]["predictions"]] == [names[4], names[2], names[0]]
    assert [x["disease"] for x in result["star_ci"]["predictions"]] == [names[2], names[5], names[1]]
    assert result["stable_fci"]["probability_sum_check"] == "PASS"
    assert result["star_ci"]["probability_sum_check"] == "PASS"
    assert result["stable_fci"]["selected_model_feature_count"] == 2
    assert result["star_ci"]["selected_model_feature_count"] == 1
    assert result["stable_fci"]["ignored_selected_symptom_count"] == 0
    assert result["star_ci"]["ignored_selected_symptom_count"] == 1
