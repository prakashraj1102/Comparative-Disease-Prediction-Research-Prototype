import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reference_reports"


def test_star_reference_contains_42_rows_and_41_probability_columns():
    frame = pd.read_csv(REF / "star_ci_reference_prediction_probabilities.csv")
    probability_columns = [name for name in frame.columns if name.startswith("proba__")]
    assert len(frame) == 42
    assert len(probability_columns) == 41
    assert frame["patient_index"].astype(int).tolist() == list(range(42))
    assert np.allclose(frame[probability_columns].sum(axis=1).to_numpy(), 1.0, atol=1e-10)


def test_star_sample_zero_reference_matches_notebook_example():
    fixture = json.loads((REF / "star_ci_sample0_reference.json").read_text())
    assert fixture["observed_symptoms"] == [
        "skin_rash", "nodal_skin_eruptions", "dischromic _patches"
    ]
    assert fixture["true_disease"] == "disease_Fungal infection"
    assert fixture["predicted_disease"] == "disease_Fungal infection"
    assert fixture["expected_top_5"][0]["disease"] == "disease_Fungal infection"
    assert np.isclose(fixture["expected_top_5"][0]["probability"], 0.9935232057942125, atol=1e-12)


def test_stable_fci_saved_reference_metrics_cover_all_three_models():
    expected_files = [
        "stable_fci_logistic_regression_summary.json",
        "stable_fci_random_forest_summary.json",
        "stable_fci_xgboost_summary.json",
    ]
    for filename in expected_files:
        payload = json.loads((REF / filename).read_text())
        metrics = payload["stable_fci"]["test_metrics"]
        assert metrics["n_features"] == 37
        assert metrics["split"] == "test"
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["top3_accuracy"] <= 1.0
