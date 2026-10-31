from __future__ import annotations

from pathlib import Path
import sys

import pytest
import sklearn
import xgboost

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if sklearn.__version__ != "1.9.0" or xgboost.__version__ != "3.1.3":
    pytest.skip(
        "Serialized model loading requires scikit-learn==1.9.0 and xgboost==3.1.3.",
        allow_module_level=True,
    )

from core.model_loader import load_all_models
from core.input_processor import prepare_inputs_for_both


def test_all_six_models_load_and_predict():
    loaded = load_all_models(ROOT)
    inputs = prepare_inputs_for_both([], loaded["configurations"])
    for approach in ("stable_fci", "star_ci"):
        for model_name, model in loaded["models"][approach].items():
            probabilities = model.predict_proba(inputs[approach])
            assert probabilities.shape == (1, 41)
            assert (probabilities >= 0).all()
            assert abs(float(probabilities[0].sum()) - 1.0) < 1e-5
