from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.model_loader import load_all_models
from core.input_processor import prepare_inputs_for_both

loaded = load_all_models(ROOT, enforce_versions=True)
configs = loaded["configurations"]
inputs = prepare_inputs_for_both([], configs)

print("MODEL LOAD AUDIT: PASS")
for approach in ("stable_fci", "star_ci"):
    for model_name, model in loaded["models"][approach].items():
        X = inputs[approach]
        proba = model.predict_proba(X)
        print(
            f"{approach}/{model_name}: input={X.shape}, "
            f"probabilities={proba.shape}, row_sum={float(proba[0].sum()):.8f}"
        )
print("This was an inference smoke test only; no model was fitted.")
