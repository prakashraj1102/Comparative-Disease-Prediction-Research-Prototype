from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, top_k_accuracy_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.model_loader import load_all_models
from core.input_processor import create_master_input, prepare_inputs_for_both
from core.prediction_engine import predict_one

MODEL_NAMES = ("logistic_regression", "random_forest", "xgboost")
METRICS = ("accuracy", "precision_macro", "recall_macro", "f1_macro", "top3_accuracy")
ABS_TOL = 1e-10


class ValidationFailure(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationFailure(message)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValidationFailure(f"Required reference file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_binary_frame(frame: pd.DataFrame, label: str) -> None:
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValidationFailure(f"{label} has duplicate column names: {duplicates}")
    if frame.isna().any().any():
        locations = np.argwhere(frame.isna().to_numpy())[:10].tolist()
        raise ValidationFailure(f"{label} contains missing values; first positions: {locations}")
    values = set(pd.unique(frame.to_numpy().ravel()).tolist())
    if not values.issubset({0, 1, 0.0, 1.0, False, True}):
        raise ValidationFailure(f"{label} must contain only binary 0/1 values; found: {sorted(map(str, values))[:10]}")


def _master_features(configs: dict[str, Any]) -> list[str]:
    return [item["feature_name"] for item in configs["master_symptoms"]["symptoms"]]


def _symptoms_from_row(row: pd.Series, master_features: list[str]) -> list[str]:
    return [name for name in master_features if int(row[name]) == 1]


def _expected_top_k(probabilities: np.ndarray, class_names: list[str], k: int = 3) -> list[tuple[str, float]]:
    values = np.asarray(probabilities, dtype=float).reshape(-1)
    order = np.argsort(-values, kind="stable")[:k]
    return [(class_names[int(index)], float(values[int(index)])) for index in order]


def _check_app_parity(
    loaded: dict[str, Any],
    source_rows: pd.DataFrame,
    *,
    approach: str,
    model_name: str,
    master_features: list[str],
    sample_positions: list[int],
) -> dict[str, Any]:
    configs = loaded["configurations"]
    feature_key = "stable_fci_features" if approach == "stable_fci" else "star_ci_features"
    model_features = configs[feature_key]["feature_names"]
    model = loaded["models"][approach][model_name]
    class_names = loaded["class_names"][approach][model_name]
    checked = 0
    max_probability_difference = 0.0

    for position in sample_positions:
        source_row = source_rows.iloc[position]
        selected = _symptoms_from_row(source_row, master_features)

        # Independent expected input: project the original source row directly by
        # the model's recorded feature names. This is intentionally separate from
        # the application's checkbox-to-feature projection.
        expected_X = source_rows.iloc[[position]].loc[:, model_features]
        expected_probabilities = np.asarray(model.predict_proba(expected_X), dtype=float)[0]
        expected_top3 = _expected_top_k(expected_probabilities, class_names, k=3)

        app_result = predict_one(
            loaded,
            selected,
            approach=approach,
            model_name=model_name,
            top_k=3,
        )
        actual_top3 = app_result["predictions"]
        actual_names = [item["disease"] for item in actual_top3]
        expected_names = [item[0] for item in expected_top3]
        _require(
            actual_names == expected_names,
            f"Top-3 disease mismatch: {approach}/{model_name}, source row {position}. "
            f"Expected {expected_names}; got {actual_names}.",
        )
        actual_scores = np.asarray([item["model_score"] for item in actual_top3], dtype=float)
        expected_scores = np.asarray([item[1] for item in expected_top3], dtype=float)
        difference = float(np.max(np.abs(actual_scores - expected_scores))) if len(actual_scores) else 0.0
        max_probability_difference = max(max_probability_difference, difference)
        _require(
            np.allclose(actual_scores, expected_scores, rtol=0.0, atol=ABS_TOL),
            f"Top-3 probability mismatch: {approach}/{model_name}, source row {position}, "
            f"max abs difference={difference:.3e}.",
        )
        checked += 1

    return {
        "application_parity": "PASS",
        "application_parity_samples": checked,
        "application_parity_max_top3_probability_abs_difference": max_probability_difference,
    }


def _stable_validation(
    project_root: Path,
    loaded: dict[str, Any],
    stable_dataset_path: Path,
    master_features: list[str],
) -> list[dict[str, Any]]:
    if not stable_dataset_path.is_file():
        raise ValidationFailure(f"Stable-FCI source dataset not found: {stable_dataset_path}")
    source = pd.read_csv(stable_dataset_path)
    disease_columns = [str(column) for column in source.columns if str(column).startswith("disease_")]
    symptom_columns = [str(column) for column in source.columns if str(column) not in disease_columns]

    _require(len(source) == 4920, f"Stable-FCI source dataset must have 4,920 rows; found {len(source)}.")
    _require(len(disease_columns) == 41, f"Stable-FCI source dataset must have 41 disease columns; found {len(disease_columns)}.")
    _require(len(symptom_columns) == 132, f"Stable-FCI source dataset must have 132 symptom columns; found {len(symptom_columns)}.")
    _require(set(symptom_columns) == set(master_features), "Stable-FCI source symptom columns do not exactly match the audited 132-symptom master list.")

    disease_frame = source.loc[:, disease_columns]
    symptom_frame = source.loc[:, master_features]
    _validate_binary_frame(disease_frame, "Stable-FCI disease matrix")
    _validate_binary_frame(symptom_frame, "Stable-FCI symptom matrix")
    row_sums = disease_frame.sum(axis=1).to_numpy()
    _require(np.all(row_sums == 1), "Stable-FCI disease matrix is not exactly one-hot in every row.")

    # Reproduce the exact split in Stable_FCI notebook cell 10.
    labels = disease_frame.idxmax(axis=1).str.replace("disease_", "", regex=False)
    train_indices, test_indices = train_test_split(
        source.index,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )
    stable_test_X = symptom_frame.loc[test_indices].reset_index(drop=True)
    stable_test_y = labels.loc[test_indices].reset_index(drop=True)
    _require(len(stable_test_X) == 984, f"Recreated Stable-FCI test split should have 984 rows; found {len(stable_test_X)}.")
    _require(stable_test_y.nunique() == 41, "Recreated Stable-FCI test split does not contain all 41 diseases.")

    reference_split = _read_json(project_root / "reference_reports" / "stable_fci_train_test_split.json")
    _require(reference_split.get("n_train") == 3936 and reference_split.get("n_test") == 984,
             "Archived Stable-FCI split metadata does not match the expected 3,936/984 protocol.")
    train_counts = labels.loc[train_indices].value_counts().to_dict()
    test_counts = stable_test_y.value_counts().to_dict()
    archived_train_counts = reference_split.get("train_class_counts", {})
    archived_test_counts = reference_split.get("test_class_counts", {})
    _require(train_counts == archived_train_counts,
             "Recreated Stable-FCI training class counts differ from the archived split metadata.")
    _require(test_counts == archived_test_counts,
             "Recreated Stable-FCI test class counts differ from the archived split metadata.")

    rows: list[dict[str, Any]] = []
    expected_by_model = {
        "logistic_regression": "stable_fci_logistic_regression_summary.json",
        "random_forest": "stable_fci_random_forest_summary.json",
        "xgboost": "stable_fci_xgboost_summary.json",
    }
    # At least one held-out example from every disease is passed through the
    # app API, in addition to scoring the complete 984-row test split.
    representative_positions = []
    for disease in sorted(stable_test_y.unique().tolist()):
        matches = np.flatnonzero(stable_test_y.to_numpy() == disease)
        _require(len(matches) > 0, f"No Stable-FCI test example found for disease {disease!r}.")
        representative_positions.append(int(matches[0]))

    for model_name in MODEL_NAMES:
        model = loaded["models"]["stable_fci"][model_name]
        class_names = loaded["class_names"]["stable_fci"][model_name]
        feature_names = loaded["configurations"]["stable_fci_features"]["feature_names"]
        X = stable_test_X.loc[:, feature_names]
        probabilities = np.asarray(model.predict_proba(X), dtype=float)
        _require(probabilities.shape == (984, 41),
                 f"Stable-FCI {model_name} returned probability shape {probabilities.shape}, expected (984, 41).")
        _require(np.isfinite(probabilities).all(), f"Stable-FCI {model_name} returned non-finite probabilities.")
        _require(np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6),
                 f"Stable-FCI {model_name} has probability rows that do not sum to one.")

        predictions = np.asarray(model.predict(X)).astype(str)
        accuracy = float(accuracy_score(stable_test_y, predictions))
        precision, recall, f1, _ = precision_recall_fscore_support(
            stable_test_y, predictions, average="macro", zero_division=0,
        )
        top3 = float(top_k_accuracy_score(stable_test_y, probabilities, k=3, labels=class_names))
        actual_metrics = {
            "accuracy": accuracy,
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
            "top3_accuracy": top3,
        }

        summary = _read_json(project_root / "reference_reports" / expected_by_model[model_name])
        expected_metrics = summary["stable_fci"]["test_metrics"]
        reference_differences = {
            key: abs(actual_metrics[key] - float(expected_metrics[key])) for key in METRICS
        }
        reference_match = all(value <= ABS_TOL for value in reference_differences.values())
        if not reference_match:
            detail = {key: {"actual": actual_metrics[key], "reference": float(expected_metrics[key]),
                            "abs_difference": reference_differences[key]} for key in METRICS
                      if reference_differences[key] > ABS_TOL}
            raise ValidationFailure(
                f"Stable-FCI {model_name} metrics do not reproduce the archived notebook results: {detail}"
            )

        parity = _check_app_parity(
            loaded,
            stable_test_X,
            approach="stable_fci",
            model_name=model_name,
            master_features=master_features,
            sample_positions=representative_positions,
        )
        rows.append({
            "approach": "stable_fci",
            "model": model_name,
            "source_test_rows": 984,
            "reference_notebook_metrics": "PASS",
            "reference_metric_max_abs_difference": max(reference_differences.values()),
            **actual_metrics,
            **parity,
            "status": "PASS",
        })
    return rows


def _star_validation(
    project_root: Path,
    loaded: dict[str, Any],
    star_test_symptoms_path: Path,
    master_features: list[str],
) -> list[dict[str, Any]]:
    if not star_test_symptoms_path.is_file():
        raise ValidationFailure(f"STAR test symptom matrix not found: {star_test_symptoms_path}")
    raw_test = pd.read_csv(star_test_symptoms_path)
    _require(len(raw_test) == 42, f"STAR test symptoms file should have 42 rows; found {len(raw_test)}.")
    _require(len(raw_test.columns) == 132, f"STAR test symptoms file should have 132 columns; found {len(raw_test.columns)}.")
    _require(not raw_test.columns.duplicated().any(), "STAR test symptoms file contains duplicate column headers.")
    _require(set(raw_test.columns) == set(master_features),
             "STAR test symptom columns do not exactly match the audited 132-symptom master list.")
    _validate_binary_frame(raw_test, "STAR test symptom matrix")
    test_X_full = raw_test.loc[:, master_features].copy()

    fixture = _read_json(project_root / "reference_reports" / "star_ci_sample0_reference.json")
    actual_row0_symptoms = [name for name in loaded["configurations"]["star_ci_features"]["feature_names"]
                            if int(test_X_full.iloc[0][name]) == 1]
    _require(set(actual_row0_symptoms) == set(fixture["observed_symptoms"]),
             "STAR test row 0 does not match the three selected symptoms printed by the original notebook example. "
             f"Expected {fixture['observed_symptoms']}; got {actual_row0_symptoms}.")

    reference_path = project_root / "reference_reports" / "star_ci_reference_prediction_probabilities.csv"
    if not reference_path.is_file():
        raise ValidationFailure(f"Archived STAR Logistic Regression prediction reference is missing: {reference_path}")
    reference = pd.read_csv(reference_path)
    _require(len(reference) == 42, f"Archived STAR prediction reference should have 42 rows; found {len(reference)}.")
    if "patient_index" in reference.columns:
        expected_indices = list(range(42))
        _require(reference["patient_index"].astype(int).tolist() == expected_indices,
                 "Archived STAR prediction rows are not in their original 0..41 order.")

    rows: list[dict[str, Any]] = []
    star_features = loaded["configurations"]["star_ci_features"]["feature_names"]
    X = test_X_full.loc[:, star_features]

    for model_name in MODEL_NAMES:
        model = loaded["models"]["star_ci"][model_name]
        class_names = loaded["class_names"]["star_ci"][model_name]
        probabilities = np.asarray(model.predict_proba(X), dtype=float)
        _require(probabilities.shape == (42, 41),
                 f"STAR {model_name} returned probability shape {probabilities.shape}, expected (42, 41).")
        _require(np.isfinite(probabilities).all(), f"STAR {model_name} returned non-finite probabilities.")
        _require(np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6),
                 f"STAR {model_name} has probability rows that do not sum to one.")

        # Check the frontend's symptom-to-feature mapping and top-three rendering
        # against direct inference on the exact source test rows.
        parity = _check_app_parity(
            loaded,
            test_X_full,
            approach="star_ci",
            model_name=model_name,
            master_features=master_features,
            sample_positions=list(range(42)),
        )

        reference_match = "NOT_AVAILABLE"
        max_reference_difference = None
        if model_name == "logistic_regression":
            # The STAR notebook saved all 42 LR probability vectors. Compare every
            # one of the 41 output scores, not only the winning class or top three.
            raw_classes = [str(value) for value in getattr(model, "classes_")]
            reference_probability_columns = [f"proba__{label}" for label in raw_classes]
            missing = [column for column in reference_probability_columns if column not in reference.columns]
            _require(not missing, f"STAR reference probability CSV is missing class columns: {missing}")
            expected = reference.loc[:, reference_probability_columns].to_numpy(dtype=float)
            _require(expected.shape == probabilities.shape,
                     f"STAR reference probability shape {expected.shape} differs from model output {probabilities.shape}.")
            max_reference_difference = float(np.max(np.abs(probabilities - expected)))
            _require(np.allclose(probabilities, expected, rtol=0.0, atol=ABS_TOL),
                     f"STAR Logistic Regression probabilities differ from archived notebook predictions; "
                     f"max abs difference={max_reference_difference:.3e}.")
            raw_predictions = np.asarray(model.predict(X)).astype(str)
            _require(raw_predictions.tolist() == reference["predicted_disease"].astype(str).tolist(),
                     "STAR Logistic Regression predicted disease labels do not match the archived notebook output.")
            reference_match = "PASS"

        rows.append({
            "approach": "star_ci",
            "model": model_name,
            "source_test_rows": 42,
            "reference_notebook_metrics": reference_match,
            "reference_metric_max_abs_difference": max_reference_difference,
            **parity,
            "status": "PASS",
        })
    return rows



def _validate_input_edge_cases(loaded: dict[str, Any], master_features: list[str]) -> dict[str, Any]:
    configs = loaded["configurations"]
    # Empty selection is intentionally not sent to a classifier: the Streamlit
    # predict button is disabled until at least one symptom is selected.
    empty_inputs = prepare_inputs_for_both([], configs)
    _require(empty_inputs["master"].shape == (1, 132) and int(empty_inputs["master"].to_numpy().sum()) == 0,
             "Empty-selection input preparation did not produce an all-zero 132-feature vector.")
    _require(empty_inputs["stable_fci"].shape == (1, 37), "Empty-selection Stable-FCI projection has the wrong shape.")
    _require(empty_inputs["star_ci"].shape == (1, 106), "Empty-selection STAR projection has the wrong shape.")

    cases = {
        "single_symptom": [master_features[0]],
        "three_symptoms": ["skin_rash", "nodal_skin_eruptions", "dischromic _patches"],
        "all_symptoms": list(master_features),
    }
    checked_predictions = 0
    for case_name, selected in cases.items():
        for approach in ("stable_fci", "star_ci"):
            for model_name in MODEL_NAMES:
                result = predict_one(
                    loaded, selected, approach=approach, model_name=model_name, top_k=3,
                )
                _require(len(result["predictions"]) == 3,
                         f"{case_name}: {approach}/{model_name} did not return exactly three predictions.")
                scores = np.asarray([item["model_score"] for item in result["predictions"]], dtype=float)
                _require(np.isfinite(scores).all() and (scores >= 0).all() and (scores <= 1).all(),
                         f"{case_name}: {approach}/{model_name} returned invalid top-three scores.")
                _require(result["probability_sum_check"] == "PASS",
                         f"{case_name}: {approach}/{model_name} failed the probability-sum check.")
                checked_predictions += 1
    return {
        "empty_selection": "PASS (all-zero input prepared; UI prevents prediction)",
        "single_symptom_model_predictions": 6,
        "three_symptom_model_predictions": 6,
        "all_symptoms_model_predictions": 6,
        "non_empty_edge_case_predictions_checked": checked_predictions,
    }

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 4 validation: replay saved experiment test inputs through the deployed prediction engine."
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--stable-dataset", type=Path, required=True,
                        help="Exact 4,920-row train_encoded.csv used by the Stable-FCI notebook.")
    parser.add_argument("--star-test-symptoms", type=Path, required=True,
                        help="Exact 42-row Test_Only_Symptoms.csv used by the STAR notebook.")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    stable_path = args.stable_dataset.expanduser().resolve()
    star_path = args.star_test_symptoms.expanduser().resolve()
    output_dir = (args.output_dir or (project_root / "reports" / "stage4_validation")).resolve()

    try:
        print("STAGE 4 VALIDATION")
        print("=" * 72)
        print("No training or causal discovery will be run.")
        print("Loading and integrity-checking the six existing model artifacts...")
        loaded = load_all_models(project_root, enforce_versions=True)
        master_features = _master_features(loaded["configurations"])
        _require(len(master_features) == 132, f"Expected 132 master symptoms; found {len(master_features)}.")
        print("Checking single, multi-symptom, all-symptom and empty-selection handling...")
        input_edge_cases = _validate_input_edge_cases(loaded, master_features)
        print("  PASS: input edge cases")

        print("\n[1/2] Recreating Stable-FCI held-out split and validating saved metrics...")
        stable_rows = _stable_validation(project_root, loaded, stable_path, master_features)
        for row in stable_rows:
            print(f"  PASS Stable-FCI / {row['model']}: test metrics reproduced; "
                  f"app parity rows={row['application_parity_samples']}")

        print("\n[2/2] Replaying STAR external test symptoms and checking saved LR probabilities...")
        star_rows = _star_validation(project_root, loaded, star_path, master_features)
        for row in star_rows:
            extra = "; archived 41-class notebook probability match" if row["reference_notebook_metrics"] == "PASS" else ""
            print(f"  PASS STAR / {row['model']}: app parity rows={row['application_parity_samples']}{extra}")

        result_rows = stable_rows + star_rows
        summary = {
            "stage": "Stage 4 - Prediction Validation",
            "status": "PASS",
            "created_at": datetime.now().astimezone().isoformat(),
            "project_root": str(project_root),
            "stable_dataset": str(stable_path),
            "star_test_symptoms": str(star_path),
            "runtime": {
                "scikit_learn": __import__("sklearn").__version__,
                "xgboost": __import__("xgboost").__version__,
            },
            "models_checked": 6,
            "stable_fci_test_rows": 984,
            "star_ci_test_rows": 42,
            "master_symptoms": len(master_features),
            "stable_fci_selected_features": len(loaded["configurations"]["stable_fci_features"]["feature_names"]),
            "star_ci_selected_features": len(loaded["configurations"]["star_ci_features"]["feature_names"]),
            "input_edge_cases": input_edge_cases,
            "results": result_rows,
            "limitations": [
                "STAR archived per-row probability vectors are available for Logistic Regression only; STAR RF and XGBoost are validated against direct inference from their unchanged serialized models.",
                "Stable-FCI archive contains aggregate test metrics but no per-row probability export; the 984-row test metrics are therefore reproduced from the exact original dataset and recorded split, and app parity is checked on one held-out example per disease.",
                "This validates software inference consistency and reproduction of saved notebook outputs; it does not establish clinical validity or probability calibration."
            ],
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "stage4_validation_summary.json"
        csv_path = output_dir / "stage4_validation_details.csv"
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        pd.DataFrame(result_rows).to_csv(csv_path, index=False)
        print("\n" + "=" * 72)
        print("STAGE 4 VALIDATION: PASS")
        print(f"Summary: {json_path}")
        print(f"Details: {csv_path}")
        print("All six models passed the configured inference checks.")
        return 0

    except Exception as exc:
        print("\nSTAGE 4 VALIDATION: FAIL")
        print(f"Reason: {exc}", file=sys.stderr)
        print("No model files were modified and no training was performed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
