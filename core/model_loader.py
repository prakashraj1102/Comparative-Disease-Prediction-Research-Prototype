from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
import xgboost

from core.config import load_configurations
from core.model_compat import register_stable_fci_wrapper_for_unpickling


class ModelArtifactError(RuntimeError):
    """Raised when a saved model does not match its audited configuration."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _effective_estimator(model: Any) -> Any:
    return getattr(model, "_model", model)


def _feature_names(model: Any) -> list[str]:
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        names = getattr(_effective_estimator(model), "feature_names_in_", None)
    if names is None:
        raise ModelArtifactError(f"{type(model).__name__} does not expose feature_names_in_.")
    return [str(name) for name in names]


def _model_feature_count(model: Any) -> int | None:
    value = getattr(model, "n_features_in_", None)
    if value is None:
        value = getattr(_effective_estimator(model), "n_features_in_", None)
    return int(value) if value is not None else None


def _classes(model: Any) -> list[Any]:
    values = getattr(model, "classes_", None)
    if values is None:
        raise ModelArtifactError(f"{type(model).__name__} does not expose classes_.")
    return [value.item() if hasattr(value, "item") else value for value in values]


def _assert_runtime_versions(registry: dict[str, Any]) -> None:
    required = registry["required_runtime"]
    actual = {"scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__}
    mismatches = [
        f"{name}: required {required[name]}, installed {actual[name]}"
        for name in required
        if actual[name] != required[name]
    ]
    if mismatches:
        raise ModelArtifactError(
            "Refusing to load the serialized model files under a different library version. "
            "Create/activate the project's pinned virtual environment first.\n- "
            + "\n- ".join(mismatches)
        )


def _validate_one_model(
    model: Any,
    approach_key: str,
    model_key: str,
    expected_features: list[str],
    canonical_diseases: list[str],
    star_disease_labels: list[str],
) -> list[str]:
    feature_names = _feature_names(model)
    if feature_names != expected_features:
        raise ModelArtifactError(
            f"Feature-order mismatch for {approach_key}/{model_key}. "
            "The saved model's feature_names_in_ differs from the audited configuration."
        )
    actual_count = _model_feature_count(model)
    if actual_count is not None and actual_count != len(expected_features):
        raise ModelArtifactError(
            f"Feature-count mismatch for {approach_key}/{model_key}: "
            f"model has {actual_count}, configuration has {len(expected_features)}."
        )

    raw_classes = _classes(model)
    if len(raw_classes) != len(canonical_diseases):
        raise ModelArtifactError(
            f"Class-count mismatch for {approach_key}/{model_key}: {len(raw_classes)}."
        )

    if approach_key == "stable_fci":
        mapped_classes = [str(value) for value in raw_classes]
    elif approach_key == "star_ci" and model_key == "xgboost":
        expected_indices = list(range(len(canonical_diseases)))
        if [int(value) for value in raw_classes] != expected_indices:
            raise ModelArtifactError("STAR XGBoost classes_ is not the expected 0..40 integer sequence.")
        mapped_classes = canonical_diseases.copy()
    else:
        mapped_classes = [str(value).removeprefix("disease_") for value in raw_classes]

    if mapped_classes != canonical_diseases:
        raise ModelArtifactError(
            f"Disease-class ordering mismatch for {approach_key}/{model_key}."
        )
    return mapped_classes


def load_all_models(project_root: str | Path, *, enforce_versions: bool = True) -> dict[str, Any]:
    """Load and validate the six existing models. Does not fit or alter them."""
    root = Path(project_root).resolve()
    configs = load_configurations(root)
    registry = configs["model_registry"]
    manifest_path = root / "artifact_manifest.json"
    if not manifest_path.is_file():
        raise ModelArtifactError(f"Missing artifact manifest: {manifest_path}")
    import json
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_by_path = {item["path"]: item for item in manifest.get("models", [])}

    if enforce_versions:
        _assert_runtime_versions(registry)

    register_stable_fci_wrapper_for_unpickling()
    canonical_diseases = [item["display_name"] for item in configs["disease_labels"]["classes"]]
    star_labels = [item["star_ci_model_label"] for item in configs["disease_labels"]["classes"]]

    loaded: dict[str, Any] = {
        "models": {"stable_fci": {}, "star_ci": {}},
        "class_names": {"stable_fci": {}, "star_ci": {}},
        "configurations": configs,
    }

    for approach_key, approach_spec in registry["approaches"].items():
        feature_config_key = approach_spec["feature_config"].removesuffix(".json")
        expected_features = configs[feature_config_key]["feature_names"]
        for model_key, model_spec in approach_spec["models"].items():
            artifact_path = root / model_spec["path"]
            if not artifact_path.is_file():
                raise ModelArtifactError(f"Missing model artifact: {artifact_path}")
            manifest_key = model_spec["path"]
            manifest_item = manifest_by_path.get(manifest_key)
            if manifest_item is None:
                raise ModelArtifactError(f"No SHA-256 manifest entry for {manifest_key}")
            actual_hash = _sha256(artifact_path)
            if actual_hash != manifest_item["sha256"]:
                raise ModelArtifactError(f"SHA-256 mismatch for model artifact {manifest_key}")

            with warnings.catch_warnings():
                # Version differences are handled before loading; do not suppress other warnings.
                warnings.simplefilter("default")
                model = joblib.load(artifact_path)

            mapped_classes = _validate_one_model(
                model,
                approach_key,
                model_key,
                expected_features,
                canonical_diseases,
                star_labels,
            )
            loaded["models"][approach_key][model_key] = model
            loaded["class_names"][approach_key][model_key] = mapped_classes

    return loaded
