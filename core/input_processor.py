from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


class SymptomInputError(ValueError):
    """Raised when a selected symptom is unknown or duplicated."""


def _master_feature_names(master_config: dict[str, Any]) -> list[str]:
    items = master_config.get("symptoms", [])
    names = [item.get("feature_name") for item in items]
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Master symptom configuration contains invalid feature names.")
    if len(names) != len(set(names)):
        raise ValueError("Master symptom configuration contains duplicate internal feature names.")
    if len(names) != master_config.get("feature_count"):
        raise ValueError("Master symptom feature_count does not match the symptom list.")
    return names


def create_master_input(
    selected_symptoms: Iterable[str],
    master_config: dict[str, Any],
) -> pd.DataFrame:
    """Create a 1-row binary input over all original symptoms in source order.

    Checked symptoms are encoded as 1; unchecked symptoms are encoded as 0.
    Names are matched exactly. The function never strips, renames, or merges IDs.
    """
    if isinstance(selected_symptoms, (str, bytes)):
        raise SymptomInputError("selected_symptoms must be an iterable of exact symptom names, not a single string.")

    selected_list = list(selected_symptoms)
    if len(selected_list) != len(set(selected_list)):
        raise SymptomInputError("The selected symptom list contains duplicate entries.")

    master_names = _master_feature_names(master_config)
    master_set = set(master_names)
    unknown = [name for name in selected_list if name not in master_set]
    if unknown:
        raise SymptomInputError(f"Unknown symptom identifier(s): {unknown}")

    selected_set = set(selected_list)
    values = [[1 if name in selected_set else 0 for name in master_names]]
    return pd.DataFrame(values, columns=master_names, dtype="int8")


def prepare_approach_input(
    master_input: pd.DataFrame,
    feature_config: dict[str, Any],
) -> pd.DataFrame:
    """Project the 132-column input into one approach's exact model feature order."""
    expected = feature_config.get("feature_names", [])
    declared_count = feature_config.get("feature_count")
    if not expected or len(expected) != declared_count:
        raise ValueError("Approach feature configuration has an inconsistent feature count.")
    if len(expected) != len(set(expected)):
        raise ValueError("Approach feature list contains duplicate feature identifiers.")

    missing_from_master = [name for name in expected if name not in master_input.columns]
    if missing_from_master:
        raise ValueError(f"Approach feature(s) missing from master input: {missing_from_master}")

    projected = master_input.loc[:, expected].copy()
    if projected.shape != (1, declared_count):
        raise ValueError(
            f"Expected one input row with {declared_count} features; got {projected.shape}."
        )
    return projected


def prepare_inputs_for_both(
    selected_symptoms: Iterable[str],
    configurations: dict[str, dict[str, Any]],
) -> dict[str, pd.DataFrame]:
    master = create_master_input(selected_symptoms, configurations["master_symptoms"])
    return {
        "master": master,
        "stable_fci": prepare_approach_input(master, configurations["stable_fci_features"]),
        "star_ci": prepare_approach_input(master, configurations["star_ci_features"]),
    }
