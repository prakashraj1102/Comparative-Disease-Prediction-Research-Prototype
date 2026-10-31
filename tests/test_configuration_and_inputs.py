from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.input_processor import (
    SymptomInputError,
    create_master_input,
    prepare_approach_input,
    prepare_inputs_for_both,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configurations"


def read_config(name):
    return json.loads((CONFIG_DIR / name).read_text(encoding="utf-8"))


@pytest.fixture
def configs():
    return {
        "master_symptoms": read_config("master_symptoms.json"),
        "stable_fci_features": read_config("stable_fci_features.json"),
        "star_ci_features": read_config("star_ci_features.json"),
        "disease_labels": read_config("disease_labels.json"),
    }


def test_feature_counts_and_uniqueness(configs):
    master = [x["feature_name"] for x in configs["master_symptoms"]["symptoms"]]
    stable = configs["stable_fci_features"]["feature_names"]
    star = configs["star_ci_features"]["feature_names"]
    diseases = configs["disease_labels"]["classes"]
    assert len(master) == 132 and len(set(master)) == 132
    assert len(stable) == 37 and len(set(stable)) == 37
    assert len(star) == 106 and len(set(star)) == 106
    assert len(diseases) == 41
    assert set(stable).issubset(master)
    assert set(star).issubset(master)


def test_empty_selection_produces_all_zero_vectors(configs):
    result = prepare_inputs_for_both([], configs)
    assert result["master"].shape == (1, 132)
    assert result["stable_fci"].shape == (1, 37)
    assert result["star_ci"].shape == (1, 106)
    assert int(result["master"].to_numpy().sum()) == 0
    assert int(result["stable_fci"].to_numpy().sum()) == 0
    assert int(result["star_ci"].to_numpy().sum()) == 0


def test_selected_symptoms_are_projected_without_reordering(configs):
    selected = ["itching", "fluid_overload", "fluid_overload.1"]
    result = prepare_inputs_for_both(selected, configs)
    for key in ("master", "stable_fci", "star_ci"):
        frame = result[key]
        if "fluid_overload" in frame.columns:
            assert int(frame.loc[0, "fluid_overload"]) == 1
        if "fluid_overload.1" in frame.columns:
            assert int(frame.loc[0, "fluid_overload.1"]) == 1
    assert list(result["stable_fci"].columns) == configs["stable_fci_features"]["feature_names"]
    assert list(result["star_ci"].columns) == configs["star_ci_features"]["feature_names"]
    assert int(result["master"].loc[0, "fluid_overload"]) == 1
    assert int(result["master"].loc[0, "fluid_overload.1"]) == 1


def test_unknown_symptoms_are_rejected(configs):
    with pytest.raises(SymptomInputError, match="Unknown symptom"):
        create_master_input(["not_a_real_symptom"], configs["master_symptoms"])


def test_duplicate_selection_entries_are_rejected(configs):
    with pytest.raises(SymptomInputError, match="duplicate entries"):
        create_master_input(["itching", "itching"], configs["master_symptoms"])


def test_star_disease_prefix_mapping_is_exact(configs):
    classes = configs["disease_labels"]["classes"]
    assert len(classes) == 41
    assert all(
        item["star_ci_model_label"].removeprefix("disease_") == item["display_name"]
        for item in classes
    )
