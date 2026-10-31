from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def load_configurations(project_root: str | Path) -> dict[str, dict[str, Any]]:
    root = Path(project_root)
    config_dir = root / "configurations"
    names = [
        "master_symptoms.json",
        "stable_fci_features.json",
        "star_ci_features.json",
        "disease_labels.json",
        "model_registry.json",
    ]
    return {name.removesuffix(".json"): load_json(config_dir / name) for name in names}
