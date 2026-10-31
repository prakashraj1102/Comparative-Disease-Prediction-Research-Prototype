from __future__ import annotations

import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configurations"


def read(name):
    with (CONFIG / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)

master = read("master_symptoms.json")
stable = read("stable_fci_features.json")
star = read("star_ci_features.json")
diseases = read("disease_labels.json")
registry = read("model_registry.json")
audit = read("audit_summary.json")
manifest = json.loads((ROOT / "artifact_manifest.json").read_text(encoding="utf-8"))

master_names = [item["feature_name"] for item in master["symptoms"]]
assert len(master_names) == master["feature_count"] == 132
assert len(set(master_names)) == len(master_names)
assert len(stable["feature_names"]) == stable["feature_count"] == 37
assert len(star["feature_names"]) == star["feature_count"] == 106
assert all(name in master_names for name in stable["feature_names"])
assert all(name in master_names for name in star["feature_names"])
assert len(diseases["classes"]) == diseases["class_count"] == 41
assert len({item["display_name"] for item in diseases["classes"]}) == 41
assert all(
    item["star_ci_model_label"].removeprefix("disease_") == item["display_name"]
    for item in diseases["classes"]
)

for item in manifest["models"]:
    artifact = ROOT / item["path"]
    assert artifact.is_file(), f"Missing artifact: {artifact}"
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert digest == item["sha256"], f"SHA-256 mismatch: {artifact}"

print("CONFIGURATION AND ARTIFACT INTEGRITY AUDIT: PASS")
print(f"Master symptoms: {len(master_names)}")
print(f"Stable-FCI features: {len(stable['feature_names'])}")
print(f"STAR CI features: {len(star['feature_names'])}")
print(f"Disease classes: {len(diseases['classes'])}")
print(f"Common selected symptoms: {audit['counts']['common_selected_symptoms']}")
print(f"Stable-FCI-only symptoms: {audit['counts']['stable_fci_only']}")
print(f"STAR-only symptoms: {audit['counts']['star_ci_only']}")
print("Stable and STAR feature identifiers are preserved exactly.")
print(f"Model artifacts hash-verified: {len(manifest['models'])}")
print("No causal discovery or model training was performed.")
