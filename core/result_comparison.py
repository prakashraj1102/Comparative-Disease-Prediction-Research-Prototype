from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def compare_top_predictions(
    stable_predictions: Sequence[Mapping[str, Any]],
    star_predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare two top-k result lists without combining or averaging scores."""
    stable_by_disease = {str(item["disease"]): item for item in stable_predictions}
    star_by_disease = {str(item["disease"]): item for item in star_predictions}

    # Preserve Stable-FCI rank order, then append STAR-only diseases in STAR rank order.
    ordered_diseases = list(stable_by_disease)
    ordered_diseases.extend(
        disease for disease in star_by_disease if disease not in stable_by_disease
    )

    rows = []
    for disease in ordered_diseases:
        stable_item = stable_by_disease.get(disease)
        star_item = star_by_disease.get(disease)
        rows.append(
            {
                "disease": disease,
                "stable_fci_rank": stable_item["rank"] if stable_item else None,
                "stable_fci_score_percent": (
                    stable_item["model_score_percent"] if stable_item else None
                ),
                "star_ci_rank": star_item["rank"] if star_item else None,
                "star_ci_score_percent": (
                    star_item["model_score_percent"] if star_item else None
                ),
                "present_in_both_top_k": stable_item is not None and star_item is not None,
            }
        )

    stable_top = stable_predictions[0]["disease"] if stable_predictions else None
    star_top = star_predictions[0]["disease"] if star_predictions else None
    common = [
        disease for disease in stable_by_disease if disease in star_by_disease
    ]

    return {
        "stable_top_disease": stable_top,
        "star_top_disease": star_top,
        "top_prediction_agrees": stable_top is not None and stable_top == star_top,
        "common_top_k_diseases": common,
        "common_top_k_count": len(common),
        "rows": rows,
    }
