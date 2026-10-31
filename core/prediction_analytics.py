from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def analyze_prediction_agreement(
    stable_predictions: Sequence[Mapping[str, Any]],
    star_predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Describe top-k overlap and rank alignment without combining model scores.

    Ranks are treated as ordinal positions within each model's own prediction list.
    Probability scores are deliberately not compared or averaged here.
    """
    stable_by_rank = _index_predictions(stable_predictions, "Stable-FCI")
    star_by_rank = _index_predictions(star_predictions, "STAR CI-selected")

    stable_by_disease = {str(item["disease"]): item for item in stable_predictions}
    star_by_disease = {str(item["disease"]): item for item in star_predictions}

    stable_diseases = list(stable_by_disease)
    star_diseases = list(star_by_disease)
    stable_set = set(stable_diseases)
    star_set = set(star_diseases)
    shared = [disease for disease in stable_diseases if disease in star_set]
    stable_only = [disease for disease in stable_diseases if disease not in star_set]
    star_only = [disease for disease in star_diseases if disease not in stable_set]
    union_count = len(stable_set | star_set)
    jaccard = (len(shared) / union_count) if union_count else 0.0

    max_rank = max(
        max(stable_by_rank, default=0),
        max(star_by_rank, default=0),
    )
    rank_rows: list[dict[str, Any]] = []
    exact_rank_matches = 0
    for rank in range(1, max_rank + 1):
        stable_item = stable_by_rank.get(rank)
        star_item = star_by_rank.get(rank)
        same_disease = bool(
            stable_item is not None
            and star_item is not None
            and str(stable_item["disease"]) == str(star_item["disease"])
        )
        exact_rank_matches += int(same_disease)
        rank_rows.append(
            {
                "rank": rank,
                "stable_fci_disease": str(stable_item["disease"]) if stable_item else None,
                "star_ci_disease": str(star_item["disease"]) if star_item else None,
                "same_disease_at_rank": same_disease,
            }
        )

    shared_rank_rows: list[dict[str, Any]] = []
    rank_shifts: list[int] = []
    for disease in shared:
        stable_rank = int(stable_by_disease[disease]["rank"])
        star_rank = int(star_by_disease[disease]["rank"])
        shift = star_rank - stable_rank
        rank_shifts.append(abs(shift))
        shared_rank_rows.append(
            {
                "disease": disease,
                "stable_fci_rank": stable_rank,
                "star_ci_rank": star_rank,
                "star_minus_stable_rank": shift,
                "absolute_rank_shift": abs(shift),
            }
        )

    stable_top = stable_by_rank.get(1)
    star_top = star_by_rank.get(1)
    top_agrees = bool(
        stable_top is not None
        and star_top is not None
        and str(stable_top["disease"]) == str(star_top["disease"])
    )

    return {
        "stable_top_disease": str(stable_top["disease"]) if stable_top else None,
        "star_top_disease": str(star_top["disease"]) if star_top else None,
        "top_prediction_agrees": top_agrees,
        "stable_top_k_count": len(stable_diseases),
        "star_top_k_count": len(star_diseases),
        "shared_diseases": shared,
        "shared_count": len(shared),
        "stable_only_diseases": stable_only,
        "star_only_diseases": star_only,
        "union_count": union_count,
        "jaccard_overlap": jaccard,
        "jaccard_overlap_percent": jaccard * 100.0,
        "exact_rank_matches": exact_rank_matches,
        "rank_positions_compared": max_rank,
        "shared_disease_rank_rows": shared_rank_rows,
        "mean_absolute_rank_shift": (
            sum(rank_shifts) / len(rank_shifts) if rank_shifts else None
        ),
        "rank_rows": rank_rows,
    }


def _index_predictions(
    predictions: Sequence[Mapping[str, Any]], approach_name: str
) -> dict[int, Mapping[str, Any]]:
    by_rank: dict[int, Mapping[str, Any]] = {}
    diseases: set[str] = set()
    for item in predictions:
        try:
            rank = int(item["rank"])
            disease = str(item["disease"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{approach_name} prediction entries must contain a valid rank and disease."
            ) from exc
        if rank < 1:
            raise ValueError(f"{approach_name} ranks must be positive integers.")
        if rank in by_rank:
            raise ValueError(f"{approach_name} contains duplicate rank {rank}.")
        if disease in diseases:
            raise ValueError(f"{approach_name} contains duplicate disease {disease!r}.")
        by_rank[rank] = item
        diseases.add(disease)
    return by_rank
