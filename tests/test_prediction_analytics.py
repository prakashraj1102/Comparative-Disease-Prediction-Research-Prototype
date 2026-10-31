import pytest

from core.prediction_analytics import analyze_prediction_agreement


def item(rank, disease, score=0.5):
    return {
        "rank": rank,
        "disease": disease,
        "model_score": score,
        "model_score_percent": score * 100,
    }


def test_identical_order_has_full_overlap_and_zero_rank_shift():
    stable = [item(1, "A", .8), item(2, "B", .1), item(3, "C", .05)]
    star = [item(1, "A", .7), item(2, "B", .2), item(3, "C", .05)]

    result = analyze_prediction_agreement(stable, star)

    assert result["top_prediction_agrees"] is True
    assert result["shared_count"] == 3
    assert result["union_count"] == 3
    assert result["jaccard_overlap"] == 1.0
    assert result["exact_rank_matches"] == 3
    assert result["mean_absolute_rank_shift"] == 0.0


def test_same_diseases_in_different_order_reports_rank_shift():
    stable = [item(1, "A"), item(2, "B"), item(3, "C")]
    star = [item(1, "B"), item(2, "A"), item(3, "C")]

    result = analyze_prediction_agreement(stable, star)

    assert result["top_prediction_agrees"] is False
    assert result["shared_count"] == 3
    assert result["exact_rank_matches"] == 1
    assert result["mean_absolute_rank_shift"] == pytest.approx(2 / 3)
    assert result["shared_disease_rank_rows"][0]["star_minus_stable_rank"] == 1


def test_partial_overlap_and_unique_diseases_are_reported():
    stable = [item(1, "A"), item(2, "B"), item(3, "C")]
    star = [item(1, "A"), item(2, "D"), item(3, "E")]

    result = analyze_prediction_agreement(stable, star)

    assert result["shared_diseases"] == ["A"]
    assert result["stable_only_diseases"] == ["B", "C"]
    assert result["star_only_diseases"] == ["D", "E"]
    assert result["union_count"] == 5
    assert result["jaccard_overlap"] == pytest.approx(0.2)


def test_no_overlap_is_handled_without_rank_shift():
    result = analyze_prediction_agreement(
        [item(1, "A"), item(2, "B"), item(3, "C")],
        [item(1, "D"), item(2, "E"), item(3, "F")],
    )

    assert result["shared_count"] == 0
    assert result["jaccard_overlap"] == 0.0
    assert result["mean_absolute_rank_shift"] is None


def test_duplicate_disease_is_rejected():
    with pytest.raises(ValueError, match="duplicate disease"):
        analyze_prediction_agreement(
            [item(1, "A"), item(2, "A")],
            [item(1, "A")],
        )


def test_empty_prediction_lists_return_safe_zero_overlap():
    result = analyze_prediction_agreement([], [])

    assert result["top_prediction_agrees"] is False
    assert result["union_count"] == 0
    assert result["jaccard_overlap"] == 0.0
    assert result["mean_absolute_rank_shift"] is None
    assert result["rank_rows"] == []


def test_duplicate_rank_is_rejected():
    with pytest.raises(ValueError, match="duplicate rank"):
        analyze_prediction_agreement(
            [item(1, "A"), item(1, "B")],
            [item(1, "A")],
        )
