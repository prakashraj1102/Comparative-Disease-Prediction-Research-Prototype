from core.result_comparison import compare_top_predictions


def item(rank, disease, percent):
    return {
        "rank": rank,
        "disease": disease,
        "model_score_percent": percent,
    }


def test_compare_top_predictions_keeps_pipeline_scores_separate():
    stable = [item(1, "Disease A", 70.0), item(2, "Disease B", 20.0), item(3, "Disease C", 10.0)]
    star = [item(1, "Disease A", 60.0), item(2, "Disease D", 25.0), item(3, "Disease B", 15.0)]

    result = compare_top_predictions(stable, star)

    assert result["top_prediction_agrees"] is True
    assert result["common_top_k_diseases"] == ["Disease A", "Disease B"]
    assert result["common_top_k_count"] == 2
    row_a = next(row for row in result["rows"] if row["disease"] == "Disease A")
    assert row_a["stable_fci_score_percent"] == 70.0
    assert row_a["star_ci_score_percent"] == 60.0


def test_compare_top_predictions_reports_disagreement_and_star_only_order():
    stable = [item(1, "Disease A", 70.0)]
    star = [item(1, "Disease B", 80.0), item(2, "Disease C", 20.0)]

    result = compare_top_predictions(stable, star)

    assert result["top_prediction_agrees"] is False
    assert [row["disease"] for row in result["rows"]] == [
        "Disease A", "Disease B", "Disease C"
    ]
    assert result["common_top_k_count"] == 0
