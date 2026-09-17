import polars as pl

from flights_airport_nn.compare_adsbx_nearest import rates


def test_missing_adsbx_endpoint_counts_as_error_in_full_cohort():
    frame = pl.DataFrame(
        {
            "adsbx_covered": [True, False, True],
            "adsbx_correct": [True, False, False],
            "model_correct": [True, True, False],
            "adsbx_model_correct": [True, False, True],
            "adsblol_nearest_correct": [False, True, False],
        }
    )
    result = rates(frame)
    assert result["endpoints"] == 3
    assert result["adsbx_covered"] == 2
    assert result["adsbx_nearest_accuracy_all"] == 1 / 3
    assert result["adsbx_nearest_accuracy_covered"] == 1 / 2
    assert result["adsblol_model_accuracy_all"] == 2 / 3
    assert result["adsblol_model_accuracy_covered"] == 1 / 2
    assert result["adsbx_model_accuracy_all"] == 2 / 3
    assert result["adsbx_model_accuracy_covered"] == 1.0
