import csv
from datetime import date
from pathlib import Path

import pytest

from flights.evaluation.evaluation import main
from flights.flights_comparison import df_flights_comparison_stats


EVALUATION_STATS_PATH = (
    Path(__file__).parent / "2026-06-26_22-12_36ff961_flights_evaluation.csv"
)
EVALUATION_MATRIX_PATH = (
    Path(__file__).parent / "2026-07-19_16-29_6da3710_flights_evaluation.csv"
)
BEST_STATS = {
    "true_positive",
    "precision",
    "recall",
    "f1",
    "airport_ident_match_pct"
}
WORST_STATS = {"false_positive", "false_negative"}


def load_daily_stats_bounds() -> dict[date, dict[str, float]]:
    daily_stats = {}
    with EVALUATION_STATS_PATH.open() as f:
        for row in csv.DictReader(f):
            if row["test_date"] == "average":
                continue
            daily_stats[date.fromisoformat(row["test_date"])] = {
                key: float(row[key])
                for key in BEST_STATS | WORST_STATS
            }
    return daily_stats


DAILY_STATS_BOUNDS = load_daily_stats_bounds()


def load_evaluation_matrix() -> list[dict[str, str]]:
    with EVALUATION_MATRIX_PATH.open() as f:
        return [
            row
            for row in csv.DictReader(f)
            if row["test_date"] != "average"
        ]


EVALUATION_MATRIX = load_evaluation_matrix()


def assert_stats_meet_bounds(
    stats: dict[str, int | float],
    expected_stats: dict[str, float],
) -> None:
    for key in BEST_STATS:
        bound = expected_stats[key]
        actual = stats[key]
        assert actual >= bound, f"{key}={actual} should be >= best minimum {bound}"

    for key in WORST_STATS:
        bound = expected_stats[key]
        actual = stats[key]
        assert actual <= bound, f"{key}={actual} should be <= worst maximum {bound}"


@pytest.mark.parametrize(
    "single_test_date",
    sorted(DAILY_STATS_BOUNDS),
    ids=lambda test_date: test_date.isoformat(),
)
def test_model_to_sfdps_flight_evaluation_meets_daily_bounds(single_test_date: date):
    results, _ = main(test_src="algorithm", gold_src="sfdps", test_dates=single_test_date, pia_or_american_ladd_only=True)

    assert len(results) == 1
    assert_stats_meet_bounds(
        df_flights_comparison_stats(results[0]),
        DAILY_STATS_BOUNDS[single_test_date],
    )


@pytest.mark.parametrize(
    "expected",
    EVALUATION_MATRIX,
    ids=lambda row: "-".join(
        value
        for value in (
            row["test_date"],
            row["adsb_src"] or row["test_src"],
            row["test_src"],
            row["gold_src"],
        )
    ),
)
def test_flight_evaluation_matrix_meets_recorded_bounds(expected: dict[str, str]):
    results, _ = main(
        test_src=expected["test_src"],
        gold_src=expected["gold_src"],
        test_dates=date.fromisoformat(expected["test_date"]),
        adsb_src=expected["adsb_src"] or "adsblol",
        matching_adsb_src=expected["matching_adsb_src"],
        use_all_icaos=expected["use_all_icaos"] == "true",
        filter_rotorcraft=expected["filter_rotorcraft"] == "true",
        pia_or_american_ladd_only=(
            expected["pia_or_american_ladd_only"] == "true"
        ),
    )

    assert len(results) == 1
    assert_stats_meet_bounds(
        df_flights_comparison_stats(results[0]),
        {
            key: float(expected[key])
            for key in BEST_STATS | WORST_STATS
        },
    )

def test_algorithm_flights_recreate_meets_daily_bounds():
    dt = date(2026,3,1)
    from flights_algorithm.main import run_main
    run_main(target_date = dt)
    results, _ = main(test_src="algorithm", gold_src="sfdps", test_dates=dt, pia_or_american_ladd_only=True)
    assert len(results) == 1
    assert_stats_meet_bounds(df_flights_comparison_stats(results[0]), DAILY_STATS_BOUNDS[dt])
