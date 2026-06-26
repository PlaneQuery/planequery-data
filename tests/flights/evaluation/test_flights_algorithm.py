import csv
from datetime import date
from pathlib import Path

import pytest

from evaluation.flights.evaluation import main


EVALUATION_STATS_PATH = (
    Path(__file__).parent / "2026-06-26_22-12_36ff961_flights_evaluation.csv"
)
BEST_STATS = {
    "true_positive",
    "precision",
    "recall",
    "f1",
    "airport_ident_match_pct",
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
    outputs = main(test_src="model", gold_src="sfdps", test_dates=single_test_date, pia_or_american_ladd_only=True)

    assert len(outputs) == 1
    _, stats = outputs[0]
    assert_stats_meet_bounds(stats, DAILY_STATS_BOUNDS[single_test_date])
