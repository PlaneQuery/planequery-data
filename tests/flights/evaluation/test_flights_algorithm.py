import csv
from datetime import date
from pathlib import Path

import pytest

from data_engineering.adsb.read_adsb import ADSBSource
from flights.evaluation.evaluation import main
from flights.flights_comparison import df_flights_comparison_stats
from flights.sources import (
    ALGORITHM_NEAREST_SOURCE,
    SFDPS_SOURCE,
)


EVALUATION_STATS_PATH = (
    Path(__file__).parent / "2026-06-26_22-12_36ff961_flights_evaluation.csv"
)
BEST_STATS = {
    "true_positive",
    "precision",
    "recall",
    "f1",
    "airport_id_match_pct"
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
    results, _ = main(
        test_src=ALGORITHM_NEAREST_SOURCE,
        gold_src=SFDPS_SOURCE,
        test_dates=single_test_date,
        pia_or_american_ladd_only=True,
    )

    assert len(results) == 1
    assert_stats_meet_bounds(
        df_flights_comparison_stats(results[0]),
        DAILY_STATS_BOUNDS[single_test_date],
    )


def test_algorithm_flights_recreate_meets_daily_bounds():
    dt = date(2026, 3, 1)
    from flights.flight_source import write_flights

    write_flights(ALGORITHM_NEAREST_SOURCE, dt, ADSBSource.ADSBLOL)
    results, _ = main(
        test_src=ALGORITHM_NEAREST_SOURCE,
        gold_src=SFDPS_SOURCE,
        test_dates=dt,
        pia_or_american_ladd_only=True,
    )
    assert len(results) == 1
    assert_stats_meet_bounds(
        df_flights_comparison_stats(results[0]),
        DAILY_STATS_BOUNDS[dt],
    )
