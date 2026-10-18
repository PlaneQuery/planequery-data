from dataclasses import replace
from datetime import date

import polars as pl
import pytest

from data_engineering.adsb.read_adsb import ADSBSource
from flights.evaluation.models import (
    EVALUATION_STATS_SCHEMA,
    EvaluationResult,
    EvaluationResults,
    EvaluationStats,
)


def _stats(test_date: date, *, true_positive: int = 2) -> EvaluationStats:
    return EvaluationStats(
        test_date=test_date,
        adsb_source=ADSBSource.ADSBLOL,
        test_source="algorithm-nn",
        gold_source="bts",
        gold_icao_count=4,
        gold_icao_missing_from_test_count=1,
        true_positive=true_positive,
        false_positive=1,
        false_negative=2,
        takeoff_airport_id_match_count=true_positive,
        landing_airport_id_match_count=true_positive - 1,
        airport_id_match_count=true_positive - 1,
        takeoff_airport_id_non_null_count=true_positive,
        landing_airport_id_non_null_count=true_positive,
    )


def test_evaluation_stats_are_explicit_and_round_trip_through_rows():
    stats = _stats(date(2026, 3, 1))

    assert stats.precision == pytest.approx(2 / 3)
    assert stats.recall == pytest.approx(1 / 2)
    assert stats.f1 == pytest.approx(4 / 7)
    assert stats.fully_correct_precision == pytest.approx(1 / 3)
    assert stats.fully_correct_recall == pytest.approx(1 / 4)
    assert stats.fully_correct_f1 == pytest.approx(2 / 7)
    assert stats.airport_id_match_pct == 0.5
    assert stats.gold_icao_missing_from_test_pct == 0.25
    assert stats.takeoff_airport_id_null_pct == 0.0
    assert stats.landing_airport_id_null_pct == 0.0
    assert stats.takeoff_airport_id_non_null_match_pct == 1.0
    assert stats.landing_airport_id_non_null_match_pct == 0.5
    assert EvaluationStats.from_row(stats.to_row()) == stats


def test_evaluation_stats_are_created_directly_from_comparison():
    comparison = pl.DataFrame(
        {
            "match_status": ["both", "both", "df0_only", "df1_only"],
            "same_takeoff_airport_id": [True, False, False, False],
            "same_landing_airport_id": [True, True, False, False],
            "same_airport_id": [True, False, False, False],
            "takeoff_airport_id": [1, None, None, None],
            "landing_airport_id": [2, 2, None, None],
        }
    )

    stats = EvaluationStats.from_comparison(
        comparison,
        test_date=date(2026, 3, 1),
        adsb_source=ADSBSource.ADSBLOL,
        test_source="algorithm-nn",
        gold_source="bts",
        gold_icao_count=3,
        gold_icao_missing_from_test_count=1,
    )

    assert stats.true_positive == 2
    assert stats.false_positive == 1
    assert stats.false_negative == 1
    assert stats.gold_icao_missing_from_test_pct == pytest.approx(1 / 3)
    assert stats.precision == pytest.approx(2 / 3)
    assert stats.recall == pytest.approx(2 / 3)
    assert stats.fully_correct_precision == pytest.approx(1 / 3)
    assert stats.fully_correct_recall == pytest.approx(1 / 3)
    assert stats.fully_correct_f1 == pytest.approx(1 / 3)
    assert stats.takeoff_airport_id_match_pct == 0.5
    assert stats.landing_airport_id_match_pct == 1.0
    assert stats.airport_id_match_pct == 0.5
    assert stats.takeoff_airport_id_null_count == 2
    assert stats.landing_airport_id_null_count == 1
    assert stats.takeoff_airport_id_null_pct == pytest.approx(2 / 3)
    assert stats.landing_airport_id_null_pct == pytest.approx(1 / 3)
    assert stats.takeoff_airport_id_non_null_count == 1
    assert stats.landing_airport_id_non_null_count == 2
    assert stats.takeoff_airport_id_non_null_match_pct == 1.0
    assert stats.landing_airport_id_non_null_match_pct == 1.0
    assert EvaluationStats.from_row(stats.to_row()) == stats


def test_non_null_accuracy_excludes_null_and_unmatched_test_airports():
    comparison = pl.DataFrame({
        "match_status": ["both", "both", "both", "df0_only", "df1_only"],
        "takeoff_airport_id": [1, 9, None, 1, None],
        "landing_airport_id": [2, None, 9, None, None],
        "same_takeoff_airport_id": [True, False, False, False, False],
        "same_landing_airport_id": [True, False, False, False, False],
        "same_airport_id": [True, False, False, False, False],
    })
    stats = EvaluationStats.from_comparison(
        comparison,
        test_date=date(2026, 3, 1),
        adsb_source=None,
        test_source="opensky",
        gold_source="bts",
        gold_icao_count=1,
        gold_icao_missing_from_test_count=0,
    )
    for endpoint in ("takeoff", "landing"):
        assert getattr(stats, f"{endpoint}_airport_id_non_null_count") == 2
        assert getattr(stats, f"{endpoint}_airport_id_non_null_match_pct") == 0.5
        assert getattr(stats, f"{endpoint}_airport_id_match_pct") == pytest.approx(1 / 3)


@pytest.mark.parametrize("statuses", [[], ["df1_only"], ["both", "df0_only"]])
def test_null_airport_metrics_with_empty_or_all_null_test_flights(statuses):
    comparison = pl.DataFrame({
        "match_status": pl.Series(statuses, dtype=pl.String),
        "takeoff_airport_id": pl.Series([None] * len(statuses), dtype=pl.Int64),
        "landing_airport_id": pl.Series([None] * len(statuses), dtype=pl.Int64),
        "same_takeoff_airport_id": pl.Series([False] * len(statuses), dtype=pl.Boolean),
        "same_landing_airport_id": pl.Series([False] * len(statuses), dtype=pl.Boolean),
        "same_airport_id": pl.Series([False] * len(statuses), dtype=pl.Boolean),
    })
    stats = EvaluationStats.from_comparison(
        comparison,
        test_date=date(2026, 3, 1),
        adsb_source=None,
        test_source="opensky",
        gold_source="bts",
        gold_icao_count=1,
        gold_icao_missing_from_test_count=0,
    )
    expected_count = sum(status != "df1_only" for status in statuses)
    assert stats.takeoff_airport_id_null_count == expected_count
    assert stats.landing_airport_id_null_count == expected_count
    assert stats.takeoff_airport_id_null_pct == float(expected_count > 0)
    assert stats.landing_airport_id_null_pct == float(expected_count > 0)
    assert stats.takeoff_airport_id_non_null_count == 0
    assert stats.landing_airport_id_non_null_count == 0
    assert stats.takeoff_airport_id_non_null_match_pct == 0.0
    assert stats.landing_airport_id_non_null_match_pct == 0.0


def test_evaluation_results_offer_an_explicit_frame_adapter():
    first = EvaluationResult(
        comparison=pl.DataFrame({"match_status": ["both"]}),
        stats=replace(
            _stats(date(2026, 2, 1), true_positive=2),
            takeoff_airport_id_null_count=1,
            landing_airport_id_null_count=2,
        ),
    )
    second = EvaluationResult(
        comparison=pl.DataFrame({"match_status": ["both", "both"]}),
        stats=_stats(date(2026, 3, 1), true_positive=4),
    )
    results = EvaluationResults(entries=(first, second))

    assert results.stats == (first.stats, second.stats)
    assert results.comparisons[0].height == 1
    assert results.to_frame(include_averages=False).schema == (
        EVALUATION_STATS_SCHEMA
    )
    average = results.to_frame().filter(pl.col("test_date") == "average")
    assert average.item(0, "true_positive") == 3.0
    assert average.item(0, "takeoff_airport_id_null_count") == 0.5
    assert average.item(0, "landing_airport_id_null_count") == 1.0
    assert average.item(0, "takeoff_airport_id_null_pct") == pytest.approx(1 / 6)
    assert average.item(0, "landing_airport_id_null_pct") == pytest.approx(1 / 3)
    assert average.item(0, "takeoff_airport_id_non_null_count") == 3.0
    assert average.item(0, "landing_airport_id_non_null_count") == 3.0
    assert average.item(0, "takeoff_airport_id_non_null_match_pct") == 1.0
    assert average.item(0, "landing_airport_id_non_null_match_pct") == 0.625


def test_evaluation_averages_do_not_mix_input_windows():
    results = EvaluationResults(entries=(
        EvaluationResult(
            comparison=pl.DataFrame(),
            stats=_stats(date(2026, 3, 1), true_positive=2),
        ),
        EvaluationResult(
            comparison=pl.DataFrame(),
            stats=replace(
                _stats(date(2026, 3, 2), true_positive=4), target_day_adsb=True,
            ),
        ),
    ))
    averages = results.to_frame().filter(pl.col("test_date") == "average")
    assert averages.height == 2
    assert averages.filter(pl.col("target_day_adsb")).item(
        0, "true_positive",
    ) == 4.0
    assert averages.filter(~pl.col("target_day_adsb")).item(
        0, "true_positive",
    ) == 2.0


def test_evaluation_stats_reject_impossible_match_counts():
    with pytest.raises(ValueError, match="cannot exceed true_positive"):
        EvaluationStats(
            test_date=date(2026, 3, 1),
            adsb_source=None,
            test_source="test",
            gold_source="gold",
            gold_icao_count=1,
            gold_icao_missing_from_test_count=0,
            true_positive=1,
            false_positive=0,
            false_negative=0,
            takeoff_airport_id_match_count=2,
            landing_airport_id_match_count=1,
            airport_id_match_count=1,
            takeoff_airport_id_non_null_count=1,
            landing_airport_id_non_null_count=1,
        )
