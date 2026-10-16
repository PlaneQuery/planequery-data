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
    )


def test_evaluation_stats_are_explicit_and_round_trip_through_rows():
    stats = _stats(date(2026, 3, 1))

    assert stats.precision == pytest.approx(2 / 3)
    assert stats.recall == pytest.approx(1 / 2)
    assert stats.f1 == pytest.approx(4 / 7)
    assert stats.airport_id_match_pct == 0.5
    assert stats.gold_icao_missing_from_test_pct == 0.25
    assert EvaluationStats.from_row(stats.to_row()) == stats


def test_evaluation_stats_are_created_directly_from_comparison():
    comparison = pl.DataFrame(
        {
            "match_status": ["both", "both", "df0_only", "df1_only"],
            "same_takeoff_airport_id": [True, False, False, False],
            "same_landing_airport_id": [True, True, False, False],
            "same_airport_id": [True, False, False, False],
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
    assert stats.takeoff_airport_id_match_pct == 0.5
    assert stats.landing_airport_id_match_pct == 1.0
    assert stats.airport_id_match_pct == 0.5


def test_evaluation_results_offer_an_explicit_frame_adapter():
    first = EvaluationResult(
        comparison=pl.DataFrame({"match_status": ["both"]}),
        stats=_stats(date(2026, 2, 1), true_positive=2),
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
        )
