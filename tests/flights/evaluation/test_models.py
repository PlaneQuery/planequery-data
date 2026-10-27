from dataclasses import replace
from datetime import date, datetime, timedelta

import polars as pl
import pytest

from data_engineering.adsb.read_adsb import ADSBSource
from flights.evaluation.models import (
    EVALUATION_STATS_SCHEMA,
    EvaluationResult,
    EvaluationResults,
    EvaluationStats,
)


def _timing_comparison(
    takeoff_errors_minutes: list[float],
    landing_errors_minutes: list[float],
    *,
    statuses: list[str] | None = None,
) -> pl.DataFrame:
    if statuses is None:
        statuses = ["both"] * len(takeoff_errors_minutes)
    assert len(takeoff_errors_minutes) == len(landing_errors_minutes)
    assert len(statuses) == len(takeoff_errors_minutes)

    test_takeoffs: list[datetime | None] = []
    test_landings: list[datetime | None] = []
    gold_takeoffs: list[datetime | None] = []
    gold_landings: list[datetime | None] = []
    for index, (status, takeoff_error, landing_error) in enumerate(
        zip(statuses, takeoff_errors_minutes, landing_errors_minutes)
    ):
        gold_takeoff = datetime(2026, 3, 1) + timedelta(hours=3 * index)
        gold_landing = gold_takeoff + timedelta(hours=1)
        test_takeoffs.append(
            None
            if status == "df1_only"
            else gold_takeoff + timedelta(minutes=takeoff_error)
        )
        test_landings.append(
            None
            if status == "df1_only"
            else gold_landing + timedelta(minutes=landing_error)
        )
        gold_takeoffs.append(None if status == "df0_only" else gold_takeoff)
        gold_landings.append(None if status == "df0_only" else gold_landing)

    datetime_type = pl.Datetime("us")
    return pl.DataFrame({
        "match_status": pl.Series(statuses, dtype=pl.String),
        "takeoff_time": pl.Series(test_takeoffs, dtype=datetime_type),
        "landing_time": pl.Series(test_landings, dtype=datetime_type),
        "takeoff_time_df1": pl.Series(gold_takeoffs, dtype=datetime_type),
        "landing_time_df1": pl.Series(gold_landings, dtype=datetime_type),
    })


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
    comparison = _timing_comparison(
        [5, -10, 0, 0],
        [5, 10, 0, 0],
        statuses=["both", "both", "df0_only", "df1_only"],
    ).with_columns(
        pl.Series("same_takeoff_airport_id", [True, False, False, False]),
        pl.Series("same_landing_airport_id", [True, True, False, False]),
        pl.Series("same_airport_id", [True, False, False, False]),
        pl.Series("takeoff_airport_id", [1, None, None, None], dtype=pl.Int64),
        pl.Series("landing_airport_id", [2, 2, None, None], dtype=pl.Int64),
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
    assert stats.median_max_endpoint_time_error_minutes == 7.5
    assert stats.p90_max_endpoint_time_error_minutes == pytest.approx(9.5)
    assert stats.both_endpoints_within_5m_recall == pytest.approx(1 / 3)
    assert stats.both_endpoints_within_15m_recall == pytest.approx(2 / 3)
    assert stats.both_endpoints_within_30m_recall == pytest.approx(2 / 3)
    assert stats.takeoff_time_bias_minutes == pytest.approx(-2.5)
    assert stats.landing_time_bias_minutes == pytest.approx(7.5)
    assert EvaluationStats.from_row(stats.to_row()) == stats


def test_non_null_accuracy_excludes_null_and_unmatched_test_airports():
    comparison = _timing_comparison(
        [0] * 5,
        [0] * 5,
        statuses=["both", "both", "both", "df0_only", "df1_only"],
    ).with_columns(
        pl.Series("takeoff_airport_id", [1, 9, None, 1, None], dtype=pl.Int64),
        pl.Series("landing_airport_id", [2, None, 9, None, None], dtype=pl.Int64),
        pl.Series("same_takeoff_airport_id", [True, False, False, False, False]),
        pl.Series("same_landing_airport_id", [True, False, False, False, False]),
        pl.Series("same_airport_id", [True, False, False, False, False]),
    )
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
    comparison = _timing_comparison(
        [0] * len(statuses),
        [0] * len(statuses),
        statuses=statuses,
    ).with_columns(
        pl.Series("takeoff_airport_id", [None] * len(statuses), dtype=pl.Int64),
        pl.Series("landing_airport_id", [None] * len(statuses), dtype=pl.Int64),
        pl.Series("same_takeoff_airport_id", [False] * len(statuses), dtype=pl.Boolean),
        pl.Series("same_landing_airport_id", [False] * len(statuses), dtype=pl.Boolean),
        pl.Series("same_airport_id", [False] * len(statuses), dtype=pl.Boolean),
    )
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
    if "both" not in statuses:
        assert stats.median_max_endpoint_time_error_minutes is None
        assert stats.p90_max_endpoint_time_error_minutes is None
        assert stats.takeoff_time_bias_minutes is None
        assert stats.landing_time_bias_minutes is None


def test_evaluation_results_offer_an_explicit_frame_adapter():
    first = EvaluationResult(
        comparison=_timing_comparison([1], [1]),
        stats=replace(
            _stats(date(2026, 2, 1), true_positive=2),
            takeoff_airport_id_null_count=1,
            landing_airport_id_null_count=2,
        ),
    )
    second = EvaluationResult(
        comparison=_timing_comparison(
            [10, 20, 0],
            [10, 20, 0],
            statuses=["both", "both", "df1_only"],
        ),
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
    assert average.item(
        0, "median_max_endpoint_time_error_minutes"
    ) == 10.0
    assert average.item(
        0, "p90_max_endpoint_time_error_minutes"
    ) == pytest.approx(18.0)
    assert average.item(0, "both_endpoints_within_5m_recall") == 0.25
    assert average.item(0, "both_endpoints_within_15m_recall") == 0.5
    assert average.item(0, "both_endpoints_within_30m_recall") == 0.75
    assert average.item(0, "takeoff_time_bias_minutes") == pytest.approx(31 / 3)
    assert average.item(0, "landing_time_bias_minutes") == pytest.approx(31 / 3)


def test_evaluation_averages_do_not_mix_input_windows():
    results = EvaluationResults(entries=(
        EvaluationResult(
            comparison=_timing_comparison([0], [0]),
            stats=_stats(date(2026, 3, 1), true_positive=2),
        ),
        EvaluationResult(
            comparison=_timing_comparison([0], [0]),
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


def test_timing_averages_support_source_pairs_without_adsb():
    results = EvaluationResults(entries=(
        EvaluationResult(
            comparison=_timing_comparison([5], [5]),
            stats=replace(
                _stats(date(2026, 3, 1), true_positive=1),
                adsb_source=None,
            ),
        ),
        EvaluationResult(
            comparison=_timing_comparison([15], [15]),
            stats=replace(
                _stats(date(2026, 3, 2), true_positive=1),
                adsb_source=None,
            ),
        ),
    ))

    average = results.to_frame().filter(pl.col("test_date") == "average")

    assert average.item(0, "adsb_src") is None
    assert average.item(0, "median_max_endpoint_time_error_minutes") == 10.0
    assert average.item(0, "both_endpoints_within_5m_recall") == 0.5


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
