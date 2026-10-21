from datetime import date, datetime, timedelta
from types import SimpleNamespace

import polars as pl
import pytest

from airports.models import AirportCatalogProvenance
from flights.evaluation import evaluation
from flights.evaluation.cache import EvaluationCache
from flights.evaluation.models import EvaluationResults, EvaluationStats
from flights.flight_type import Flight, flights_to_flights_df
from flights.sources import BTS_SOURCE, OPENSKY_SOURCE, SFDPS_SOURCE


def _flight(
    icao: str,
    takeoff_time: datetime,
    *,
    takeoff_airport_id: int | None = 1,
    landing_airport_id: int | None = 2,
) -> Flight:
    return Flight(
        icao=icao,
        takeoff_time=takeoff_time,
        landing_time=takeoff_time + timedelta(hours=1),
        takeoff_airport_ident="KAAA",
        landing_airport_ident="KBBB",
        takeoff_airport_id=takeoff_airport_id,
        landing_airport_id=landing_airport_id,
    )


def _airport_catalog(target_date: date):
    return SimpleNamespace(
        airports=(),
        provenance=AirportCatalogProvenance(
            flight_date=target_date,
            observed_commit_sha="observed-commit",
            knowledge_cutoff_date=date(2026, 4, 1),
            knowledge_commit_sha="knowledge-commit",
            catalog_available_date=date(2026, 4, 1),
            policy_version="test-policy",
        ),
    )


def test_compare_day_filters_flights_to_target_date(monkeypatch):
    target_date = date(2026, 3, 1)
    start = datetime(2026, 3, 1)
    end = start + timedelta(days=1)
    flights = flights_to_flights_df([
        _flight("previous-day", start - timedelta(microseconds=1)),
        _flight("start", start),
        _flight("end", end - timedelta(hours=1, microseconds=1)),
        _flight("overnight", end - timedelta(hours=1)),
        _flight("next-day", end),
    ])
    catalog = _airport_catalog(target_date)

    def compare(test, gold, start_dt, end_dt, airport_catalog, **kwargs):
        assert test.get_column("icao").to_list() == ["start", "end"]
        assert gold.get_column("icao").to_list() == ["start", "end"]
        assert start_dt == start
        assert end_dt == end
        assert airport_catalog is catalog
        return test

    monkeypatch.setattr(evaluation, "df_flights_comparison", compare)
    evaluation._compare_day(flights, flights, target_date, catalog)


def test_evaluate_keeps_gold_only_icaos_and_excludes_test_only(
    monkeypatch,
    tmp_path,
):
    target_date = date(2026, 3, 1)
    noon = datetime(2026, 3, 1, 12)
    source_frames = {
        BTS_SOURCE.name: flights_to_flights_df(
            [
                _flight("shared", noon),
                _flight("shared", noon + timedelta(hours=3)),
                _flight("test-only", noon + timedelta(hours=5)),
                _flight("shared", noon + timedelta(days=1)),
            ]
        ),
        SFDPS_SOURCE.name: flights_to_flights_df(
            [
                _flight("shared", noon),
                _flight("gold-only", noon + timedelta(hours=5)),
            ]
        ),
    }

    monkeypatch.setattr(
        evaluation,
        "source_flights_reader",
        lambda source, adsb_source=None: (
            lambda _: source_frames[source.name].clone()
        ),
    )
    monkeypatch.setattr(
        evaluation,
        "airport_catalog_for_date",
        _airport_catalog,
    )

    results = evaluation.evaluate(
        test_source=BTS_SOURCE,
        gold_source=SFDPS_SOURCE,
        test_dates=target_date,
        evaluation_cache=EvaluationCache(tmp_path / "cache"),
    )

    assert isinstance(results, EvaluationResults)
    assert len(results.entries) == 1
    comparison = results.entries[0].comparison
    assert comparison.get_column("icao").unique().sort().to_list() == [
        "gold-only",
        "shared",
    ]
    assert (
        comparison.filter(pl.col("icao") == "gold-only")
        .get_column("match_status")
        .to_list()
        == ["df1_only"]
    )
    assert "test-only" not in comparison.get_column("icao").to_list()
    expected_metrics = {
        "gold_icao_count": 2,
        "gold_icao_missing_from_test_count": 1,
        "gold_icao_missing_from_test_pct": 0.5,
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "fully_correct_precision": 0.5,
        "fully_correct_recall": 0.5,
        "fully_correct_f1": 0.5,
        "takeoff_airport_id_total_count": 1,
        "takeoff_airport_id_match_count": 1,
        "takeoff_airport_id_incorrect_match_count": 0,
        "takeoff_airport_id_match_pct": 1.0,
        "landing_airport_id_total_count": 1,
        "landing_airport_id_match_count": 1,
        "landing_airport_id_incorrect_match_count": 0,
        "landing_airport_id_match_pct": 1.0,
        "airport_id_total_count": 1,
        "airport_id_match_count": 1,
        "airport_id_incorrect_match_count": 0,
        "airport_id_match_pct": 1.0,
        "median_max_endpoint_time_error_minutes": 0.0,
        "p90_max_endpoint_time_error_minutes": 0.0,
        "both_endpoints_within_5m_recall": 0.5,
        "both_endpoints_within_15m_recall": 0.5,
        "both_endpoints_within_30m_recall": 0.5,
        "takeoff_time_bias_minutes": 0.0,
        "landing_time_bias_minutes": 0.0,
    }
    stats = results.entries[0].stats
    assert isinstance(stats, EvaluationStats)
    row = stats.to_row()
    assert {key: row[key] for key in expected_metrics} == pytest.approx(
        expected_metrics
    )


def test_opensky_null_airports_are_reported_and_cached(monkeypatch, tmp_path, capsys):
    target_date = date(2026, 3, 1)
    noon = datetime(2026, 3, 1, 12)
    frames = {
        OPENSKY_SOURCE.name: flights_to_flights_df([
            _flight("shared", noon, takeoff_airport_id=None),
            _flight("shared", noon + timedelta(hours=3), landing_airport_id=None),
            _flight("shared", noon + timedelta(hours=6), landing_airport_id=None),
            _flight("test-only", noon, takeoff_airport_id=None),
            _flight("shared", noon + timedelta(days=1), takeoff_airport_id=None),
        ]),
        BTS_SOURCE.name: flights_to_flights_df([
            _flight("shared", noon),
            _flight("shared", noon + timedelta(hours=3)),
            _flight("gold-only", noon),
        ]),
    }
    monkeypatch.setattr(
        evaluation, "source_flights_reader",
        lambda source, adsb_source=None: lambda _: frames[source.name].clone(),
    )
    monkeypatch.setattr(evaluation, "airport_catalog_for_date", _airport_catalog)
    cache = EvaluationCache(tmp_path / "cache")
    results = evaluation.evaluate(
        OPENSKY_SOURCE, BTS_SOURCE, target_date, evaluation_cache=cache,
    )
    stats = results.stats[0]
    assert stats.true_positive == 2
    assert stats.false_positive == 1
    assert stats.false_negative == 1
    assert stats.takeoff_airport_id_null_count == 1
    assert stats.landing_airport_id_null_count == 2
    assert stats.takeoff_airport_id_null_pct == pytest.approx(1 / 3)
    assert stats.landing_airport_id_null_pct == pytest.approx(2 / 3)
    assert stats.takeoff_airport_id_non_null_count == 1
    assert stats.landing_airport_id_non_null_count == 1
    assert stats.takeoff_airport_id_non_null_match_pct == 1.0
    assert stats.landing_airport_id_non_null_match_pct == 1.0
    assert stats.median_max_endpoint_time_error_minutes == 0.0
    assert stats.p90_max_endpoint_time_error_minutes == 0.0
    assert stats.both_endpoints_within_5m_recall == pytest.approx(2 / 3)
    assert stats.both_endpoints_within_15m_recall == pytest.approx(2 / 3)
    assert stats.both_endpoints_within_30m_recall == pytest.approx(2 / 3)
    assert stats.takeoff_time_bias_minutes == 0.0
    assert stats.landing_time_bias_minutes == 0.0
    report = (
        "Null airports in opensky (3 retained test flights): "
        "takeoff 1 (33.33%), landing 2 (66.67%)"
    )
    accuracy_report = (
        "Non-null airport accuracy in opensky (matched flights): "
        "takeoff 100.00%, landing 100.00%"
    )
    output = capsys.readouterr().out
    assert report in output
    assert accuracy_report in output

    def unexpected_read(*args, **kwargs):
        raise AssertionError("Cached evaluation should not read source flights")

    monkeypatch.setattr(evaluation, "source_flights_reader", unexpected_read)
    cached = evaluation.evaluate(
        OPENSKY_SOURCE, BTS_SOURCE, target_date, evaluation_cache=cache,
    )
    assert cached.stats == results.stats
    output = capsys.readouterr().out
    assert report in output
    assert accuracy_report in output
    _, path = evaluation.export_cached_evaluations(
        tmp_path / "stats.csv", test_source="opensky", evaluation_cache=cache,
    )
    row = pl.read_csv(path).row(0, named=True)
    for endpoint, count in (("takeoff", 1), ("landing", 2)):
        assert row[f"{endpoint}_airport_id_null_count"] == count
        assert row[f"{endpoint}_airport_id_null_pct"] == pytest.approx(count / 3)
        assert row[f"{endpoint}_airport_id_non_null_count"] == 1
        assert row[f"{endpoint}_airport_id_non_null_match_pct"] == 1.0
    assert row["median_max_endpoint_time_error_minutes"] == 0.0
    assert row["p90_max_endpoint_time_error_minutes"] == 0.0
    assert row["both_endpoints_within_15m_recall"] == pytest.approx(2 / 3)


def test_gold_icao_coverage_uses_unique_input_aircraft_before_matching(
    monkeypatch, tmp_path, capsys,
):
    target_date = date(2026, 3, 1)
    noon = datetime(2026, 3, 1, 12)
    frames = {
        BTS_SOURCE.name: flights_to_flights_df([
            _flight("shared", noon),
            _flight("present-unmatched", noon),
            _flight("missing", noon + timedelta(days=1)),
        ]),
        SFDPS_SOURCE.name: flights_to_flights_df([
            _flight("shared", noon),
            _flight("present-unmatched", noon + timedelta(hours=3)),
            _flight("missing", noon),
            _flight("missing", noon + timedelta(hours=3)),
            _flight("boundary-only", datetime(2026, 3, 1, 0, 10)),
            _flight("other-day", noon + timedelta(days=1)),
        ]),
    }
    monkeypatch.setattr(
        evaluation, "source_flights_reader",
        lambda source, adsb_source=None: lambda _: frames[source.name].clone(),
    )
    monkeypatch.setattr(evaluation, "airport_catalog_for_date", _airport_catalog)
    cache = EvaluationCache(tmp_path / "cache")
    results = evaluation.evaluate(
        BTS_SOURCE, SFDPS_SOURCE, target_date, evaluation_cache=cache,
    )
    stats = results.stats[0]
    assert stats.gold_icao_count == 4
    assert stats.gold_icao_missing_from_test_count == 2
    assert stats.gold_icao_missing_from_test_pct == 0.5
    assert "boundary-only" not in results.comparisons[0]["icao"].to_list()
    assert results.to_frame().item(0, "gold_icao_missing_from_test_pct") == 0.5
    assert "Gold ICAOs absent from test: 2/4 (50.00%)" in capsys.readouterr().out

    def unexpected_read(*args, **kwargs):
        raise AssertionError("Cached evaluation should not read source flights")

    monkeypatch.setattr(evaluation, "source_flights_reader", unexpected_read)
    cached = evaluation.evaluate(
        BTS_SOURCE, SFDPS_SOURCE, target_date, evaluation_cache=cache,
    )
    assert cached.stats == results.stats
    assert "Gold ICAOs absent from test: 2/4 (50.00%)" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("null_source", "source_role"),
    [
        (BTS_SOURCE, "test_source"),
        (SFDPS_SOURCE, "gold_source"),
    ],
)
def test_evaluate_asserts_icaos_are_not_null(
    monkeypatch,
    tmp_path,
    null_source,
    source_role,
):
    target_date = date(2026, 3, 1)
    noon = datetime(2026, 3, 1, 12)
    valid_frame = flights_to_flights_df([_flight("abc123", noon)])
    null_icao_frame = valid_frame.with_columns(
        pl.lit(None, dtype=pl.String).alias("icao")
    )
    source_frames = {
        BTS_SOURCE.name: valid_frame,
        SFDPS_SOURCE.name: valid_frame,
    }
    source_frames[null_source.name] = null_icao_frame

    monkeypatch.setattr(
        evaluation,
        "source_flights_reader",
        lambda source, adsb_source=None: (
            lambda _: source_frames[source.name].clone()
        ),
    )
    monkeypatch.setattr(
        evaluation,
        "airport_catalog_for_date",
        _airport_catalog,
    )

    with pytest.raises(
        AssertionError,
        match=rf"{source_role} contains 1 null icao value",
    ):
        evaluation.evaluate(
            test_source=BTS_SOURCE,
            gold_source=SFDPS_SOURCE,
            test_dates=target_date,
            evaluation_cache=EvaluationCache(tmp_path / "cache"),
        )


@pytest.mark.parametrize(
    ("empty_source", "source_role"),
    [
        (BTS_SOURCE, "test_source"),
        (SFDPS_SOURCE, "gold_source"),
    ],
)
def test_evaluate_asserts_flight_sources_are_not_empty(
    monkeypatch,
    tmp_path,
    empty_source,
    source_role,
):
    target_date = date(2026, 3, 1)
    noon = datetime(2026, 3, 1, 12)
    valid_frame = flights_to_flights_df([_flight("abc123", noon)])
    source_frames = {
        BTS_SOURCE.name: valid_frame,
        SFDPS_SOURCE.name: valid_frame,
    }
    source_frames[empty_source.name] = flights_to_flights_df([])

    monkeypatch.setattr(
        evaluation,
        "source_flights_reader",
        lambda source, adsb_source=None: (
            lambda _: source_frames[source.name].clone()
        ),
    )
    monkeypatch.setattr(
        evaluation,
        "airport_catalog_for_date",
        _airport_catalog,
    )

    with pytest.raises(
        AssertionError,
        match=rf"{source_role} contains no flights for {target_date}",
    ):
        evaluation.evaluate(
            test_source=BTS_SOURCE,
            gold_source=SFDPS_SOURCE,
            test_dates=target_date,
            evaluation_cache=EvaluationCache(tmp_path / "cache"),
        )
