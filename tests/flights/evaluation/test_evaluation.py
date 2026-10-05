from datetime import date, datetime, timedelta
from types import SimpleNamespace

import polars as pl
import pytest

from flights.evaluation import evaluation
from flights.flight_type import Flight, flights_to_flights_df
from flights.sources import BTS_SOURCE, SFDPS_SOURCE


def _flight(
    icao: str,
    takeoff_time: datetime,
    *,
    takeoff_airport_id: int = 1,
    landing_airport_id: int = 2,
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


def test_evaluate_keeps_gold_only_icaos_and_excludes_test_only(monkeypatch):
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
        lambda _: SimpleNamespace(airports=()),
    )

    comparisons, stats = evaluation.evaluate(
        test_source=BTS_SOURCE,
        gold_source=SFDPS_SOURCE,
        test_dates=target_date,
        write_outputs=False,
        use_cache=False,
    )

    assert len(comparisons) == 1
    comparison = comparisons[0]
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
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
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
    }
    row = stats.row(0, named=True)
    assert {key: row[key] for key in expected_metrics} == pytest.approx(
        expected_metrics
    )
