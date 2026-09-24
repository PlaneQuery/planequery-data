from datetime import date, datetime, timedelta
from types import SimpleNamespace

import polars as pl
import pytest

from flights.evaluation.consecutive_airports import (
    consecutive_airport_details,
    consecutive_airport_mismatches,
    consecutive_airport_stats,
)
from planequery_data.utils import haversine


AIRPORT_IDS = {
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "UNKNOWN": 99,
    "KJFK": 10,
    "KLGA": 11,
}


class AirportCatalog:
    id_index = {
        AIRPORT_IDS[ident]: SimpleNamespace(
            id=AIRPORT_IDS[ident],
            lat=lat,
            lon=lon,
        )
        for ident, (lat, lon) in {
            "A": (0.0, 0.0),
            "B": (0.0, 0.1),
            "C": (0.0, 1.0),
            "D": (0.0, 2.0),
            "KJFK": (40.6413, -73.7781),
            "KLGA": (40.7769, -73.8740),
        }.items()
    }
    airports = tuple(id_index.values())


def flight(
    icao: str,
    hour: int,
    takeoff_airport_ident: str | None,
    landing_airport_ident: str | None,
) -> dict:
    takeoff_time = datetime(2026, 1, 1, hour)
    return {
        "icao": icao,
        "takeoff_time": takeoff_time,
        "landing_time": takeoff_time + timedelta(minutes=30),
        "takeoff_airport_id": (
            AIRPORT_IDS[takeoff_airport_ident]
            if takeoff_airport_ident is not None
            else None
        ),
        "takeoff_airport_ident": takeoff_airport_ident,
        "landing_airport_id": (
            AIRPORT_IDS[landing_airport_ident]
            if landing_airport_ident is not None
            else None
        ),
        "landing_airport_ident": landing_airport_ident,
    }


def test_consecutive_airport_stats_are_per_icao_and_time_ordered():
    df_flights = pl.DataFrame(
        [
            flight("abc123", 2, "A", "C"),
            flight("def456", 1, "D", "A"),
            flight("abc123", 1, "A", "A"),
            flight("abc123", 3, "D", "A"),
            flight("abc123", 4, None, "A"),
        ]
    )

    mismatches = consecutive_airport_mismatches(
        df_flights,
        airport_catalog=AirportCatalog(),
    )
    details = consecutive_airport_details(
        df_flights,
        airport_catalog=AirportCatalog(),
    )
    stats = consecutive_airport_stats(
        df_flights,
        airport_catalog=AirportCatalog(),
    )

    assert mismatches.select(
        "previous_landing_airport_ident",
        "takeoff_airport_ident",
    ).rows() == [("C", "D")]
    assert details.select(
        "previous_takeoff_time",
        "previous_landing_time",
        "previous_landing_airport_ident",
        "takeoff_airport_ident",
        "consecutive_airport_mismatch",
    ).rows() == [
        (
            datetime(2026, 1, 1, 1),
            datetime(2026, 1, 1, 1, 30),
            "A",
            "A",
            False,
        ),
        (
            datetime(2026, 1, 1, 2),
            datetime(2026, 1, 1, 2, 30),
            "C",
            "D",
            True,
        ),
    ]
    assert details.get_column(
        "consecutive_airport_displacement_km"
    ).to_list()[0] == 0.0
    expected_distance = haversine(0.0, 1.0, 0.0, 2.0)
    assert mismatches.item(
        0, "consecutive_airport_displacement_km"
    ) == pytest.approx(expected_distance)
    assert stats["consecutive_airport_comparison_count"] == 2
    assert stats["consecutive_airport_mismatch_count"] == 1
    assert stats["consecutive_airport_mismatch_rate"] == pytest.approx(1 / 2)
    assert stats["consecutive_airport_displacement_km_p50"] == pytest.approx(
        expected_distance
    )
    assert stats["consecutive_airport_displacement_km_p90"] == pytest.approx(
        expected_distance
    )
    assert stats["consecutive_airport_displacement_km_p95"] == pytest.approx(
        expected_distance
    )
    assert stats["consecutive_airport_displacement_km_p99"] == pytest.approx(
        expected_distance
    )
    assert stats["consecutive_airport_displacement_km_max"] == pytest.approx(
        expected_distance
    )
    assert stats["consecutive_airport_displacement_over_50km_count"] == 1
    assert stats[
        "consecutive_airport_displacement_over_50km_rate"
    ] == pytest.approx(1 / 2)


def test_missing_airport_id_in_catalog_has_unknown_displacement():
    df_flights = pl.DataFrame(
        [
            flight("abc123", 1, "A", "UNKNOWN"),
            flight("abc123", 2, "B", "C"),
        ]
    )

    details = consecutive_airport_details(
        df_flights,
        airport_catalog=AirportCatalog(),
    )
    stats = consecutive_airport_stats(
        df_flights,
        airport_catalog=AirportCatalog(),
    )

    assert details.get_column(
        "consecutive_airport_displacement_km"
    ).to_list() == [None]
    assert stats["consecutive_airport_comparison_count"] == 1
    assert stats["consecutive_airport_mismatch_count"] == 1
    assert stats["consecutive_airport_displacement_km_max"] is None


def test_missing_airport_idents_are_not_comparable():
    df_flights = pl.DataFrame(
        [
            flight("abc123", 1, "A", None),
            flight("abc123", 2, "B", "C"),
        ]
    )

    stats = consecutive_airport_stats(
        df_flights,
        airport_catalog=AirportCatalog(),
    )

    assert stats["consecutive_airport_comparison_count"] == 0
    assert stats["consecutive_airport_mismatch_count"] == 0
    assert stats["consecutive_airport_mismatch_rate"] == 0.0
    assert stats["consecutive_airport_displacement_km_p50"] is None


def test_custom_displacement_threshold_changes_metric_names():
    df_flights = pl.DataFrame(
        [
            flight("abc123", 1, "A", "A"),
            flight("abc123", 2, "B", "C"),
        ]
    )

    stats = consecutive_airport_stats(
        df_flights,
        displacement_threshold_km=10,
        airport_catalog=AirportCatalog(),
    )

    assert stats["consecutive_airport_displacement_over_10km_count"] == 1
    assert stats["consecutive_airport_displacement_over_10km_rate"] == 1.0


def test_evaluation_main_returns_consecutive_airport_detail_columns(
    monkeypatch,
    tmp_path,
):
    import flights.evaluation.evaluation as evaluation
    from flights.sources import BTS_SOURCE, SFDPS_SOURCE

    df_flights = pl.DataFrame(
        [
            flight("abc123", 1, "KJFK", "KJFK"),
            flight("abc123", 2, "KLGA", "KJFK"),
        ]
    )
    source_frames = {
        BTS_SOURCE.name: df_flights,
        SFDPS_SOURCE.name: df_flights,
    }

    monkeypatch.setattr(
        evaluation,
        "source_flights_reader",
        lambda source, adsb_src=None: (
            lambda _: source_frames[source.name].clone()
        ),
    )
    monkeypatch.setattr(
        evaluation,
        "default_stats_path",
        lambda: tmp_path / "stats.csv",
    )
    monkeypatch.setattr(
        evaluation,
        "airport_catalog_for_date",
        lambda _: AirportCatalog(),
    )

    results, df_stats = evaluation.main(
        test_src=BTS_SOURCE,
        gold_src=SFDPS_SOURCE,
        test_dates=date(2026, 1, 1),
        use_all_icaos=True,
        include_consecutive_airport_metrics=True,
    )

    result = results[0]
    largest_gap = (
        result
        .filter(pl.col("consecutive_airport_mismatch").fill_null(False))
        .sort("consecutive_airport_displacement_km", descending=True)
        .row(0, named=True)
    )
    assert largest_gap["icao"] == "abc123"
    assert largest_gap["takeoff_time"] == datetime(2026, 1, 1, 2)
    assert largest_gap["previous_takeoff_time"] == datetime(2026, 1, 1, 1)
    assert largest_gap["previous_landing_airport_ident"] == "KJFK"
    assert largest_gap["takeoff_airport_ident"] == "KLGA"
    assert largest_gap["consecutive_airport_displacement_km"] > 0
    assert df_stats.item(0, "consecutive_airport_comparison_count") == 1
