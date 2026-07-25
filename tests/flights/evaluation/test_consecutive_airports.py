from datetime import datetime, timedelta

import polars as pl
import pytest

from flights.evaluation.consecutive_airports import (
    consecutive_airport_mismatches,
    consecutive_airport_stats,
)
from utils import haversine


class CoordinatesLookup:
    coordinates = {
        "A": (0.0, 0.0),
        "B": (0.0, 0.1),
        "C": (0.0, 1.0),
        "D": (0.0, 2.0),
    }

    def get_airport_coordinates(
        self,
        airport_ident: str,
    ) -> tuple[float, float] | None:
        return self.coordinates.get(airport_ident)


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
        "takeoff_airport_ident": takeoff_airport_ident,
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
        airport_lookup=CoordinatesLookup(),
    )
    stats = consecutive_airport_stats(
        df_flights,
        airport_lookup=CoordinatesLookup(),
    )

    assert mismatches.select(
        "previous_landing_airport_ident",
        "takeoff_airport_ident",
    ).rows() == [("C", "D")]
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


def test_unknown_airport_coordinates_still_count_as_a_mismatch():
    df_flights = pl.DataFrame(
        [
            flight("abc123", 1, "A", "UNKNOWN"),
            flight("abc123", 2, "B", "C"),
        ]
    )

    stats = consecutive_airport_stats(
        df_flights,
        airport_lookup=CoordinatesLookup(),
    )

    assert stats["consecutive_airport_comparison_count"] == 1
    assert stats["consecutive_airport_mismatch_count"] == 1
    assert stats["consecutive_airport_mismatch_rate"] == 1.0
    assert stats["consecutive_airport_displacement_km_p50"] is None
    assert stats["consecutive_airport_displacement_km_max"] is None
    assert stats["consecutive_airport_displacement_over_50km_count"] == 0
    assert stats["consecutive_airport_displacement_over_50km_rate"] == 0.0


def test_missing_airport_idents_are_not_comparable():
    df_flights = pl.DataFrame(
        [
            flight("abc123", 1, "A", None),
            flight("abc123", 2, "B", "C"),
        ]
    )

    stats = consecutive_airport_stats(
        df_flights,
        airport_lookup=CoordinatesLookup(),
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
        airport_lookup=CoordinatesLookup(),
    )

    assert stats["consecutive_airport_displacement_over_10km_count"] == 1
    assert stats["consecutive_airport_displacement_over_10km_rate"] == 1.0
