from datetime import date

import polars as pl

from flights.flight_source import NonADSBFlightSource
from flights.flight_type import FLIGHT_POLARS_SCHEMA
from flights.providers import opensky
from flights.providers import precomputed
from flights.sources import ADSBX_SOURCE, FLIGHT_SOURCES_BY_NAME, OPENSKY_SOURCE


def _minimal_flights() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "icao": ["abc123"],
            "takeoff_time": [date(2026, 3, 1)],
            "landing_time": [date(2026, 3, 1)],
            "takeoff_airport_ident": ["KAAA"],
            "landing_airport_ident": ["KBBB"],
        }
    )


def test_precomputed_sources_are_registered_as_non_adsb_sources():
    assert FLIGHT_SOURCES_BY_NAME["adsbx"] is ADSBX_SOURCE
    assert FLIGHT_SOURCES_BY_NAME["opensky"] is OPENSKY_SOURCE
    assert isinstance(ADSBX_SOURCE, NonADSBFlightSource)
    assert isinstance(OPENSKY_SOURCE, NonADSBFlightSource)


def test_adsbx_source_reads_and_normalizes_precomputed_product(monkeypatch):
    day = date(2026, 3, 1)
    monkeypatch.setattr(
        precomputed,
        "get_adsbx_flights_for_day",
        lambda target_date: _minimal_flights() if target_date == day else None,
    )

    result = ADSBX_SOURCE.create_flights_for_day(day, airport_catalog=None)

    assert result.columns == list(FLIGHT_POLARS_SCHEMA)
    assert result["icao"].to_list() == ["abc123"]


def test_opensky_source_reads_and_normalizes_trino_download(monkeypatch):
    day = date(2026, 3, 1)
    monkeypatch.setattr(
        opensky,
        "read_opensky_flights",
        lambda target_date: _minimal_flights() if target_date == day else None,
    )

    result = OPENSKY_SOURCE.create_flights_for_day(day, airport_catalog=None)

    assert result.columns == list(FLIGHT_POLARS_SCHEMA)
    assert result["icao"].to_list() == ["abc123"]
