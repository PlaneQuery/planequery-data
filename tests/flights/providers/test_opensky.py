from dataclasses import dataclass
from datetime import date

import polars as pl

from flights.flight_source import NonADSBFlightSource
from flights.flight_type import FLIGHT_POLARS_SCHEMA
from flights.providers import opensky
from flights.sources import FLIGHT_SOURCES_BY_NAME, OPENSKY_SOURCE


@dataclass(frozen=True)
class FakeAirport:
    id: int
    ident: str


class FakeAirportCatalog:
    def resolve_code(self, code, *, code_priority):
        assert code_priority == opensky.OPENSKY_AIRPORT_CODE_PRIORITY
        if code == "AMBIGUOUS":
            raise ValueError("ambiguous airport code")
        return {
            "KAAA": FakeAirport(101, "KAAA"),
            "KBBB": FakeAirport(202, "KBBB"),
        }.get(code)


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


def test_opensky_source_is_registered_as_an_independent_source():
    assert FLIGHT_SOURCES_BY_NAME["opensky"] is OPENSKY_SOURCE
    assert isinstance(OPENSKY_SOURCE, NonADSBFlightSource)


def test_opensky_source_reads_and_normalizes_trino_download(monkeypatch):
    day = date(2026, 3, 1)
    monkeypatch.setattr(
        opensky,
        "read_opensky_flights",
        lambda target_date: _minimal_flights() if target_date == day else None,
    )

    result = OPENSKY_SOURCE.create_flights_for_day(day, FakeAirportCatalog())

    assert result.columns == list(FLIGHT_POLARS_SCHEMA)
    assert result["icao"].to_list() == ["abc123"]
    assert result.select(
        "takeoff_airport_id",
        "takeoff_airport_ident",
        "landing_airport_id",
        "landing_airport_ident",
    ).row(0) == (101, "KAAA", 202, "KBBB")
    assert OPENSKY_SOURCE.version == 4


def test_opensky_source_keeps_flights_with_unresolved_airports(monkeypatch):
    day = date(2026, 3, 1)
    monkeypatch.setattr(
        opensky,
        "read_opensky_flights",
        lambda target_date: pl.DataFrame(
            {
                "icao": ["abc123", "def456"],
                "takeoff_time": [day, day],
                "landing_time": [day, day],
                "takeoff_airport_ident": ["UNKNOWN", "AMBIGUOUS"],
                "landing_airport_ident": ["KBBB", None],
            }
        ),
    )

    result = OPENSKY_SOURCE.create_flights_for_day(day, FakeAirportCatalog())

    assert result.select(
        "takeoff_airport_id",
        "takeoff_airport_ident",
        "landing_airport_id",
        "landing_airport_ident",
    ).to_dicts() == [
        {
            "takeoff_airport_id": None,
            "takeoff_airport_ident": None,
            "landing_airport_id": 202,
            "landing_airport_ident": "KBBB",
        },
        {
            "takeoff_airport_id": None,
            "takeoff_airport_ident": None,
            "landing_airport_id": None,
            "landing_airport_ident": None,
        },
    ]
