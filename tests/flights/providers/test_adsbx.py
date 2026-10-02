from dataclasses import dataclass
from datetime import date

import polars as pl

from flights.flight_source import NonADSBFlightSource
from flights.flight_type import FLIGHT_POLARS_SCHEMA
from flights.providers import adsbx
from flights.sources import ADSBX_SOURCE, FLIGHT_SOURCES_BY_NAME


@dataclass(frozen=True)
class FakeAirport:
    id: int
    ident: str


class FakeAirportCatalog:
    def resolve_code(self, code, *, code_priority):
        assert code_priority == adsbx.ADSBX_AIRPORT_CODE_PRIORITY
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


def test_adsbx_source_is_registered_as_an_independent_source():
    assert FLIGHT_SOURCES_BY_NAME["adsbx"] is ADSBX_SOURCE
    assert isinstance(ADSBX_SOURCE, NonADSBFlightSource)


def test_adsbx_source_resolves_airports_and_enriches_aircraft_flags(monkeypatch):
    day = date(2026, 3, 1)
    monkeypatch.setattr(
        adsbx,
        "read_adsbx_published_same_day_flights",
        lambda target_date: _minimal_flights() if target_date == day else None,
    )
    enrichment_calls = []

    def add_aircraft_info(flights, at_date):
        enrichment_calls.append((flights, at_date))
        return flights.with_columns(pl.lit(True).alias("pia"))

    monkeypatch.setattr(adsbx, "add_icao_info_as_of", add_aircraft_info)

    result = ADSBX_SOURCE.create_flights_for_day(day, FakeAirportCatalog())

    assert enrichment_calls[0][1] == day
    assert result.columns == list(FLIGHT_POLARS_SCHEMA)
    assert result["icao"].to_list() == ["abc123"]
    assert result["pia"].to_list() == [True]
    assert result.select(
        "takeoff_airport_id",
        "takeoff_airport_ident",
        "landing_airport_id",
        "landing_airport_ident",
    ).row(0) == (101, "KAAA", 202, "KBBB")
    assert ADSBX_SOURCE.version == 3


def test_adsbx_source_keeps_flights_with_unresolved_airports(monkeypatch):
    day = date(2026, 3, 1)
    monkeypatch.setattr(
        adsbx,
        "read_adsbx_published_same_day_flights",
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
    monkeypatch.setattr(
        adsbx,
        "add_icao_info_as_of",
        lambda flights, _day: flights,
    )

    result = ADSBX_SOURCE.create_flights_for_day(day, FakeAirportCatalog())

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
