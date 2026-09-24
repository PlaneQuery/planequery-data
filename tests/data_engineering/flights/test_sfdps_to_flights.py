import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import polars as pl
import pytest

os.environ.setdefault("OUTPUT_DIR", str(Path(__file__).resolve().parents[3]))

from flights.providers import sfdps


@dataclass(frozen=True)
class FakeAirport:
    id: int
    ident: str


@dataclass(frozen=True)
class FakeAirportCatalog:
    def resolve_code(
        self,
        code: str,
        *,
        code_priority: tuple[str, ...],
    ) -> FakeAirport | None:
        assert code_priority == sfdps.SFDPS_AIRPORT_CODE_PRIORITY
        if code == "AMBIGUOUS":
            raise ValueError("ambiguous airport code")
        return {
            "ALM": FakeAirport(101, "KALM"),
            "IA8": FakeAirport(202, "IA8"),
        }.get(code)


def test_resolve_sfdps_airports_replaces_raw_codes_with_stable_ids():
    source = pl.DataFrame(
        {
            "takeoff_airport_ident": ["ALM", None],
            "landing_airport_ident": ["IA8", None],
        },
        schema={
            "takeoff_airport_ident": pl.String,
            "landing_airport_ident": pl.String,
        },
    )

    result = sfdps.resolve_sfdps_airports(source, FakeAirportCatalog())

    assert result.to_dicts() == [
        {
            "takeoff_airport_ident": "KALM",
            "landing_airport_ident": "IA8",
            "takeoff_airport_id": 101,
            "landing_airport_id": 202,
        },
        {
            "takeoff_airport_ident": None,
            "landing_airport_ident": None,
            "takeoff_airport_id": None,
            "landing_airport_id": None,
        },
    ]


def test_resolve_sfdps_airports_maps_unknown_non_null_code_to_null(caplog):
    source = pl.DataFrame(
        {
            "takeoff_airport_ident": ["UNKNOWN", "UNKNOWN"],
            "landing_airport_ident": ["IA8", None],
        },
        schema={
            "takeoff_airport_ident": pl.String,
            "landing_airport_ident": pl.String,
        },
    )

    result = sfdps.resolve_sfdps_airports(source, FakeAirportCatalog())

    assert result.to_dicts() == [
        {
            "takeoff_airport_ident": None,
            "landing_airport_ident": "IA8",
            "takeoff_airport_id": None,
            "landing_airport_id": 202,
        },
        {
            "takeoff_airport_ident": None,
            "landing_airport_ident": None,
            "takeoff_airport_id": None,
            "landing_airport_id": None,
        },
    ]
    assert (
        "SFDPS unresolved airport/point codes mapped to null: "
        "'UNKNOWN' (2 occurrences)" in caplog.text
    )


def test_resolve_sfdps_airports_does_not_hide_ambiguous_codes():
    source = pl.DataFrame(
        {
            "takeoff_airport_ident": ["AMBIGUOUS"],
            "landing_airport_ident": [None],
        },
        schema={
            "takeoff_airport_ident": pl.String,
            "landing_airport_ident": pl.String,
        },
    )

    with pytest.raises(ValueError, match="ambiguous airport code"):
        sfdps.resolve_sfdps_airports(source, FakeAirportCatalog())


def test_create_sfdps_flights_day_resolves_airport_codes(monkeypatch):
    target_date = date(2026, 3, 1)
    source = pl.DataFrame(
        {
            "icao": ["abc123"],
            "takeoff_time": [datetime(2026, 3, 1, 1, 0)],
            "takeoff_airport_ident": ["ALM"],
            "landing_airport_ident": ["IA8"],
        }
    )
    monkeypatch.setattr(sfdps, "sfdps_to_flights", lambda _day: source)
    monkeypatch.setattr(sfdps, "filter_sfdps_flights", lambda flights, _day: flights)
    monkeypatch.setattr(sfdps, "sfdps_flights_derive_icao", lambda flights, _day: flights)
    airport_catalog = FakeAirportCatalog()

    result = sfdps.create_sfdps_flights_day(target_date, airport_catalog)

    assert result.to_dicts() == [
        {
            "icao": "abc123",
            "takeoff_time": datetime(2026, 3, 1, 1, 0),
            "takeoff_airport_ident": "KALM",
            "landing_airport_ident": "IA8",
            "takeoff_airport_id": 101,
            "landing_airport_id": 202,
        }
    ]
    assert sfdps.SFDPSFlightSource().version == 4


def test_sfdps_to_flights_only_treats_valid_registrations_as_registrations(
    monkeypatch,
):
    callsigns = ["NKS108", "N12345", "N123AB", "N123A4", "C-GABC"]
    source = pl.DataFrame(
        {
            "gufi": [f"gufi-{index}" for index in range(len(callsigns))],
            "timestamp": [datetime(2026, 3, 1, 2, 0)] * len(callsigns),
            "callsign": callsigns,
            "flightStatus": ["COMPLETED"] * len(callsigns),
            "takeoff_time": [datetime(2026, 3, 1, 1, 0)] * len(callsigns),
            "takeoff_airport_icao": ["KJFK"] * len(callsigns),
            "landing_time": [datetime(2026, 3, 1, 2, 0)] * len(callsigns),
            "landing_airport_icao": ["KBOS"] * len(callsigns),
            "estimated_landing_time": [None] * len(callsigns),
            "estimated_takeoff_time": [None] * len(callsigns),
        },
        schema_overrides={
            "estimated_landing_time": pl.Datetime("us"),
            "estimated_takeoff_time": pl.Datetime("us"),
        },
    )
    monkeypatch.setattr(sfdps, "_read_sfdps_for_day", lambda _current_day: source)

    result = (
        sfdps.sfdps_to_flights(datetime(2026, 3, 1))
        .select("callsign", "registration")
        .sort("callsign")
    )

    assert result.to_dicts() == [
        {"callsign": "C-GABC", "registration": "C-GABC"},
        {"callsign": "N12345", "registration": "N12345"},
        {"callsign": "N123A4", "registration": None},
        {"callsign": "N123AB", "registration": "N123AB"},
        {"callsign": "NKS108", "registration": None},
    ]


def test_airline_callsign_starting_with_n_uses_callsign_icao_matching(monkeypatch):
    source = pl.DataFrame(
        {
            "gufi": ["gufi-1"],
            "timestamp": [datetime(2026, 3, 1, 2, 0)],
            "callsign": ["NKS108"],
            "flightStatus": ["COMPLETED"],
            "takeoff_time": [datetime(2026, 3, 1, 1, 0)],
            "takeoff_airport_icao": ["KJFK"],
            "landing_time": [datetime(2026, 3, 1, 2, 0)],
            "landing_airport_icao": ["KBOS"],
            "estimated_landing_time": [None],
            "estimated_takeoff_time": [None],
        },
        schema_overrides={
            "estimated_landing_time": pl.Datetime("us"),
            "estimated_takeoff_time": pl.Datetime("us"),
        },
    )
    monkeypatch.setattr(sfdps, "_read_sfdps_for_day", lambda _current_day: source)
    monkeypatch.setattr(
        sfdps,
        "_adsb_registration_map",
        lambda _current_day, _registrations: pl.DataFrame(
            schema={
                "icao": pl.String,
                "registration": pl.String,
                "pia": pl.Boolean,
                "ladd": pl.Boolean,
            }
        ),
    )
    monkeypatch.setattr(
        sfdps,
        "_adsb_callsign_segments",
        lambda _current_day, _callsigns: pl.DataFrame(
            {
                "icao": ["a91349"],
                "callsign": ["NKS108"],
                "start_time": [datetime(2026, 3, 1, 0, 55)],
                "end_time": [datetime(2026, 3, 1, 2, 5)],
                "pia": [False],
                "ladd": [False],
            }
        ),
    )
    monkeypatch.setattr(
        sfdps,
        "add_adsb_icao_info",
        lambda flights, _current_day: flights,
    )

    flights = sfdps.sfdps_to_flights(datetime(2026, 3, 1))
    result = sfdps.sfdps_flights_derive_icao(flights, datetime(2026, 3, 1))

    assert result.select("icao", "callsign").to_dicts() == [
        {"icao": "a91349", "callsign": "NKS108"}
    ]
