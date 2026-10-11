import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

os.environ.setdefault("OUTPUT_DIR", str(Path(__file__).resolve().parents[3]))

from flights.providers import sfdps
from flights.providers.bts import BTSFlightSource
from flights.verified.source import VerifiedFlightSource


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


def test_resolve_sfdps_airports_maps_unknown_non_null_code_to_null():
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


def test_resolve_sfdps_airports_maps_ambiguous_code_to_null():
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

    result = sfdps.resolve_sfdps_airports(source, FakeAirportCatalog())

    assert result.to_dicts() == [
        {
            "takeoff_airport_ident": None,
            "landing_airport_ident": None,
            "takeoff_airport_id": None,
            "landing_airport_id": None,
        }
    ]


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
    assert sfdps.SFDPSFlightSource().version == 6


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


@pytest.mark.parametrize("landing", [datetime(2026, 3, 1, 2), datetime(2026, 3, 2, 2)])
def test_airline_callsign_starting_with_n_uses_callsign_icao_matching(monkeypatch, landing):
    source = pl.DataFrame(
        {
            "gufi": ["gufi-1"],
            "timestamp": [datetime(2026, 3, 1, 2, 0)],
            "callsign": ["NKS108"],
            "flightStatus": ["COMPLETED"],
            "takeoff_time": [datetime(2026, 3, 1, 1, 0)],
            "takeoff_airport_icao": ["KJFK"],
            "landing_time": [landing],
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


@pytest.mark.parametrize("day", [datetime(2026, 3, 1), datetime(2026, 12, 31)])
def test_reconstructs_overnight_flights_from_both_days(monkeypatch, day):
    next_day = day + timedelta(days=1)

    def message(gufi, timestamp, takeoff, landing):
        return {
            "gufi": gufi,
            "timestamp": timestamp,
            "callsign": "UAL123",
            "flightStatus": "ACTIVE" if landing is None else "COMPLETED",
            "takeoff_time": takeoff,
            "takeoff_airport_icao": "KJFK" if takeoff is not None else None,
            "landing_time": landing,
            "landing_airport_icao": "KLAX",
            "estimated_takeoff_time": None,
            "estimated_landing_time": None,
        }

    same_day = message("same-day", day + timedelta(hours=12),
                       day + timedelta(hours=10), day + timedelta(hours=12))
    messages = pl.DataFrame([
        same_day,
        message("midnight-departure", day + timedelta(hours=1),
                day, day + timedelta(hours=1)),
        message("overnight", day + timedelta(hours=22),
                day + timedelta(hours=22), None),
        # The arrival update omits departure state: it must be carried over
        # from the previous day's message with the same GUFI.
        message("overnight", next_day + timedelta(hours=1),
                None, next_day + timedelta(hours=1)),
        {**same_day, "timestamp": next_day + timedelta(minutes=5)},
        message("next-day-departure", next_day + timedelta(hours=3),
                next_day + timedelta(hours=2), next_day + timedelta(hours=3)),
        message("previous-day-departure", day + timedelta(hours=1),
                day - timedelta(hours=2), day + timedelta(hours=1)),
    ], schema_overrides={
        "estimated_takeoff_time": pl.Datetime("us"),
        "estimated_landing_time": pl.Datetime("us"),
    })
    read_days = []

    def read_day(current_day):
        read_days.append(current_day)
        # Deliberately unsorted: reconstruction must order the merged history.
        return messages.filter(pl.col("timestamp").dt.date() == current_day.date()).reverse()

    monkeypatch.setattr(sfdps, "_read_sfdps_for_day", read_day)
    flights = sfdps.sfdps_to_flights(day)
    result = sfdps.filter_sfdps_flights(flights, day)

    assert read_days == [day, next_day]
    assert result.height == 3
    assert result.sort("takeoff_time").select(
        "takeoff_time", "landing_time", "takeoff_airport_ident"
    ).to_dicts() == [
        {"takeoff_time": day,
         "landing_time": day + timedelta(hours=1), "takeoff_airport_ident": "KJFK"},
        {"takeoff_time": day + timedelta(hours=10),
         "landing_time": day + timedelta(hours=12), "takeoff_airport_ident": "KJFK"},
        {"takeoff_time": day + timedelta(hours=22),
         "landing_time": next_day + timedelta(hours=1), "takeoff_airport_ident": "KJFK"},
    ]


def test_sfdps_requires_following_day_logs(monkeypatch):
    monkeypatch.setattr(sfdps, "_missing_sfdps_days", lambda: {date(2026, 3, 2)})
    with pytest.raises(AssertionError, match="following day's logs"):
        sfdps.create_sfdps_flights_day(date(2026, 3, 1), FakeAirportCatalog())


def test_sfdps_verified_cache_version_changes_without_changing_bts():
    assert VerifiedFlightSource(sfdps.SFDPSFlightSource()).version == 2
    assert VerifiedFlightSource(BTSFlightSource()).version == 1
