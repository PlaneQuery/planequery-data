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
from data_engineering.swim import read_sfdps as sfdps_reader


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
    monkeypatch.setattr(sfdps, "add_adsb_icao_info", lambda flights, _day: flights)
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
    assert sfdps.SFDPSFlightSource().version == 9


def test_sfdps_to_flights_uses_aircraft_identification_as_callsign(monkeypatch):
    callsigns = ["NKS108", "N12345", "N123AB", "N123A4", "C-GABC"]
    registrations = ["N345NK", None, "N123AB", None, "C-GABC"]
    source = pl.DataFrame(
        {
            "gufi": [f"gufi-{index}" for index in range(len(callsigns))],
            "timestamp": [datetime(2026, 3, 1, 2, 0)] * len(callsigns),
            "aircraft_identification": callsigns,
            "aircraft_address": [None] * len(callsigns),
            "aircraft_registration": registrations,
            "aircraft_type": [None] * len(callsigns),
            "flightStatus": ["COMPLETED"] * len(callsigns),
            "takeoff_time": [datetime(2026, 3, 1, 1, 0)] * len(callsigns),
            "takeoff_airport_icao": ["KJFK"] * len(callsigns),
            "landing_time": [datetime(2026, 3, 1, 2, 0)] * len(callsigns),
            "landing_airport_icao": ["KBOS"] * len(callsigns),
            "estimated_landing_time": [None] * len(callsigns),
            "estimated_takeoff_time": [None] * len(callsigns),
        },
        schema_overrides={
            "aircraft_address": pl.String,
            "aircraft_registration": pl.String,
            "aircraft_type": pl.String,
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
        {"callsign": "N12345", "registration": None},
        {"callsign": "N123A4", "registration": None},
        {"callsign": "N123AB", "registration": "N123AB"},
        {"callsign": "NKS108", "registration": "N345NK"},
    ]


@pytest.mark.parametrize("landing", [datetime(2026, 3, 1, 2), datetime(2026, 3, 2, 2)])
@pytest.mark.parametrize("callsign", ["NKS108", "N12345"])
def test_uses_sfdps_addresses_without_callsign_or_registration_matching(
    monkeypatch, landing, callsign,
):
    source = pl.DataFrame(
        {
            "gufi": [f"gufi-{index}" for index in range(4)],
            "timestamp": [landing] * 4,
            "aircraft_identification": [callsign] * 4,
            "aircraft_address": ["A91349", "012ABC", None, ""],
            "aircraft_registration": [None] * 4,
            "aircraft_type": [None] * 4,
            "flightStatus": ["COMPLETED"] * 4,
            "takeoff_time": [datetime(2026, 3, 1, 1, 0)] * 4,
            "takeoff_airport_icao": ["ALM"] * 4,
            "landing_time": [landing] * 4,
            "landing_airport_icao": ["IA8"] * 4,
            "estimated_landing_time": [None] * 4,
            "estimated_takeoff_time": [None] * 4,
        },
        schema_overrides={
            "aircraft_registration": pl.String,
            "aircraft_type": pl.String,
            "estimated_landing_time": pl.Datetime("us"),
            "estimated_takeoff_time": pl.Datetime("us"),
        },
    )
    monkeypatch.setattr(sfdps, "_read_sfdps_for_day", lambda _current_day: source)
    scans = []

    def scan_metadata(current_day, *, columns, icaos):
        scans.append(current_day)
        assert sorted(icaos) == ["012abc", "a91349"]
        assert "callsign" not in columns
        metadata = sfdps.with_flight_schema_columns(pl.DataFrame({
            "icao": ["a91349"],
            "registration": ["N684NK"],
            "aircraft_type": ["A321"],
        })).with_columns(pl.lit(datetime(2026, 3, 1, 2)).alias("time"))
        return metadata.select(columns).lazy()

    monkeypatch.setattr(sfdps, "scan_adsb", scan_metadata)
    result = sfdps.create_sfdps_flights_day(date(2026, 3, 1), FakeAirportCatalog())

    assert scans == [date(2026, 3, 1)]
    assert result.select("icao", "callsign", "registration", "aircraft_type").to_dicts() == [
        {"icao": "012abc", "callsign": callsign, "registration": "", "aircraft_type": ""},
        {"icao": "a91349", "callsign": callsign, "registration": "N684NK", "aircraft_type": "A321"},
    ]


@pytest.mark.parametrize("day", [datetime(2026, 3, 1), datetime(2026, 12, 31)])
def test_reconstructs_overnight_flights_from_both_days(monkeypatch, day):
    next_day = day + timedelta(days=1)

    def message(gufi, timestamp, takeoff, landing):
        return {
            "gufi": gufi,
            "timestamp": timestamp,
            "aircraft_identification": "UAL123",
            "aircraft_address": "A123BC" if takeoff is not None else None,
            "aircraft_registration": "N12345",
            "aircraft_type": "B738",
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
    assert result["icao"].to_list() == ["a123bc"] * 3
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
    assert VerifiedFlightSource(sfdps.SFDPSFlightSource()).version == 5
    assert VerifiedFlightSource(BTSFlightSource()).version == 1


def test_reader_preserves_aircraft_address_changes(monkeypatch, tmp_path):
    parquet_path = tmp_path / "sfdps.parquet"
    pl.DataFrame({
        "gufi": ["gufi-1"] * 3,
        "timestamp": [datetime(2026, 3, 1, hour) for hour in (1, 2, 3)],
        "aircraft_identification": ["UAL123"] * 3,
        "aircraft_address": ["ABC123", "DEF456", None],
        "aircraft_registration": ["N12345"] * 3,
        "aircraft_type": ["B738"] * 3,
        "flight_status": ["COMPLETED"] * 3,
        "departure_point": ["KJFK"] * 3,
        "departure_actual_time": [datetime(2026, 3, 1, 0)] * 3,
        "departure_estimated_time": [datetime(2026, 3, 1, 0)] * 3,
        "arrival_point": ["KLAX"] * 3,
        "arrival_actual_time": [datetime(2026, 3, 1, 1)] * 3,
        "arrival_estimated_time": [datetime(2026, 3, 1, 1)] * 3,
    }).write_parquet(parquet_path)
    monkeypatch.setattr(sfdps_reader, "_parquet_path", lambda _day: parquet_path)

    result = sfdps_reader.read_sfdps(datetime(2026, 3, 1))

    assert "callsign" not in result.columns
    assert result["aircraft_address"].to_list() == ["ABC123", "DEF456", None]
    assert result.select(
        "aircraft_identification",
        "aircraft_registration",
        "aircraft_type",
    ).to_dicts() == [
        {
            "aircraft_identification": "UAL123",
            "aircraft_registration": "N12345",
            "aircraft_type": "B738",
        }
    ] * 3


@pytest.mark.parametrize("addresses, expected", [
    (["ABC123", None], "abc123"),
    ([None, "DEF456"], "def456"),
    (["ABC123", "DEF456"], "def456"),
    ([None, None], None),
])
def test_reconstruction_carries_aircraft_address_state(monkeypatch, addresses, expected):
    day = datetime(2026, 3, 1)
    source = pl.DataFrame({
        "gufi": ["gufi-1"] * 2,
        "timestamp": [day + timedelta(hours=1), day + timedelta(hours=2)],
        "aircraft_identification": ["UAL123"] * 2,
        "aircraft_address": addresses,
        "aircraft_registration": ["N12345"] * 2,
        "aircraft_type": ["B738"] * 2,
        "flightStatus": ["ACTIVE", "COMPLETED"],
        "takeoff_time": [day, None],
        "takeoff_airport_icao": ["KJFK", None],
        "landing_time": [None, day + timedelta(hours=2)],
        "landing_airport_icao": ["KLAX"] * 2,
        "estimated_landing_time": [None] * 2,
        "estimated_takeoff_time": [None] * 2,
    }, schema_overrides={
        "aircraft_address": pl.String,
        "estimated_landing_time": pl.Datetime("us"),
        "estimated_takeoff_time": pl.Datetime("us"),
    })
    monkeypatch.setattr(
        sfdps, "_read_sfdps_for_day",
        lambda current_day: source if current_day == day else source.clear(),
    )

    result = sfdps.sfdps_to_flights(day)

    assert result["icao"].to_list() == [expected]
