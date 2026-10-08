import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


os.environ.setdefault("OUTPUT_DIR", str(Path(__file__).resolve().parents[3]))

import polars as pl

from flights.providers import bts as bts_module
from flights.providers.bts import (
    _bts_month_paths,
    _compute_bts_timestamps,
    _create_flights_from_bts_month,
    create_flights_from_bts,
    iata_airport_mapping,
)


@dataclass(frozen=True)
class FakeAirportCatalog:
    name: str = "test-catalog"


def test_bts_month_paths_include_both_adjacent_months(monkeypatch, tmp_path):
    monkeypatch.setattr(bts_module, "OUTPUT_DIR", tmp_path)

    paths = _bts_month_paths(2026, 1)

    assert [path.relative_to(tmp_path).as_posix() for path in paths] == [
        "data/raw/bts/v2/year=2025/month=12/T_ONTIME_REPORTING.csv",
        "data/raw/bts/v2/year=2026/month=01/T_ONTIME_REPORTING.csv",
        "data/raw/bts/v2/year=2026/month=02/T_ONTIME_REPORTING.csv",
    ]


def test_iata_airport_mapping_omits_unknown_non_null_code():
    class UnknownAirportCatalog:
        def resolve_code(self, code, *, code_priority):
            assert code_priority == ("iata_code",)
            return None

    result = iata_airport_mapping(UnknownAirportCatalog(), ["ZZZ", "ZZZ"])

    assert result.is_empty()
    assert result.schema == {
        "iata": pl.String,
        "airport_id": pl.Int64,
        "ident": pl.String,
    }


def test_iata_airport_mapping_omits_ambiguous_code():
    class AmbiguousAirportCatalog:
        def resolve_code(self, code, *, code_priority):
            raise ValueError(f"ambiguous airport code: {code}")

    result = iata_airport_mapping(AmbiguousAirportCatalog(), ["AAA"])

    assert result.is_empty()


def test_compute_bts_timestamps_does_not_roll_one_minute_early_departure_to_next_day():
    timestamps = _compute_bts_timestamps(
        {
            "fl_date": date(2026, 2, 28),
            "origin_timezone": "America/Boise",
            "wheels_off": 524,
            "crs_dep_time": 525,
            "air_time": 33,
        }
    )

    assert timestamps == {
        "time_takeoff": datetime(2026, 2, 28, 12, 24),
        "time_landing": datetime(2026, 2, 28, 12, 57),
    }


def test_compute_bts_timestamps_rolls_true_midnight_wrap_to_next_day():
    timestamps = _compute_bts_timestamps(
        {
            "fl_date": date(2026, 2, 28),
            "origin_timezone": "America/Boise",
            "wheels_off": 5,
            "crs_dep_time": 2355,
            "air_time": 30,
        }
    )

    assert timestamps == {
        "time_takeoff": datetime(2026, 3, 1, 7, 5),
        "time_landing": datetime(2026, 3, 1, 7, 35),
    }


def test_compute_bts_timestamps_uses_arrival_delay_for_delayed_next_day_flight():
    timestamps = _compute_bts_timestamps(
        {
            "fl_date": date(2026, 3, 1),
            "origin_timezone": "America/Chicago",
            "dest_timezone": "America/Chicago",
            "wheels_off": 1319,
            "wheels_on": 1406,
            "crs_dep_time": 2234,
            "crs_arr_time": 2348,
            "arr_delay": 864,
            "air_time": 47,
        }
    )

    assert timestamps == {
        "time_takeoff": datetime(2026, 3, 2, 19, 19),
        "time_landing": datetime(2026, 3, 2, 20, 6),
    }


def test_compute_bts_timestamps_uses_arrival_delay_for_same_day_flight():
    timestamps = _compute_bts_timestamps(
        {
            "fl_date": date(2026, 3, 1),
            "origin_timezone": "America/New_York",
            "dest_timezone": "America/New_York",
            "wheels_off": 600,
            "wheels_on": 703,
            "crs_dep_time": 549,
            "crs_arr_time": 703,
            "arr_delay": 0,
            "air_time": 63,
        }
    )

    assert timestamps == {
        "time_takeoff": datetime(2026, 3, 1, 11, 0),
        "time_landing": datetime(2026, 3, 1, 12, 3),
    }


def test_create_flights_from_bts_month_keeps_short_flights(monkeypatch, tmp_path):
    source_path = tmp_path / "T_ONTIME_REPORTING.csv"
    pl.DataFrame(
        {
            "FL_DATE": ["03/01/2026 12:00:00 AM"],
            "OP_UNIQUE_CARRIER": ["AA"],
            "OP_CARRIER_FL_NUM": [123],
            "ORIGIN": ["AAA"],
            "DEST": ["BBB"],
            "WHEELS_OFF": [600],
            "WHEELS_ON": [620],
            "CANCELLED": [0],
            "AIR_TIME": [20],
            "DIVERTED": [0],
            "CRS_DEP_TIME": [600],
            "CRS_ARR_TIME": [620],
            "ARR_DELAY": [0],
            "TAIL_NUM": ["N12345"],
        }
    ).write_csv(source_path)

    monkeypatch.setattr(
        bts_module,
        "_bts_month_paths",
        lambda _year, _month: [source_path],
    )
    monkeypatch.setattr(
        bts_module,
        "get_registration_to_icao",
        lambda at_date=None: pl.DataFrame(
            {"registration": ["N12345"], "icao": ["ABC123"]}
        ),
    )
    monkeypatch.setattr(
        bts_module,
        "iata_airport_mapping",
        lambda _catalog, _codes: pl.DataFrame(
            {
                "iata": ["AAA", "BBB"],
                "airport_id": [1, 2],
                "ident": ["KAAA", "KBBB"],
            }
        ),
    )
    monkeypatch.setattr(
        bts_module,
        "airport_timezone_mapping",
        lambda _catalog: pl.DataFrame(
            {"airport_id": [1, 2], "timezone": ["UTC", "UTC"]}
        ),
    )

    result = _create_flights_from_bts_month(
        2026,
        3,
        FakeAirportCatalog(),
    )

    assert result.select(
        "icao",
        "takeoff_time",
        "landing_time",
    ).row(0) == (
        "abc123",
        datetime(2026, 3, 1, 6),
        datetime(2026, 3, 1, 6, 20),
    )


def test_create_flights_from_bts_returns_only_target_day(monkeypatch):
    df = pl.DataFrame(
        {
            "icao": ["outside-before", "abc123", "abc123", "def456", "outside-after"],
            "takeoff_time": [
                datetime(2026, 2, 28, 23),
                datetime(2026, 3, 1, 10),
                datetime(2026, 3, 1, 10, 30),
                datetime(2026, 3, 1, 12),
                datetime(2026, 4, 1),
            ],
            "landing_time": [
                datetime(2026, 3, 1),
                datetime(2026, 3, 1, 11),
                datetime(2026, 3, 1, 11, 30),
                datetime(2026, 3, 1, 13),
                datetime(2026, 4, 1, 1),
            ],
        }
    )
    airport_catalog = FakeAirportCatalog()

    def create_month(year, month, catalog, openairframes_date=None):
        assert (year, month) == (2026, 3)
        assert catalog is airport_catalog
        assert openairframes_date == date(2026, 3, 1)
        return df

    monkeypatch.setattr(bts_module, "_create_flights_from_bts_month", create_month)
    result = create_flights_from_bts(date(2026, 3, 1), airport_catalog)

    assert result.get_column("icao").to_list() == ["def456"]
    assert bts_module.BTSFlightSource().version == 11
