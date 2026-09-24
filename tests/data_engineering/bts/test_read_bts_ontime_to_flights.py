import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


os.environ.setdefault("OUTPUT_DIR", str(Path(__file__).resolve().parents[3]))

import polars as pl
import pytest

from flights.providers import bts as bts_module
from flights.providers.bts import (
    _compute_bts_timestamps,
    create_flights_from_bts,
    iata_airport_mapping,
)


@dataclass(frozen=True)
class FakeAirportCatalog:
    name: str = "test-catalog"


def test_iata_airport_mapping_rejects_unknown_non_null_code():
    class UnknownAirportCatalog:
        def resolve_code(self, code, *, code_priority):
            assert code_priority == ("iata_code",)
            return None

    with pytest.raises(ValueError, match="Unknown BTS IATA airport code 'ZZZ'"):
        iata_airport_mapping(UnknownAirportCatalog(), ["ZZZ"])


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

    def create_month(year, month, catalog):
        assert (year, month) == (2026, 3)
        assert catalog is airport_catalog
        return df

    monkeypatch.setattr(bts_module, "_create_flights_from_bts_month", create_month)
    result = create_flights_from_bts(date(2026, 3, 1), airport_catalog)

    assert result.get_column("icao").to_list() == ["def456"]
    assert bts_module.BTSFlightSource().version == 6
