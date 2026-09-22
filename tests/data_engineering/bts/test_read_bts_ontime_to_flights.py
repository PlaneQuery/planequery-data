import os
from datetime import date, datetime
from pathlib import Path


os.environ.setdefault("OUTPUT_DIR", str(Path(__file__).resolve().parents[3]))

import polars as pl

from data_engineering.bts import read_bts_ontime_to_flights as bts_module
from data_engineering.bts.read_bts_ontime_to_flights import (
    _compute_bts_timestamps,
    _write_bts_flights_by_day,
)


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


def test_write_bts_flights_by_day_writes_every_day_in_target_month(monkeypatch):
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
    writes = []

    def capture_write(day_df, target_date, source, adsb_src):
        writes.append((target_date, source, adsb_src, day_df))

    monkeypatch.setattr(bts_module, "write_flights_for_day", capture_write)

    result = _write_bts_flights_by_day(df, 2026, 3)

    assert len(writes) == 31
    assert writes[0][:3] == (date(2026, 3, 1), bts_module.FlightSource.BTS, None)
    assert writes[-1][0] == date(2026, 3, 31)
    assert writes[1][3].is_empty()
    assert result.get_column("icao").to_list() == ["def456"]
