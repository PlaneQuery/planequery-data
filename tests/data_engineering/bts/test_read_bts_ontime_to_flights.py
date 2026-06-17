import os
from datetime import date, datetime
from pathlib import Path


os.environ.setdefault("OUTPUT_DIR", str(Path(__file__).resolve().parents[3]))

from data_engineering.bts.read_bts_ontime_to_flights import _compute_bts_timestamps


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
