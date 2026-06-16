from datetime import date, datetime, timedelta
from types import SimpleNamespace

import polars as pl

from flights_algorithm.main import (
    FlightSegment,
    _segment_takes_off_in_window,
    _segments_to_flights_df,
    identify_flight_segments,
    run_main_2,
    stitch_flight_segments,
)


def _row(
    time: datetime,
    lon: float,
    icao: str = "abc123",
    ground_speed_kt: float | None = None,
    on_ground: bool | None = None,
    baro_altitude_ft: float | None = None,
):
    return SimpleNamespace(
        time=time,
        icao=icao,
        lat=0.0,
        lon=lon,
        ground_speed_kt=ground_speed_kt,
        on_ground=on_ground,
        baro_altitude_ft=baro_altitude_ft,
    )


def test_identifies_takeoff_and_landing_from_calculated_ground_speed():
    start = datetime(2026, 1, 1, 12, 0)
    messages = [
        _row(start, 0.00),
        _row(start + timedelta(minutes=1), 0.02),
        _row(start + timedelta(minutes=4), 0.08),
        _row(start + timedelta(minutes=6), 0.0805),
        _row(start + timedelta(minutes=9), 0.0807),
    ]

    segments = identify_flight_segments(messages)

    assert len(segments) == 1
    assert segments[0].takeoff_time == start + timedelta(minutes=1)
    assert segments[0].landing_time == start + timedelta(minutes=6)


def test_large_message_gap_closes_current_segment():
    start = datetime(2026, 1, 1, 12, 0)
    messages = [
        _row(start, 0.00),
        _row(start + timedelta(minutes=1), 0.02),
        _row(start + timedelta(minutes=10), 0.20),
        _row(start + timedelta(minutes=45), 0.80),
        _row(start + timedelta(minutes=46), 0.82),
    ]

    segments = identify_flight_segments(messages)

    assert len(segments) == 1
    assert segments[0].takeoff_time == start + timedelta(minutes=1)
    assert segments[0].landing_time == start + timedelta(minutes=10)


def test_on_ground_state_can_identify_takeoff_and_landing():
    start = datetime(2026, 1, 1, 12, 0)
    messages = [
        _row(start, 0.00, on_ground=True),
        _row(start + timedelta(minutes=1), 0.02, on_ground=False),
        _row(start + timedelta(minutes=8), 0.20, on_ground=False),
        _row(start + timedelta(minutes=12), 0.205, on_ground=True),
        _row(start + timedelta(minutes=15), 0.205, on_ground=True),
    ]

    segments = identify_flight_segments(messages)

    assert len(segments) == 1
    assert segments[0].takeoff_time == start + timedelta(minutes=1)
    assert segments[0].landing_time == start + timedelta(minutes=12)


def test_open_in_flight_segment_closes_at_last_message():
    start = datetime(2026, 1, 1, 12, 0)
    messages = [
        _row(start, 0.00, on_ground=True),
        _row(start + timedelta(minutes=1), 0.02, on_ground=False),
        _row(start + timedelta(minutes=10), 0.20, on_ground=False),
    ]

    segments = identify_flight_segments(messages)

    assert len(segments) == 1
    assert segments[0].takeoff_time == start + timedelta(minutes=1)
    assert segments[0].landing_time == start + timedelta(minutes=10)


def test_continues_flight_across_plausible_airborne_gap():
    start = datetime(2026, 1, 1, 12, 0)
    messages = [
        _row(start, 0.00, on_ground=True, ground_speed_kt=0.0),
        _row(start + timedelta(minutes=1), 0.02, on_ground=False, ground_speed_kt=120.0, baro_altitude_ft=1000),
        _row(start + timedelta(minutes=20), 1.20, on_ground=False, ground_speed_kt=260.0, baro_altitude_ft=8000),
        _row(start + timedelta(hours=2), 8.0, on_ground=False, ground_speed_kt=260.0, baro_altitude_ft=8000),
        _row(start + timedelta(hours=2, minutes=20), 8.2, on_ground=False, ground_speed_kt=95.0, baro_altitude_ft=700),
        _row(start + timedelta(hours=2, minutes=23), 8.22, on_ground=False, ground_speed_kt=50.0, baro_altitude_ft=400),
    ]

    segments = identify_flight_segments(messages)

    assert len(segments) == 1
    assert segments[0].takeoff_time == start + timedelta(minutes=1)
    assert segments[0].landing_time == start + timedelta(hours=2, minutes=20)


def test_does_not_continue_across_implausibly_slow_airborne_gap():
    start = datetime(2026, 1, 1, 12, 0)
    messages = [
        _row(start, 0.00, on_ground=True, ground_speed_kt=0.0),
        _row(start + timedelta(minutes=1), 0.02, on_ground=False, ground_speed_kt=120.0, baro_altitude_ft=1000),
        _row(start + timedelta(minutes=10), 0.30, on_ground=False, ground_speed_kt=150.0, baro_altitude_ft=1500),
        _row(start + timedelta(hours=2), 0.35, on_ground=False, ground_speed_kt=150.0, baro_altitude_ft=1500),
        _row(start + timedelta(hours=2, minutes=10), 0.60, on_ground=False, ground_speed_kt=150.0, baro_altitude_ft=1500),
        _row(start + timedelta(hours=2, minutes=20), 0.80, on_ground=False, ground_speed_kt=100.0, baro_altitude_ft=700),
        _row(start + timedelta(hours=2, minutes=23), 0.81, on_ground=False, ground_speed_kt=50.0, baro_altitude_ft=400),
    ]

    segments = identify_flight_segments(messages)

    assert len(segments) == 2
    assert segments[0].landing_time == start + timedelta(minutes=10)
    assert segments[1].takeoff_time == start + timedelta(hours=2, minutes=10)


def test_continues_across_slow_gap_when_both_sides_are_high_altitude():
    start = datetime(2026, 1, 1, 12, 0)
    messages = [
        _row(start, 0.00, on_ground=True, ground_speed_kt=0.0),
        _row(start + timedelta(minutes=1), 0.02, on_ground=False, ground_speed_kt=120.0, baro_altitude_ft=1000),
        _row(start + timedelta(minutes=20), 1.20, on_ground=False, ground_speed_kt=300.0, baro_altitude_ft=35_000),
        _row(start + timedelta(hours=2), 1.25, on_ground=False, ground_speed_kt=300.0, baro_altitude_ft=35_000),
        _row(start + timedelta(hours=2, minutes=20), 1.45, on_ground=False, ground_speed_kt=95.0, baro_altitude_ft=700),
        _row(start + timedelta(hours=2, minutes=23), 1.47, on_ground=False, ground_speed_kt=50.0, baro_altitude_ft=400),
    ]

    segments = identify_flight_segments(messages)

    assert len(segments) == 1
    assert segments[0].takeoff_time == start + timedelta(minutes=1)
    assert segments[0].landing_time == start + timedelta(hours=2, minutes=20)


def test_backdates_takeoff_when_first_airborne_message_is_already_high():
    start = datetime(2026, 1, 1, 12, 0)
    messages = [
        _row(start, 0.00, on_ground=True, ground_speed_kt=0.0),
        _row(start + timedelta(minutes=10), 0.20, on_ground=False, ground_speed_kt=160.0, baro_altitude_ft=4000),
        _row(start + timedelta(minutes=20), 0.50, on_ground=False, ground_speed_kt=100.0, baro_altitude_ft=2500),
        _row(start + timedelta(minutes=23), 0.52, on_ground=False, ground_speed_kt=50.0, baro_altitude_ft=400),
    ]

    segments = identify_flight_segments(messages)

    assert len(segments) == 1
    assert segments[0].takeoff_time == start + timedelta(minutes=5)
    assert segments[0].landing_time == start + timedelta(minutes=20)


def test_decelerating_approach_can_identify_landing_above_absolute_landing_speed():
    start = datetime(2026, 1, 1, 12, 0)
    messages = [
        _row(start, 0.00, on_ground=True, ground_speed_kt=0.0),
        _row(start + timedelta(minutes=1), 0.02, on_ground=False, ground_speed_kt=90.0),
        _row(start + timedelta(minutes=8), 0.20, on_ground=False, ground_speed_kt=210.0),
        _row(start + timedelta(minutes=12), 0.28, on_ground=False, ground_speed_kt=125.0, baro_altitude_ft=1500),
        _row(start + timedelta(minutes=15), 0.30, on_ground=False, ground_speed_kt=100.0, baro_altitude_ft=1000),
    ]

    segments = identify_flight_segments(messages)

    assert len(segments) == 1
    assert segments[0].landing_time == start + timedelta(minutes=12)


def test_open_segment_ending_high_and_fast_is_rejected():
    start = datetime(2026, 1, 1, 12, 0)
    messages = [
        _row(start, 0.00, on_ground=True, ground_speed_kt=0.0),
        _row(start + timedelta(minutes=1), 0.02, on_ground=False, ground_speed_kt=100.0, baro_altitude_ft=500),
        _row(start + timedelta(minutes=20), 1.00, on_ground=False, ground_speed_kt=230.0, baro_altitude_ft=30_000),
    ]

    segments = identify_flight_segments(messages)

    assert segments == []


def test_rejects_segment_when_distance_travelled_is_too_short():
    start = datetime(2026, 1, 1, 12, 0)
    messages = [
        _row(start, 0.00),
        _row(start + timedelta(minutes=1), 0.02),
        _row(start + timedelta(minutes=4), 0.0201),
        _row(start + timedelta(minutes=7), 0.0202),
    ]

    segments = identify_flight_segments(messages)

    assert segments == []


def test_segments_to_flights_df_uses_flight_schema_defaults(monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", "/tmp")
    from data_engineering.flights.flight_type import FLIGHT_POLARS_SCHEMA

    start = datetime(2026, 1, 1, 12, 0)
    message = _row(start, 0.0)
    df = _segments_to_flights_df(
        [
            FlightSegment(
                icao="abc123",
                takeoff_time=start,
                landing_time=start + timedelta(minutes=10),
                takeoff_message=message,
                landing_message=message,
            )
        ]
    )

    assert df.columns == list(FLIGHT_POLARS_SCHEMA.keys())
    assert df.row(0, named=True)["icao"] == "abc123"
    assert df.row(0, named=True)["takeoff_airport_ident"] is None
    assert df.row(0, named=True)["pia"] is False


def test_segment_filter_keeps_only_flights_that_take_off_on_target_day():
    day_start = datetime(2026, 1, 1)
    day_end = day_start + timedelta(days=1)
    message = _row(day_start, 0.0)

    before_day = FlightSegment(
        icao="abc123",
        takeoff_time=day_start - timedelta(minutes=10),
        landing_time=day_start + timedelta(minutes=20),
        takeoff_message=message,
        landing_message=message,
    )
    during_day = FlightSegment(
        icao="abc123",
        takeoff_time=day_start + timedelta(hours=1),
        landing_time=day_start + timedelta(hours=2),
        takeoff_message=message,
        landing_message=message,
    )
    after_day = FlightSegment(
        icao="abc123",
        takeoff_time=day_end,
        landing_time=day_end + timedelta(minutes=30),
        takeoff_message=message,
        landing_message=message,
    )

    assert not _segment_takes_off_in_window(before_day, day_start, day_end)
    assert _segment_takes_off_in_window(during_day, day_start, day_end)
    assert not _segment_takes_off_in_window(after_day, day_start, day_end)


def test_run_main_2_reads_readsb_hist_day_parts_and_filters_icaos(tmp_path):
    target_date = date(2026, 3, 1)
    day_dir = tmp_path / "year=2026" / "month=03" / "day=01"
    day_dir.mkdir(parents=True)
    start = datetime(2026, 3, 1, 12, 0)

    rows = [
        {"time": start, "icao": "abc123", "lat": 0.0, "lon": 0.00, "ground_speed_kt": 0.0, "on_ground": True, "baro_altitude_ft": None},
        {"time": start + timedelta(minutes=1), "icao": "abc123", "lat": 0.0, "lon": 0.02, "ground_speed_kt": 100.0, "on_ground": False, "baro_altitude_ft": 1000},
        {"time": start + timedelta(minutes=8), "icao": "abc123", "lat": 0.0, "lon": 0.20, "ground_speed_kt": 180.0, "on_ground": False, "baro_altitude_ft": 5000},
        {"time": start + timedelta(minutes=12), "icao": "abc123", "lat": 0.0, "lon": 0.205, "ground_speed_kt": 20.0, "on_ground": True, "baro_altitude_ft": None},
        {"time": start + timedelta(minutes=15), "icao": "abc123", "lat": 0.0, "lon": 0.205, "ground_speed_kt": 0.0, "on_ground": True, "baro_altitude_ft": None},
        {"time": start, "icao": "def456", "lat": 0.0, "lon": 1.00, "ground_speed_kt": 0.0, "on_ground": True, "baro_altitude_ft": None},
        {"time": start + timedelta(minutes=1), "icao": "def456", "lat": 0.0, "lon": 1.02, "ground_speed_kt": 100.0, "on_ground": False, "baro_altitude_ft": 1000},
        {"time": start + timedelta(minutes=8), "icao": "def456", "lat": 0.0, "lon": 1.20, "ground_speed_kt": 180.0, "on_ground": False, "baro_altitude_ft": 5000},
        {"time": start + timedelta(minutes=12), "icao": "def456", "lat": 0.0, "lon": 1.205, "ground_speed_kt": 20.0, "on_ground": True, "baro_altitude_ft": None},
        {"time": start + timedelta(minutes=15), "icao": "def456", "lat": 0.0, "lon": 1.205, "ground_speed_kt": 0.0, "on_ground": True, "baro_altitude_ft": None},
    ]
    pl.DataFrame(rows).write_parquet(day_dir / "part-00000.parquet")

    df = run_main_2(target_date, icaos=["abc123"], parquet_root=tmp_path)

    assert df.height == 1
    assert df.row(0, named=True)["icao"] == "abc123"
    assert df.row(0, named=True)["takeoff_time"] == start + timedelta(minutes=1)
    assert df.row(0, named=True)["landing_time"] == start + timedelta(minutes=12)


def test_stitches_same_icao_segments_separated_by_short_gap():
    start = datetime(2026, 1, 1, 12, 0)
    message = _row(start, 0.0)
    segments = [
        FlightSegment(
            icao="abc123",
            takeoff_time=start,
            landing_time=start + timedelta(minutes=30),
            takeoff_message=message,
            landing_message=message,
            distance_travelled_km=20.0,
        ),
        FlightSegment(
            icao="abc123",
            takeoff_time=start + timedelta(hours=1),
            landing_time=start + timedelta(hours=2),
            takeoff_message=message,
            landing_message=message,
            distance_travelled_km=40.0,
        ),
    ]

    stitched = stitch_flight_segments(segments)

    assert len(stitched) == 1
    assert stitched[0].takeoff_time == start
    assert stitched[0].landing_time == start + timedelta(hours=2)
    assert stitched[0].distance_travelled_km == 60.0
