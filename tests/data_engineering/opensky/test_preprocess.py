from datetime import datetime, timedelta

import polars as pl

from data_engineering.opensky.preprocess import preprocess_opensky_adsb


START = datetime(2026, 3, 1)


def _rows(
    *,
    seconds: list[int],
    lat: list[float],
    lon: list[float],
    speed: list[float | None],
    altitude: list[float | None],
    on_ground: list[bool | None],
) -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "time": [START + timedelta(seconds=value) for value in seconds],
            "icao": ["abc123"] * len(seconds),
            "lat": lat,
            "lon": lon,
            "ground_speed_kt": speed,
            "baro_altitude_ft": altitude,
            "on_ground": on_ground,
            "callsign": ["TEST1"] * len(seconds),
        },
        schema_overrides={
            "ground_speed_kt": pl.Float64,
            "baro_altitude_ft": pl.Float64,
            "on_ground": pl.Boolean,
        },
    ).lazy()


def test_removes_isolated_position_spike_between_plausible_neighbors():
    df = _rows(
        seconds=[0, 10, 20],
        lat=[40.0, 45.0, 40.01],
        lon=[-75.0, -100.0, -74.99],
        speed=[250.0, 250.0, 250.0],
        altitude=[10_000.0, 10_000.0, 10_000.0],
        on_ground=[False, False, False],
    )

    result = preprocess_opensky_adsb(df).collect()

    assert result.height == 2
    assert result.get_column("time").to_list() == [
        START,
        START + timedelta(seconds=20),
    ]


def test_keeps_fast_but_physically_plausible_positions():
    df = _rows(
        seconds=[0, 10, 20],
        lat=[40.0, 40.01, 40.02],
        lon=[-75.0, -74.98, -74.96],
        speed=[500.0, 500.0, 500.0],
        altitude=[10_000.0, 10_100.0, 10_200.0],
        on_ground=[False, False, False],
    )

    result = preprocess_opensky_adsb(df).collect()

    assert result.height == 3


def test_nulls_invalid_and_isolated_altitude_spikes_without_dropping_rows():
    df = _rows(
        seconds=[0, 10, 20, 30, 40],
        lat=[40.0, 40.01, 40.02, 40.03, 40.04],
        lon=[-75.0, -74.99, -74.98, -74.97, -74.96],
        speed=[250.0] * 5,
        altitude=[10_000.0, 50_000.0, 10_200.0, 70_000.0, 10_400.0],
        on_ground=[False] * 5,
    )

    result = preprocess_opensky_adsb(df).collect()

    assert result.height == 5
    assert result.get_column("baro_altitude_ft").to_list() == [
        10_000.0,
        None,
        10_200.0,
        None,
        10_400.0,
    ]


def test_stabilizes_one_message_ground_state_flip():
    df = _rows(
        seconds=[0, 10, 20],
        lat=[40.0, 40.0, 40.0],
        lon=[-75.0, -75.0, -75.0],
        speed=[0.1, 0.1, 0.1],
        altitude=[None, None, None],
        on_ground=[True, False, True],
    )

    result = preprocess_opensky_adsb(df).collect()

    assert result.get_column("on_ground").to_list() == [True, True, True]


def test_exposes_unobserved_high_altitude_stop_to_readsb():
    df = _rows(
        seconds=[0, 10 * 60, 70 * 60, 80 * 60],
        lat=[40.0, 40.1, 41.0, 41.1],
        lon=[-100.0, -100.1, -101.0, -101.1],
        speed=[300.0] * 4,
        altitude=[30_000.0, 18_000.0, 25_000.0, 32_000.0],
        on_ground=[False] * 4,
    )

    result = preprocess_opensky_adsb(df).collect()

    assert result.get_column("baro_altitude_ft").to_list() == [
        30_000.0,
        18_000.0,
        None,
        32_000.0,
    ]
    assert result.get_column("on_ground").to_list() == [
        False,
        False,
        True,
        False,
    ]


def test_keeps_high_altitude_after_short_receiver_gap():
    df = _rows(
        seconds=[0, 30 * 60],
        lat=[40.0, 41.0],
        lon=[-100.0, -101.0],
        speed=[300.0, 300.0],
        altitude=[18_000.0, 25_000.0],
        on_ground=[False, False],
    )

    result = preprocess_opensky_adsb(df).collect()

    assert result.get_column("baro_altitude_ft").to_list() == [
        18_000.0,
        25_000.0,
    ]
    assert result.get_column("on_ground").to_list() == [False, False]


def test_keeps_high_altitude_after_long_in_flight_receiver_gap():
    df = _rows(
        seconds=[0, 10 * 60, 70 * 60, 80 * 60],
        lat=[40.0, 40.1, 41.0, 41.1],
        lon=[-100.0, -100.1, -101.0, -101.1],
        speed=[300.0] * 4,
        altitude=[32_000.0, 32_000.0, 32_000.0, 32_000.0],
        on_ground=[False] * 4,
    )

    result = preprocess_opensky_adsb(df).collect()

    assert result.get_column("baro_altitude_ft").to_list() == [
        32_000.0,
        32_000.0,
        32_000.0,
        32_000.0,
    ]
    assert result.get_column("on_ground").to_list() == [False] * 4


def test_exposes_unobserved_stop_when_callsign_changes():
    df = _rows(
        seconds=[0, 60 * 60],
        lat=[40.0, 41.0],
        lon=[-100.0, -101.0],
        speed=[300.0, 300.0],
        altitude=[32_000.0, 32_000.0],
        on_ground=[False, False],
    ).with_columns(
        pl.Series("callsign", ["TEST1", "TEST2"])
    )

    result = preprocess_opensky_adsb(df).collect()

    assert result.get_column("baro_altitude_ft").to_list() == [
        32_000.0,
        None,
    ]
    assert result.get_column("on_ground").to_list() == [False, True]
