from datetime import datetime

import polars as pl
import pytest

from data_engineering.opensky.read_opensky_trino_states import (
    MAX_STATE_VECTOR_AGE_SECONDS,
    normalize_opensky_state_vectors,
    opensky_processed_state_vectors_path,
)


def _raw_state_vectors() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "time": [101, 102, 110, 110, 110, 110],
            "icao24": ["abc123"] * 5 + ["def456"],
            "lastContact": [100.9, 101.9, 100.0, 109.0, None, 111.5],
            "lastPosUpdate": [100.5, 100.5, 109.0, 100.0, 109.0, 109.0],
            "velocity": [10.0, 10.0, 20.0, 20.0, 20.0, 30.0],
            "heading": [90.0] * 6,
            "onGround": [False] * 6,
            "baroAltitude": [1_000.0] * 6,
            "lat": [40.0] * 6,
            "lon": [-75.0] * 6,
            "callsign": ["TEST"] * 6,
        }
    ).lazy()


def test_normalize_uses_position_time_and_removes_stale_repeated_snapshots():
    result = normalize_opensky_state_vectors(_raw_state_vectors()).collect()

    assert result.height == 1
    assert result.get_column("icao").to_list() == ["abc123"]
    assert result.get_column("time").to_list() == [
        datetime(1970, 1, 1, 0, 1, 40, 500_000)
    ]
    assert result.get_column("ground_speed_kt").item() == pytest.approx(19.4384)
    assert result.get_column("baro_altitude_ft").item() == pytest.approx(3_280.84)


def test_normalize_keeps_freshness_boundary_and_rejects_older_snapshot():
    max_age = MAX_STATE_VECTOR_AGE_SECONDS
    raw = pl.DataFrame(
        {
            "time": [110, 110],
            "icao24": ["abc123", "def456"],
            "lastContact": [110 - max_age, 110 - max_age - 0.001],
            "lastPosUpdate": [110 - max_age, 110 - max_age - 0.001],
            "velocity": [10.0, 10.0],
            "heading": [90.0, 90.0],
            "onGround": [False, False],
            "baroAltitude": [1_000.0, 1_000.0],
            "lat": [40.0, 40.0],
            "lon": [-75.0, -75.0],
            "callsign": ["FRESH", "STALE"],
        }
    ).lazy()

    result = normalize_opensky_state_vectors(raw).collect()

    assert result.get_column("icao").to_list() == ["abc123"]


def test_normalize_requires_freshness_columns():
    raw = pl.DataFrame(
        {
            "time": [110],
            "icao24": ["abc123"],
        }
    ).lazy()

    with pytest.raises(
        ValueError,
        match="lastContact, lastPosUpdate",
    ):
        normalize_opensky_state_vectors(raw)


def test_processed_cache_uses_v2_path():
    path = opensky_processed_state_vectors_path("2026-03-01")

    assert "data/processed/opensky/v2" in path.as_posix()
