from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest
import pyarrow.parquet as pq

from data_engineering.adsb.parquet_schema import PARQUET_SCHEMA
from data_engineering.opensky import read_opensky_trino_states as opensky_reader
from data_engineering.opensky.read_opensky_trino_states import (
    MAX_STATE_VECTOR_AGE_SECONDS,
    OPENAIRFRAMES_COLUMNS,
    normalize_opensky_state_vectors,
    opensky_processed_state_vectors_path,
)


def _raw_state_vectors() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "time": [101, 102, 110, 110, 110, 110],
            "icao24": ["abc123"] * 5 + ["def456"],
            "lastcontact": [100.9, 101.9, 100.0, 109.0, None, 111.5],
            "lastposupdate": [100.5, 100.5, 109.0, 100.0, 109.0, 109.0],
            "velocity": [10.0, 10.0, 20.0, 20.0, 20.0, 30.0],
            "heading": [90.0] * 6,
            "vertrate": [1.0] * 6,
            "onground": [False] * 6,
            "baroaltitude": [1_000.0] * 6,
            "geoaltitude": [1_100.0] * 6,
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
            "lastcontact": [110 - max_age, 110 - max_age - 0.001],
            "lastposupdate": [110 - max_age, 110 - max_age - 0.001],
            "velocity": [10.0, 10.0],
            "heading": [90.0, 90.0],
            "vertrate": [1.0, 1.0],
            "onground": [False, False],
            "baroaltitude": [1_000.0, 1_000.0],
            "geoaltitude": [1_100.0, 1_100.0],
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
        match="lastcontact, lastposupdate",
    ):
        normalize_opensky_state_vectors(raw)


def test_processed_cache_uses_canonical_partition_path():
    path = opensky_processed_state_vectors_path("2026-03-01")

    assert path.as_posix().endswith(
        "data/processed/adsb/parquet_output/source=opensky/version=4/"
        "year=2026/month=03/day=01"
    )


def _write_raw_hour(path: Path, start: datetime) -> None:
    timestamp = int(start.timestamp())
    pl.DataFrame(
        {
            "time": [timestamp, timestamp],
            "icao24": ["010000", "f10000"],
            "lat": [40.0, 41.0],
            "lon": [-75.0, -76.0],
            "velocity": [100.0, 110.0],
            "heading": [90.0, 180.0],
            "vertrate": [0.0, 1.0],
            "callsign": ["ONE", "TWO"],
            "onground": [False, False],
            "alert": [False, False],
            "spi": [False, False],
            "squawk": ["1200", "1200"],
            "baroaltitude": [1_000.0, 2_000.0],
            "geoaltitude": [1_100.0, 2_100.0],
            "lastposupdate": [float(timestamp), float(timestamp)],
            "lastcontact": [float(timestamp), float(timestamp)],
            "serials": [[1], [2]],
            "hour": [timestamp, timestamp],
        }
    ).write_parquet(path, mkdir=True)


def _append_position_update(
    path: Path,
    *,
    snapshot_time: float,
    position_time: float,
) -> None:
    existing = pl.read_parquet(path)
    extra = existing.head(1).with_columns(
        pl.lit(int(snapshot_time), dtype=existing.schema["time"]).alias("time"),
        pl.lit(position_time).alias("lastposupdate"),
        pl.lit(snapshot_time).alias("lastcontact"),
        pl.lit("BOUNDARY").alias("callsign"),
    )
    pl.concat([existing, extra]).write_parquet(path)


def _stub_openairframes(monkeypatch) -> None:
    from data_engineering.openairframes import read as openairframes_reader

    metadata = {
        "icao": ["010000", "f10000"],
        **{
            column: (
                [False, False]
                if column in {"pia", "ladd", "military", "interesting"}
                else [None, None]
            )
            for column in OPENAIRFRAMES_COLUMNS
        },
    }
    monkeypatch.setattr(
        openairframes_reader,
        "scan_latest_icao_info_for_join",
        lambda **_kwargs: pl.DataFrame(metadata).lazy(),
    )


def _scan_processed_day(processed_day: Path) -> pl.LazyFrame:
    return pl.scan_parquet(
        sorted(processed_day.glob("icao_bucket=*/*.parquet")),
        hive_partitioning=False,
    )


def test_cache_reads_new_hour_layout_and_prunes_icao_buckets(
    monkeypatch,
    tmp_path,
):
    raw_day = tmp_path / "raw"
    processed_day = tmp_path / "processed"
    monkeypatch.setattr(
        opensky_reader,
        "raw_state_vectors_day_dir",
        lambda day: raw_day,
    )
    monkeypatch.setattr(
        opensky_reader,
        "opensky_processed_state_vectors_path",
        lambda day: processed_day,
    )
    _stub_openairframes(monkeypatch)

    start = datetime(2024, 6, 1, tzinfo=timezone.utc)
    for hour in range(24):
        _write_raw_hour(
            raw_day / f"hour={hour:02d}" / "data.parquet",
            start + timedelta(hours=hour),
        )

    opensky_reader.write_opensky_state_vectors_cache_for_day(date(2024, 6, 1))
    result = (
        _scan_processed_day(processed_day)
        .filter(pl.col("icao") == "010000")
        .select("time", "icao", "geom_altitude_ft", "baro_rate_fpm")
        .collect()
    )

    assert result.height == 24
    assert result.get_column("icao").unique().to_list() == ["010000"]
    assert result.get_column("geom_altitude_ft").item(0) == 3_608
    assert result.get_column("baro_rate_fpm").item(0) == 0
    assert (processed_day / "_SUCCESS").exists()
    assert sorted(path.name for path in processed_day.glob("icao_bucket=*")) == [
        "icao_bucket=0",
        "icao_bucket=7",
    ]
    assert len(list((processed_day / "icao_bucket=0").glob("*.parquet"))) == 1
    assert len(list((processed_day / "icao_bucket=7").glob("*.parquet"))) == 1
    assert not (processed_day / "_normalized").exists()
    for path in processed_day.glob("icao_bucket=*/*.parquet"):
        keys = pl.read_parquet(path).select("icao", "time")
        assert keys.equals(keys.sort(["icao", "time"]))
    assert all(
        pq.read_schema(path).equals(PARQUET_SCHEMA)
        for path in processed_day.glob("icao_bucket=*/*.parquet")
    )


def test_hourly_cache_deduplicates_position_update_across_hour_boundary(
    monkeypatch,
    tmp_path,
):
    raw_day = tmp_path / "raw"
    processed_day = tmp_path / "processed"
    monkeypatch.setattr(
        opensky_reader,
        "raw_state_vectors_day_dir",
        lambda day: raw_day,
    )
    monkeypatch.setattr(
        opensky_reader,
        "opensky_processed_state_vectors_path",
        lambda day: processed_day,
    )
    _stub_openairframes(monkeypatch)

    start = datetime(2024, 6, 1, tzinfo=timezone.utc)
    for hour in range(24):
        _write_raw_hour(
            raw_day / f"hour={hour:02d}" / "data.parquet",
            start + timedelta(hours=hour),
        )

    hour_boundary = (start + timedelta(hours=1)).timestamp()
    position_time = hour_boundary - 0.5
    _append_position_update(
        raw_day / "hour=00" / "data.parquet",
        snapshot_time=hour_boundary - 1,
        position_time=position_time,
    )
    _append_position_update(
        raw_day / "hour=01" / "data.parquet",
        snapshot_time=hour_boundary,
        position_time=position_time,
    )

    opensky_reader.write_opensky_state_vectors_cache_for_day(date(2024, 6, 1))
    result = (
        _scan_processed_day(processed_day)
        .filter(pl.col("icao") == "010000")
        .select("time", "icao", "callsign")
        .collect()
    )

    boundary_time = datetime(2024, 6, 1, 0, 59, 59, 500_000)
    assert result.filter(pl.col("time") == boundary_time).height == 1
    assert result.height == 25
