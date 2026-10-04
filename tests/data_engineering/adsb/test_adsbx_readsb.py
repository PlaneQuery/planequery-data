import gzip
import json
from datetime import date, datetime, timezone

import polars as pl
import pyarrow.parquet as pq
import pytest

from data_engineering.adsb.parquet_schema import PARQUET_SCHEMA
from data_engineering.adsbx.download_readsb_day import DEFAULT_OUT_DIR
from data_engineering.adsbx.published_flights import _adsbx_raw_arrivals_path
from data_engineering.adsbx.readsb import (
    DEFAULT_INPUT_ROOT,
    read_snapshot,
    readsb_snapshot_files,
    write_readsb_day_to_bucketed_parquet,
)


def test_adsbx_raw_paths_use_one_source_name():
    raw_flights_path = _adsbx_raw_arrivals_path(date(2024, 6, 1))

    assert DEFAULT_INPUT_ROOT == DEFAULT_OUT_DIR
    assert "adsbx" in DEFAULT_INPUT_ROOT.parts
    assert "adsbx" in raw_flights_path.parts
    assert "adsb-exchange" not in DEFAULT_INPUT_ROOT.parts
    assert "adsb-exchange" not in raw_flights_path.parts


def test_adsbx_writes_canonical_bucketed_day(tmp_path):
    input_day = tmp_path / "input" / "2024" / "06" / "01"
    input_day.mkdir(parents=True)
    timestamp = datetime(2024, 6, 1, tzinfo=timezone.utc).timestamp()
    snapshot = {
        "now": timestamp,
        "aircraft": [
            {
                "hex": "010000",
                "seen_pos": 0,
                "lat": 40.0,
                "lon": -75.0,
                "alt_baro": 10_000,
            }
        ],
    }
    for second in (0, 5, 10, 15):
        snapshot["now"] = timestamp + second
        with gzip.open(input_day / f"0000{second:02d}Z.json.gz", "wt") as file:
            json.dump(snapshot, file)

    output_root = tmp_path / "canonical" / "source=adsbx" / "version=2"
    write_readsb_day_to_bucketed_parquet(input_day, output_root=output_root)

    output_day = output_root / "year=2024/month=06/day=01"
    assert (output_day / "_SUCCESS").is_file()
    assert pq.read_schema(
        output_day / "icao_bucket=0/data.parquet"
    ).equals(PARQUET_SCHEMA)
    result = pl.read_parquet(output_day / "icao_bucket=0/data.parquet")
    assert result.get_column("icao").to_list() == ["010000"] * 4
    assert result.get_column("time").dt.second().to_list() == [0, 5, 10, 15]
    assert [p.name for p in readsb_snapshot_files(input_day, start_after="000005Z.json.gz")] == [
        "000010Z.json.gz", "000015Z.json.gz"
    ]


@pytest.mark.parametrize("field,value", [
    ("hex", ""), ("lat", None), ("lon", None), ("seen_pos", None),
    ("lat", "nan"), ("lon", "inf"),
    ("lat", 91), ("lon", -181), ("seen_pos", -1), ("seen_pos", "bad"),
])
def test_snapshot_requires_valid_timed_positions(tmp_path, field, value):
    aircraft = {"hex": "010000", "lat": 40, "lon": -75, "seen_pos": 2.5}
    invalid = {**aircraft, field: value}
    path = tmp_path / "000005Z.json.gz"
    with gzip.open(path, "wt") as file:
        json.dump({"now": 1717200005, "aircraft": [invalid, aircraft]}, file)

    result = read_snapshot(path)
    assert result["icao"] == ["010000"]
    assert result["time"] == [1717200002500]


@pytest.mark.parametrize("now", [None, "bad", "inf", -1])
def test_snapshot_rejects_invalid_time(tmp_path, now):
    path = tmp_path / "000005Z.json.gz"
    with gzip.open(path, "wt") as file:
        json.dump({"now": now, "aircraft": [
            {"hex": "010000", "lat": 40, "lon": -75, "seen_pos": 0}
        ]}, file)
    assert read_snapshot(path)["time"] == []
