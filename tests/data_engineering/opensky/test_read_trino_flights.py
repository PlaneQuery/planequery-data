from datetime import date, datetime

import polars as pl
import pytest

from data_engineering.opensky.read_trino_flights import read_opensky_flights


def test_reads_raw_trino_download_directly(tmp_path):
    day = date(2024, 6, 1)
    input_path = (
        tmp_path
        / "year=2024"
        / "month=06"
        / "day=01"
        / "data.parquet"
    )
    input_path.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "icao24": ["abc123"],
            "firstseen": [1717200000],
            "lastseen": [1717203600],
            "estdepartureairport": ["KJFK"],
            "estarrivalairport": ["KBOS"],
            "callsign": [" TEST123 "],
            "track": [[{"time": 1717200000}]],
            "day": [1717200000],
        }
    ).write_parquet(input_path)

    result = read_opensky_flights(day, input_dir=tmp_path)

    assert result.to_dicts() == [
        {
            "icao": "abc123",
            "callsign": "TEST123",
            "takeoff_time": datetime(2024, 6, 1, 0, 0),
            "landing_time": datetime(2024, 6, 1, 1, 0),
            "takeoff_airport_ident": "KJFK",
            "landing_airport_ident": "KBOS",
            "pia": False,
            "ladd": False,
            "military": False,
            "interesting": False,
        }
    ]


def test_missing_download_explains_how_to_create_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_trino_flights 2024-06-01"):
        read_opensky_flights(date(2024, 6, 1), input_dir=tmp_path)
