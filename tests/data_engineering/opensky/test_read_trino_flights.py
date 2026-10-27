from datetime import date, datetime, timedelta, timezone

import polars as pl
import pytest

from data_engineering.opensky.read_trino_flights import read_opensky_flights
from data_engineering.opensky.paths import raw_flights_day_path


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
    next_path = raw_flights_day_path(day + timedelta(days=1), root=tmp_path)
    next_path.parent.mkdir(parents=True)
    pl.read_parquet(input_path).head(0).write_parquet(next_path)

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


@pytest.mark.parametrize("day", [date(2024, 6, 1), date(2024, 12, 31)])
def test_groups_flights_by_takeoff_day_across_raw_partitions(tmp_path, day):
    start = int(datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc).timestamp())
    for offset, icaos, takeoffs, landings in [
        (0, ["previous", "midnight", "same_day"],
         [start - 1, start, start + 3600],
         [start + 600, start + 1800, start + 7200]),
        (1, ["overnight", "next_day"],
         [start + 86399, start + 86400],
         [start + 90000, start + 93600]),
    ]:
        path = raw_flights_day_path(day + timedelta(days=offset), root=tmp_path)
        path.parent.mkdir(parents=True)
        pl.DataFrame({
            "icao24": icaos,
            "firstseen": takeoffs,
            "lastseen": landings,
            "callsign": [" TEST "] * len(icaos),
            "estdepartureairport": ["KJFK"] * len(icaos),
            "estarrivalairport": ["KBOS"] * len(icaos),
        }).write_parquet(path)

    result = read_opensky_flights(day, input_dir=tmp_path).sort("takeoff_time")

    assert result["icao"].to_list() == ["midnight", "same_day", "overnight"]
    assert result["takeoff_time"].dt.date().to_list() == [day] * 3
    assert result["landing_time"].dt.date().to_list() == [day, day, day + timedelta(days=1)]


def test_missing_next_day_download_is_an_error(tmp_path):
    day = date(2024, 6, 1)
    path = raw_flights_day_path(day, root=tmp_path)
    path.parent.mkdir(parents=True)
    pl.DataFrame({"firstseen": []}).write_parquet(path)

    with pytest.raises(FileNotFoundError, match="download_trino_flights 2024-06-02"):
        read_opensky_flights(day, input_dir=tmp_path)
