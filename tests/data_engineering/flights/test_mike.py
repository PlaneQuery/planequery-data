from datetime import date, datetime

import polars as pl

from flights import flight_source as flight_source_module
from flights.flight_source import flight_path, write_flights
from flights.flight_type import FLIGHT_POLARS_SCHEMA
from flights.providers.mike import MikeFlightSource, create_mike_flights


def _write_input(path):
    pl.DataFrame(
        {
            "icao": [" ABC123 ", "def456"],
            "takeoff_ts": [
                "2026-03-01 10:00:00.000 UTC",
                "2026-03-02 10:00:00.000 UTC",
            ],
            "landing_ts": [
                "2026-03-01 12:00:00.000 UTC",
                "2026-03-02 12:00:00.000 UTC",
            ],
            "takeoff_airport_ident": ["KJFK", "KBOS"],
            "landing_airport_ident": ["KLAX", "KSFO"],
            "owner_operator": ["Example Air", "Other Air"],
        }
    ).write_csv(path)


def test_create_mike_flights_normalizes_and_filters_one_day(tmp_path):
    input_csv = tmp_path / "mike.csv"
    _write_input(input_csv)

    result = create_mike_flights(date(2026, 3, 1), input_csv)

    assert result.schema == FLIGHT_POLARS_SCHEMA
    assert result.height == 1
    row = result.row(0, named=True)
    assert row["takeoff_time"] == datetime(2026, 3, 1, 10)
    assert row["landing_time"] == datetime(2026, 3, 1, 12)
    assert result["icao"].to_list() == ["abc123"]
    assert result["owner"].to_list() == ["Example Air"]


def test_write_mike_flights_uses_canonical_daily_path(monkeypatch, tmp_path):
    input_csv = tmp_path / "mike.csv"
    _write_input(input_csv)
    monkeypatch.setattr(flight_source_module, "OUTPUT_DIR", tmp_path / "output")
    source = MikeFlightSource(input_csv=input_csv)
    target_date = date(2026, 3, 1)

    result = write_flights(source, target_date)
    path = flight_path(source, target_date)

    assert path.exists()
    assert pl.read_parquet(path).equals(result)
    assert result.height == 1
