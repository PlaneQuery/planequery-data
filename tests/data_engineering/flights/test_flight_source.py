from datetime import date

import polars as pl

from data_engineering.flights import flight_source as flight_source_module
from data_engineering.flights.flight_source import (
    BTSFlightSource,
    write_flights,
    flight_path,
)


def test_create_flights_writes_single_bts_day_to_canonical_path(
    monkeypatch,
    tmp_path,
):
    target_date = date(2026, 3, 1)
    expected = pl.DataFrame({"icao": ["abc123"]})
    calls = []

    monkeypatch.setattr(flight_source_module, "OUTPUT_DIR", tmp_path)

    def create_day(self, dt):
        calls.append(dt)
        return expected

    monkeypatch.setattr(BTSFlightSource, "create_flights_for_day", create_day)

    result = write_flights(BTSFlightSource(), target_date)
    path = flight_path(BTSFlightSource(), target_date)

    assert calls == [target_date]
    assert result.equals(expected)
    assert pl.read_parquet(path).equals(expected)
