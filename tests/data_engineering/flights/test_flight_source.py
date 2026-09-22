from datetime import date
from pathlib import Path

import polars as pl

from flights import flight_source as flight_source_module
from data_engineering.adsb.read_adsb import ADSBSource
from flights.algorithm import AlgorithmNearestFlightSource
from flights.flight_source import (
    write_flights,
    flight_path,
)
from flights.providers.bts import BTSFlightSource


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


def test_algorithm_nearest_path_includes_adsb_source(monkeypatch, tmp_path):
    monkeypatch.setattr(flight_source_module, "OUTPUT_DIR", tmp_path)

    path = flight_path(
        AlgorithmNearestFlightSource(),
        date(2026, 3, 1),
        ADSBSource.ADSBLOL,
    )

    assert path == Path(
        tmp_path,
        "data/flights/source=algorithm-nearest/adsb_source=adsblol/version=3",
        "year=2026/month=03/day=01/data.parquet",
    )
