from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import polars as pl
import pytest

from flights import flight_source as flight_source_module
from data_engineering.adsb.read_adsb import ADSBSource
from flights.algorithm import AlgorithmNearestFlightSource
from flights.flight_source import (
    ADSBFlightSource,
    NonADSBFlightSource,
    get_flights,
    write_flights,
    flight_path,
)
from flights.providers.bts import BTSFlightSource


@dataclass(frozen=True)
class DummyADSBFlightSource(ADSBFlightSource):
    name: str = "dummy-adsb"

    def adsb_read_interval(self, dt):
        return datetime(2026, 3, 1, 1), datetime(2026, 3, 2, 2)

    def create_flights_for_day(self, dt, df_adsb):
        return df_adsb.select("icao")


@dataclass(frozen=True)
class DummyNonADSBFlightSource(NonADSBFlightSource):
    name: str = "dummy-non-adsb"

    def create_flights_for_day(self, dt):
        return pl.DataFrame({"icao": ["abc123"]})


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


def test_get_flights_dispatches_adsb_source_and_interval(monkeypatch, tmp_path):
    target_date = date(2026, 3, 1)
    calls = []
    monkeypatch.setattr(flight_source_module, "OUTPUT_DIR", tmp_path)

    def read_adsb(start, end, *, source):
        calls.append((start, end, source))
        return pl.DataFrame({"icao": ["abc123"]})

    monkeypatch.setattr(flight_source_module, "read_adsb", read_adsb)

    result = get_flights(
        DummyADSBFlightSource(),
        target_date,
        ADSBSource.ADSBX,
        use_cache=False,
    )

    assert calls == [
        (
            datetime(2026, 3, 1, 1),
            datetime(2026, 3, 2, 2),
            ADSBSource.ADSBX,
        )
    ]
    assert result.get_column("icao").to_list() == ["abc123"]


def test_get_flights_uses_cache_before_creating(monkeypatch, tmp_path):
    target_date = date(2026, 3, 1)
    source = DummyADSBFlightSource()
    monkeypatch.setattr(flight_source_module, "OUTPUT_DIR", tmp_path)
    path = flight_path(source, target_date, ADSBSource.ADSBX)
    path.parent.mkdir(parents=True)
    expected = pl.DataFrame({"icao": ["cached"]})
    expected.write_parquet(path)

    def fail_read(*args, **kwargs):
        raise AssertionError("cached flights should not read ADS-B")

    monkeypatch.setattr(flight_source_module, "read_adsb", fail_read)

    result = get_flights(source, target_date, ADSBSource.ADSBX)

    assert result.equals(expected)


def test_flight_path_requires_adsb_source_only_for_adsb_flights(tmp_path, monkeypatch):
    monkeypatch.setattr(flight_source_module, "OUTPUT_DIR", tmp_path)

    with pytest.raises(AssertionError):
        flight_path(DummyADSBFlightSource(), date(2026, 3, 1))
    with pytest.raises(AssertionError):
        flight_path(
            DummyNonADSBFlightSource(),
            date(2026, 3, 1),
            ADSBSource.ADSBX,
        )
