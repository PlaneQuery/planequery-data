from dataclasses import dataclass
from datetime import date

import polars as pl
import pytest

from data_engineering.adsb.read_adsb import ADSBSource
from flights import cli
from flights.flight_source import ADSBFlightSource, NonADSBFlightSource


@dataclass(frozen=True)
class DummySource(NonADSBFlightSource):
    name: str = "bts"

    def create_flights_for_day(self, dt, airport_catalog):
        raise AssertionError("write_flights should be mocked")


@dataclass(frozen=True)
class DummyADSBSource(ADSBFlightSource):
    name: str = "algorithm-nearest"

    def create_flights_for_day(self, dt, df_adsb, airport_catalog):
        raise AssertionError("write_flights should be mocked")


def empty_flights():
    return pl.DataFrame({"icao": []}, schema={"icao": pl.String})


def test_write_flights_range_excludes_end_date(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        cli,
        "write_flights",
        lambda source, dt, adsb_source: calls.append((source, dt, adsb_source))
        or empty_flights(),
    )
    monkeypatch.setattr(cli, "flight_path", lambda *args: tmp_path / "data.parquet")

    cli.write_flights_range(
        date(2026, 3, 1),
        date(2026, 3, 3),
        DummySource(),
    )

    assert [call[1] for call in calls] == [
        date(2026, 3, 1),
        date(2026, 3, 2),
    ]
    assert all(call[2] is None for call in calls)


def test_write_flights_range_passes_adsb_source(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        cli,
        "write_flights",
        lambda source, dt, adsb_source: calls.append(adsb_source)
        or empty_flights(),
    )
    monkeypatch.setattr(cli, "flight_path", lambda *args: tmp_path / "data.parquet")

    cli.write_flights_range(
        date(2026, 3, 1),
        date(2026, 3, 2),
        DummyADSBSource(),
        adsb_source=ADSBSource.ADSBX,
    )

    assert calls == [ADSBSource.ADSBX]


@pytest.mark.parametrize(
    ("start_date", "end_date"),
    [
        (date(2026, 4, 1), date(2026, 4, 1)),
        (date(2026, 4, 2), date(2026, 4, 1)),
    ],
)
def test_date_range_requires_end_after_start(start_date, end_date):
    with pytest.raises(AssertionError):
        list(cli.date_range(start_date, end_date))


def test_main_passes_flight_source_object(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cli,
        "write_flights_range",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    cli.main(["2026-03-01", "2026-04-01", "bts"])

    args, kwargs = calls[0]
    assert args[:2] == (date(2026, 3, 1), date(2026, 4, 1))
    assert args[2] is cli.FLIGHT_SOURCES_BY_NAME["bts"]
    assert kwargs == {"adsb_source": ADSBSource.ADSBLOL}


@pytest.mark.parametrize("source_name", ["algorithm-nearest", "algorithm-nn"])
def test_algorithm_sources_accept_opensky(monkeypatch, source_name):
    calls = []
    monkeypatch.setattr(
        cli,
        "write_flights_range",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    cli.main(
        [
            "2026-03-01",
            "2026-03-02",
            source_name,
            "--adsb-source",
            "opensky",
        ]
    )

    args, kwargs = calls[0]
    assert args[2] is cli.FLIGHT_SOURCES_BY_NAME[source_name]
    assert kwargs == {"adsb_source": ADSBSource.OPENSKY}


def test_source_names_come_from_flight_source_objects():
    assert list(cli.FLIGHT_SOURCES_BY_NAME) == [
        source.name for source in cli.FLIGHT_SOURCES
    ]
