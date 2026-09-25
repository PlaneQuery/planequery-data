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


@pytest.mark.parametrize("source_name", ["algorithm-nearest", "algorithm-nn"])
def test_algorithm_sources_default_workers_from_machine_resources(
    monkeypatch,
    source_name,
):
    monkeypatch.setattr(cli, "default_worker_count", lambda: 6)

    assert (
        cli.default_workers_for_source(cli.FLIGHT_SOURCES_BY_NAME[source_name]) == 6
    )


def test_non_algorithm_source_defaults_to_one_worker(monkeypatch):
    monkeypatch.setattr(
        cli,
        "default_worker_count",
        lambda: (_ for _ in ()).throw(AssertionError("must not be called")),
    )

    assert cli.default_workers_for_source(DummySource()) == 1


def test_range_worker_reuses_initialized_source(monkeypatch, tmp_path):
    source = DummySource()
    calls = []
    monkeypatch.setattr(
        cli,
        "write_flights",
        lambda initialized_source, dt, adsb_source: calls.append(
            (initialized_source, dt, adsb_source)
        )
        or empty_flights(),
    )
    monkeypatch.setattr(cli, "flight_path", lambda *args: tmp_path / "data.parquet")
    cli._initialize_range_worker(source, None)

    result = cli._write_flights_day_in_worker(date(2026, 3, 1))

    assert calls == [(source, date(2026, 3, 1), None)]
    assert result == (date(2026, 3, 1), 0, tmp_path / "data.parquet")


def test_algorithm_range_uses_configured_parallel_workers(monkeypatch, tmp_path):
    calls = []
    pool_settings = []

    class InlinePool:
        def __init__(self, *, max_workers, mp_context, initializer, initargs):
            pool_settings.append((max_workers, mp_context.get_start_method()))
            initializer(*initargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def map(self, function, items):
            return map(function, items)

    monkeypatch.setattr(cli, "ProcessPoolExecutor", InlinePool)
    monkeypatch.setattr(
        cli,
        "write_flights",
        lambda source, dt, adsb_source: calls.append(dt) or empty_flights(),
    )
    monkeypatch.setattr(cli, "flight_path", lambda *args: tmp_path / "data.parquet")

    cli.write_flights_range(
        date(2026, 3, 1),
        date(2026, 3, 3),
        cli.FLIGHT_SOURCES_BY_NAME["algorithm-nearest"],
        workers=2,
    )

    assert pool_settings == [(2, "spawn")]
    assert calls == [date(2026, 3, 1), date(2026, 3, 2)]


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


def test_main_passes_explicit_worker_override(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cli,
        "write_flights_range",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    cli.main(
        [
            "2026-03-01",
            "2026-04-01",
            "algorithm-nn",
            "--workers",
            "3",
        ]
    )

    assert calls[0][1] == {
        "adsb_source": ADSBSource.ADSBLOL,
        "workers": 3,
    }


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
