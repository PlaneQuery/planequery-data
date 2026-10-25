from dataclasses import dataclass, replace
from datetime import date

import polars as pl
import pytest

from data_engineering.adsb.read_adsb import ADSBSource
from flights import cli
from flights.flight_source import ADSBFlightSource, NonADSBFlightSource


BATCHED_ALGORITHM_SOURCES = (
    "algorithm-nearest", "algorithm-nn", "sun", "openap", "plane-notify",
)


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


@pytest.mark.parametrize("provider_args, expected", [
    ([], ADSBSource.ADSBLOL),
    (["--adsb-source", "opensky"], ADSBSource.OPENSKY),
])
def test_eurocontrol_always_enriches_callsigns(monkeypatch, provider_args, expected):
    calls = []
    monkeypatch.setattr(cli, "write_flights_range", lambda *args, **kwargs: calls.append((args, kwargs)))
    cli.main([
        "2024-06-01", "2024-06-02", "eurocontrol",
        *provider_args,
    ])
    source = calls[0][0][2]
    assert source.callsign_source is expected
    assert source.enrichment_adsb_source() is expected


def test_eurocontrol_callsign_flag_is_removed():
    with pytest.raises(SystemExit):
        cli.parse_args(["2024-06-01", "2024-06-02", "eurocontrol", "--eurocontrol-callsigns"])


@pytest.mark.parametrize("provider_args", [
    [], ["--adsb-source", "opensky"], ["--adsb-source", "adsbx"],
])
def test_eurocontrol_verified_cli_always_uses_adsblol(monkeypatch, provider_args):
    calls = []
    monkeypatch.setattr(cli, "write_flights_range", lambda *args, **kwargs: calls.append(args[2]))
    cli.main(["2024-06-01", "2024-06-02", "eurocontrol-verified", *provider_args])
    assert calls[0].name == "eurocontrol-verified"
    assert calls[0].source.callsign_source is ADSBSource.ADSBLOL
    assert calls[0].enrichment_adsb_source() is ADSBSource.ADSBLOL


@pytest.mark.parametrize("source_name", ["bts-verified", "sfdps-verified"])
@pytest.mark.parametrize("provider", list(ADSBSource))
def test_verified_cli_always_uses_adsblol(monkeypatch, source_name, provider):
    calls = []
    monkeypatch.setattr(
        cli,
        "write_flights_range",
        lambda *args, **kwargs: calls.append(args[2]),
    )
    cli.main([
        "2024-06-01", "2024-06-02", source_name,
        "--adsb-source", provider.value,
    ])
    assert calls[0].name == source_name
    assert calls[0].adsb_source is ADSBSource.ADSBLOL
    assert calls[0].enrichment_adsb_source() is ADSBSource.ADSBLOL


@pytest.mark.parametrize("source_name", BATCHED_ALGORITHM_SOURCES)
@pytest.mark.parametrize("workers", [1, 4])
def test_range_skips_flights_requiring_bad_adsblol_day(
    monkeypatch, tmp_path, capsys, source_name, workers
):
    calls = []
    monkeypatch.setattr(
        cli,
        "_write_flights_day",
        lambda source, dt, adsb_source, **kwargs: (
            calls.append(dt) or dt, 0, tmp_path / "data.parquet"
        ),
    )

    cli.write_flights_range(
        date(2026, 4, 29), date(2026, 5, 3),
        cli.FLIGHT_SOURCES_BY_NAME[source_name], workers=workers,
    )

    assert calls == [date(2026, 5, 2)]
    output = capsys.readouterr().out
    assert output.count("[SKIP]") == 3
    assert "requires known bad adsblol day(s): 2026-04-30" in output


def test_range_with_only_bad_days_does_not_start_pool(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Skipped range must not start workers or write flights")

    monkeypatch.setattr(cli, "ProcessPoolExecutor", unexpected)
    monkeypatch.setattr(cli, "_write_flights_day", unexpected)
    cli.write_flights_range(
        date(2026, 4, 29), date(2026, 5, 2),
        cli.ALGORITHM_NN_SOURCE, workers=4,
    )


@pytest.mark.parametrize("source_name", BATCHED_ALGORITHM_SOURCES)
def test_target_day_range_skips_only_bad_target_days(
    monkeypatch, tmp_path, capsys, source_name
):
    calls = []
    monkeypatch.setattr(
        cli,
        "_write_flights_day",
        lambda source, dt, adsb_source, **kwargs: (
            calls.append(dt) or dt, 0, tmp_path / "data.parquet"
        ),
    )
    source = replace(cli.FLIGHT_SOURCES_BY_NAME[source_name], target_day_adsb=True)

    cli.write_flights_range(
        date(2026, 4, 29), date(2026, 5, 3), source, workers=1,
    )

    assert calls == [date(2026, 4, 29), date(2026, 5, 1), date(2026, 5, 2)]
    output = capsys.readouterr().out
    assert output.count("[SKIP]") == 1
    assert "requires known bad adsblol day(s): 2026-04-30" in output


@pytest.mark.parametrize(
    "source, provider",
    [(DummySource(), ADSBSource.ADSBLOL), (DummyADSBSource(), ADSBSource.OPENSKY)],
)
def test_bad_adsblol_dates_do_not_skip_other_sources(monkeypatch, tmp_path, source, provider):
    calls = []
    monkeypatch.setattr(
        cli, "write_flights",
        lambda source, dt, adsb_source: calls.append(dt) or empty_flights(),
    )
    monkeypatch.setattr(cli, "flight_path", lambda *args: tmp_path / "data.parquet")
    cli.write_flights_range(
        date(2026, 4, 30), date(2026, 5, 1), source, provider, workers=1,
    )
    assert calls == [date(2026, 4, 30)]


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


@pytest.mark.parametrize("source_name", BATCHED_ALGORITHM_SOURCES)
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


@pytest.mark.parametrize("source_name", BATCHED_ALGORITHM_SOURCES)
def test_adsblol_algorithm_dates_include_adjacent_days(source_name):
    source = cli.FLIGHT_SOURCES_BY_NAME[source_name]

    assert cli.adsb_dates_for_run(
        source,
        [date(2026, 1, 1)],
        ADSBSource.ADSBLOL,
    ) == [
        date(2025, 12, 31),
        date(2026, 1, 1),
        date(2026, 1, 2),
    ]


@pytest.mark.parametrize("source_name", BATCHED_ALGORITHM_SOURCES)
@pytest.mark.parametrize(
    "adsb_source, expected",
    [
        (ADSBSource.ADSBX, [date(2026, 1, 1)]),
        (ADSBSource.OPENSKY, [date(2025, 12, 31), date(2026, 1, 1), date(2026, 1, 2)]),
    ],
)
def test_non_adsblol_algorithm_dates_use_provider_window(
    source_name,
    adsb_source,
    expected,
):
    source = cli.FLIGHT_SOURCES_BY_NAME[source_name]

    assert cli.adsb_dates_for_run(
        source,
        [date(2026, 1, 1)],
        adsb_source,
    ) == expected


@pytest.mark.parametrize("source_name", BATCHED_ALGORITHM_SOURCES)
@pytest.mark.parametrize("adsb_source", list(ADSBSource))
def test_target_day_dates_exclude_adjacent_days(source_name, adsb_source):
    source = replace(cli.FLIGHT_SOURCES_BY_NAME[source_name], target_day_adsb=True)
    target_dates = [date(2026, 1, 1), date(2026, 1, 2)]

    assert cli.adsb_dates_for_run(source, target_dates, adsb_source) == target_dates


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


@pytest.mark.parametrize("source_name", BATCHED_ALGORITHM_SOURCES)
def test_algorithm_range_uses_configured_parallel_workers(monkeypatch, tmp_path, source_name):
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
        cli.FLIGHT_SOURCES_BY_NAME[source_name],
        workers=2,
    )

    assert pool_settings == [(2, "spawn")]
    assert calls == [date(2026, 3, 1), date(2026, 3, 2)]


@pytest.mark.parametrize("source_name", BATCHED_ALGORITHM_SOURCES)
def test_single_algorithm_day_uses_workers_for_adsb_batches(
    monkeypatch,
    tmp_path,
    source_name,
):
    calls = []

    def write(source, dt, adsb_source, *, batch_workers):
        calls.append((source, dt, adsb_source, batch_workers))
        return empty_flights()

    monkeypatch.setattr(cli, "write_flights", write)
    monkeypatch.setattr(cli, "flight_path", lambda *args: tmp_path / "data.parquet")
    monkeypatch.setattr(
        cli,
        "ProcessPoolExecutor",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("one day must not create the day-level pool")
        ),
    )
    source = cli.FLIGHT_SOURCES_BY_NAME[source_name]

    cli.write_flights_range(
        date(2026, 6, 1),
        date(2026, 6, 2),
        source,
        workers=4,
    )

    assert calls == [
        (
            source,
            date(2026, 6, 1),
            ADSBSource.ADSBLOL,
            4,
        )
    ]


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
    from data_engineering.adsblol import check_for_new_release

    calls = []
    prepared_years = []
    monkeypatch.setattr(
        cli,
        "write_flights_range",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        check_for_new_release,
        "prepare_releases_for_run",
        lambda years: prepared_years.append(set(years)),
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
    assert prepared_years == [{2026}]


@pytest.mark.parametrize("source_name", BATCHED_ALGORITHM_SOURCES)
@pytest.mark.parametrize("target_day_adsb", [False, True])
def test_main_configures_target_day_adsb_before_planning(
    monkeypatch, tmp_path, source_name, target_day_adsb
):
    from data_engineering.adsblol import check_for_new_release

    sources = []
    prepared_years = []
    monkeypatch.setattr(
        cli, "write_flights_range",
        lambda start, end, source, **kwargs: sources.append(source),
    )
    monkeypatch.setattr(
        check_for_new_release, "prepare_releases_for_run",
        lambda years: prepared_years.append(set(years)),
    )
    args = ["2026-01-01", "2026-01-02", source_name]
    model_path = tmp_path / "model.pt"
    if source_name == "algorithm-nn":
        args.extend(["--airport-nn-model", str(model_path)])
    if target_day_adsb:
        args.append("--target-day-adsb")

    cli.main(args)

    assert len(sources) == 1
    assert sources[0].target_day_adsb is target_day_adsb
    assert prepared_years == [{2026} if target_day_adsb else {2025, 2026}]
    assert cli.FLIGHT_SOURCES_BY_NAME[source_name].target_day_adsb is False
    if source_name == "algorithm-nn":
        assert sources[0].model_path == model_path


@pytest.mark.parametrize("source_name", ["bts", "eurocontrol-verified"])
def test_target_day_adsb_rejects_non_adsb_source(source_name):
    with pytest.raises(SystemExit, match="--target-day-adsb requires an ADS-B flight source"):
        cli.main([
            "2026-03-01", "2026-03-02", source_name, "--target-day-adsb",
        ])


def test_no_save_adsb_downloads_removes_only_run_created_artifacts(
    monkeypatch,
    tmp_path,
):
    created_path = tmp_path / "created" / "day=01"
    existing_path = tmp_path / "existing" / "day=01"
    existing_path.mkdir(parents=True)
    (existing_path / "data.parquet").touch()

    monkeypatch.setattr(
        cli,
        "_downloaded_adsb_paths",
        lambda source, target_dates: (created_path, existing_path),
    )
    monkeypatch.setattr(
        cli,
        "write_flights_range",
        lambda *args, **kwargs: (
            created_path.mkdir(parents=True),
            (created_path / "data.parquet").touch(),
        ),
    )
    from data_engineering.adsblol import check_for_new_release

    monkeypatch.setattr(
        check_for_new_release,
        "prepare_releases_for_run",
        lambda years: None,
    )

    cli.main(
        [
            "2026-03-01",
            "2026-03-02",
            "algorithm-nearest",
            "--no-save-adsb-downloads",
        ]
    )

    assert not created_path.exists()
    assert (existing_path / "data.parquet").exists()


def test_no_save_adsb_downloads_rejects_non_adsb_source():
    with pytest.raises(SystemExit, match="requires an ADS-B flight source"):
        cli.main(
            [
                "2026-03-01",
                "2026-03-02",
                "bts",
                "--no-save-adsb-downloads",
            ]
        )


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("target_day_adsb", [False, True])
def test_no_save_cleans_between_batches_and_preserves_shared_inputs(
    monkeypatch, tmp_path, workers, fail, target_day_adsb
):
    from data_engineering.adsb import canonical, download_adsb_to_parquet
    from data_engineering.adsblol import check_for_new_release

    monkeypatch.setattr(canonical, "ADSB_PARQUET_ROOT", tmp_path / "processed")
    monkeypatch.setattr(download_adsb_to_parquet, "EXTRACT_DIR", tmp_path / "extract")
    monkeypatch.setattr(download_adsb_to_parquet, "RELEASES_DIR", tmp_path / "releases")
    monkeypatch.setattr(check_for_new_release, "prepare_releases_for_run", lambda years: None)
    provider = ADSBSource.ADSBLOL
    existing = cli._downloaded_adsb_paths(provider, [date(2026, 2, 28)])[0]
    existing.mkdir(parents=True)
    (existing / "data.parquet").touch()
    output = tmp_path / "flights.parquet"
    calls = []
    previous_inputs = set()
    created = set()

    def write_batch(start, end, source, **kwargs):
        nonlocal previous_inputs
        days = list(cli.date_range(start, end))
        assert len(days) <= workers
        inputs = set(cli.adsb_dates_for_run(source, days, provider))
        if target_day_adsb:
            assert inputs == set(days)
        if calls:
            # Overlapping partitions survive; expired partitions are already gone.
            for day in previous_inputs:
                for path in cli._downloaded_adsb_paths(provider, [day]):
                    assert path.exists() == (day in inputs or path == existing)
        calls.append(days)
        for path in cli._downloaded_adsb_paths(provider, sorted(inputs)):
            path.mkdir(parents=True, exist_ok=True)
            (path / "data.parquet").touch()
            created.add(path)
        output.touch()
        previous_inputs = inputs
        if fail and len(calls) == 2:
            raise RuntimeError("conversion failed")

    monkeypatch.setattr(cli, "write_flights_range", write_batch)
    args = [
        "2026-03-01", "2026-03-06", "algorithm-nn",
        "--workers", str(workers), "--no-save-adsb-downloads",
    ]
    if target_day_adsb:
        args.append("--target-day-adsb")
    if fail:
        with pytest.raises(RuntimeError, match="conversion failed"):
            cli.main(args)
    else:
        cli.main(args)
        assert sum(map(len, calls)) == 5
    assert len(calls) >= 2
    assert (existing / "data.parquet").exists()
    assert output.exists()
    assert all(not path.exists() for path in created if path != existing)


@pytest.mark.parametrize("source_name", BATCHED_ALGORITHM_SOURCES)
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


def test_opensky_nn_cli_uses_checkpoint_and_fixed_provider(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        cli, "write_flights",
        lambda *args: calls.append(args) or empty_flights(),
    )
    monkeypatch.setattr(cli, "flight_path", lambda *args: tmp_path / "data.parquet")
    checkpoint = tmp_path / "model.pt"

    cli.main([
        "2026-03-01", "2026-03-02", "opensky-nn",
        "--airport-nn-model", str(checkpoint),
    ])

    assert len(calls) == 1
    source, day, adsb_source = calls[0]
    assert isinstance(source, cli.OpenSkyNNFlightSource)
    assert source.model_path == checkpoint
    assert source.enrichment_adsb_source() is ADSBSource.OPENSKY
    assert day == date(2026, 3, 1)
    assert adsb_source is None
