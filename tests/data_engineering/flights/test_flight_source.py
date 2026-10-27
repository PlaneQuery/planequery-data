from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from airports.models import AirportCatalogProvenance
from data_engineering.adsb import canonical as adsb_canonical
from data_engineering.adsb.read_adsb import ADSBSource
from flights import flight_source as flight_source_module
from flights.algorithm import AlgorithmNearestFlightSource
from flights.airport_nn import inference, model
from flights.airport_nn.features import COLUMNS as AIRPORT_NN_COLUMNS
from flights.airport_nn.source import AlgorithmNNFlightSource
from flights.flight_source import (
    ADSBFlightSource,
    NonADSBFlightSource,
    get_flights,
    get_flights_range,
    flight_path,
    write_flights,
)
from flights.providers.bts import BTSFlightSource
from flights.flight_manifest import (
    FlightManifest,
    flight_manifest_path,
)


@dataclass(frozen=True)
class FakeAirportCatalog:
    provenance: AirportCatalogProvenance


def fake_airport_catalog(target_date: date) -> FakeAirportCatalog:
    return FakeAirportCatalog(
        AirportCatalogProvenance(
            flight_date=target_date,
            observed_commit_sha="observed-commit",
            knowledge_cutoff_date=date(2026, 4, 1),
            knowledge_commit_sha="knowledge-commit",
            catalog_available_date=date(2026, 4, 1),
            policy_version="test",
        )
    )


@pytest.fixture(autouse=True)
def airport_catalog(monkeypatch):
    monkeypatch.setattr(
        flight_source_module,
        "airport_catalog_for_date",
        fake_airport_catalog,
    )


@dataclass(frozen=True)
class DummyADSBFlightSource(ADSBFlightSource):
    name: str = "dummy-adsb"

    def adsb_read_interval(self, dt):
        return datetime(2026, 3, 1, 1), datetime(2026, 3, 2, 2)

    def create_flights_for_day(self, dt, df_adsb, airport_catalog):
        return df_adsb.select("icao")


@dataclass(frozen=True)
class DummyBatchedADSBFlightSource(DummyADSBFlightSource):
    def adsb_icao_batch_size(self):
        return 2


@dataclass(frozen=True)
class DummySelectedADSBFlightSource(DummyADSBFlightSource):
    def adsb_icaos(self, dt, airport_catalog):
        return ["abc123", "def456"]


@dataclass(frozen=True)
class DummyNonADSBFlightSource(NonADSBFlightSource):
    name: str = "dummy-non-adsb"

    def create_flights_for_day(self, dt, airport_catalog):
        return pl.DataFrame({"icao": ["abc123"]})


def test_create_flights_writes_single_bts_day_to_canonical_path(
    monkeypatch,
    tmp_path,
):
    target_date = date(2026, 3, 1)
    expected = pl.DataFrame({"icao": ["abc123"]})
    calls = []

    monkeypatch.setattr(flight_source_module, "OUTPUT_DIR", tmp_path)
    airport_catalog = fake_airport_catalog(target_date)
    monkeypatch.setattr(
        flight_source_module,
        "airport_catalog_for_date",
        lambda dt: airport_catalog,
    )

    def create_day(self, dt, catalog):
        calls.append((dt, catalog))
        return expected

    monkeypatch.setattr(BTSFlightSource, "create_flights_for_day", create_day)

    result = write_flights(BTSFlightSource(), target_date)
    path = flight_path(BTSFlightSource(), target_date)

    assert calls == [(target_date, airport_catalog)]
    assert result.equals(expected)
    assert pl.read_parquet(path).equals(expected)
    assert path == (
        tmp_path
        / f"data/flights/source=bts/version={BTSFlightSource().version}"
        / "year=2026/month=03/day=01/data.parquet"
    )
    assert FlightManifest.read(
        flight_manifest_path(path)
    ) == FlightManifest.from_provenance(airport_catalog.provenance)


def test_algorithm_nearest_path_includes_adsb_source(monkeypatch, tmp_path):
    monkeypatch.setattr(flight_source_module, "OUTPUT_DIR", tmp_path)

    path = flight_path(
        AlgorithmNearestFlightSource(),
        date(2026, 3, 1),
        ADSBSource.ADSBLOL,
    )

    assert path == Path(
        tmp_path,
        "data/flights/source=algorithm-nearest/adsb_source=adsblol/"
        f"version={AlgorithmNearestFlightSource().version}",
        "year=2026/month=03/day=01/data.parquet",
    )


@pytest.mark.parametrize("adsb_source", tuple(ADSBSource))
@pytest.mark.parametrize("reuse_single_day", [False, True])
def test_single_day_nn_inputs_and_caches(
    monkeypatch, tmp_path, adsb_source, reuse_single_day,
):
    target_date = date(2026, 3, 1)
    start = datetime(2026, 3, 1)
    end = datetime(2026, 3, 2)
    catalog = fake_airport_catalog(target_date)
    monkeypatch.setattr(flight_source_module, "OUTPUT_DIR", tmp_path)
    source = AlgorithmNNFlightSource(
        model_path=tmp_path / "model.pt", target_day_adsb=True,
    )

    def cache_flights(cached_source, label):
        path = flight_path(cached_source, target_date, adsb_source)
        path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"icao": ["abc123"], "label": [label]}).write_parquet(path)
        cached_source.flight_manifest(catalog.provenance, adsb_source).write(
            flight_manifest_path(path)
        )

    # Neither ordinary NN flights nor ordinary nearest segments may be reused.
    cache_flights(AlgorithmNNFlightSource(), "wrong-nn-context")
    cache_flights(AlgorithmNearestFlightSource(), "wrong-nearest-context")
    if reuse_single_day:
        cache_flights(AlgorithmNearestFlightSource(target_day_adsb=True), "reused")

    messages = pl.DataFrame({
        "icao": ["abc123"] * 4,
        "time": [
            start - timedelta(microseconds=1), start,
            end - timedelta(microseconds=1), end,
        ],
    })
    reads = []

    def read_adsb(read_start, read_end, *, source, lazy=False, **kwargs):
        assert (read_start, read_end) == (start, end)
        assert source == adsb_source
        reads.append(lazy)
        frame = messages.filter(
            (pl.col("time") >= read_start) & (pl.col("time") < read_end)
        )
        return frame.lazy() if lazy else frame

    expected_times = [start, end - timedelta(microseconds=1)]
    segmented = []

    def segment(self, dt, frame, airport_catalog):
        assert not reuse_single_day
        assert frame["time"].to_list() == expected_times
        segmented.append(dt)
        return pl.DataFrame({"icao": ["abc123"], "label": ["segmented"]})

    predictions = []

    def predict(flights, frame, loaded_model, airport_catalog):
        assert frame["time"].to_list() == expected_times
        predictions.append(flights)
        return flights

    monkeypatch.setattr(flight_source_module, "read_adsb", read_adsb)
    monkeypatch.setattr(AlgorithmNearestFlightSource, "create_flights_for_day", segment)
    monkeypatch.setattr(model, "load_model", lambda path: "model")
    monkeypatch.setattr(inference, "predict_flights", predict)

    result = get_flights(source, target_date, adsb_source)
    assert result["label"].to_list() == ["reused" if reuse_single_day else "segmented"]
    assert segmented == ([] if reuse_single_day else [target_date])
    assert reads == ([False] if reuse_single_day else [True, False])
    assert len(predictions) == 1

    path = flight_path(source, target_date, adsb_source)
    manifest = FlightManifest.read(flight_manifest_path(path))
    assert manifest.target_day_adsb is True
    assert manifest.flight_cache_key != AlgorithmNNFlightSource().flight_manifest(
        catalog.provenance, adsb_source,
    ).flight_cache_key
    assert get_flights(source, target_date, adsb_source).equals(result)
    assert len(predictions) == 1


def test_target_day_adsb_reaches_unbatched_reads(monkeypatch, tmp_path):
    monkeypatch.setattr(flight_source_module, "OUTPUT_DIR", tmp_path)

    def read_adsb(start, end, **kwargs):
        assert start == datetime(2026, 3, 1)
        assert end == datetime(2026, 3, 2)
        return pl.DataFrame({"icao": ["abc123"]})

    monkeypatch.setattr(flight_source_module, "read_adsb", read_adsb)
    get_flights(
        DummyADSBFlightSource(target_day_adsb=True),
        date(2026, 3, 1), ADSBSource.ADSBX, use_cache=False,
    )


@pytest.mark.parametrize(
    ("adsb_source", "expected"),
    [
        (
            ADSBSource.ADSBLOL,
            (datetime(2026, 2, 28), datetime(2026, 3, 3)),
        ),
        (
            ADSBSource.ADSBX,
            (datetime(2026, 3, 1), datetime(2026, 3, 2)),
        ),
        (
            ADSBSource.OPENSKY,
            (datetime(2026, 2, 28), datetime(2026, 3, 3)),
        ),
    ],
)
def test_algorithm_sources_use_provider_specific_read_intervals(
    tmp_path,
    adsb_source,
    expected,
):
    target_date = date(2026, 3, 1)

    assert (
        AlgorithmNearestFlightSource().adsb_read_interval_for_source(
            target_date,
            adsb_source,
        )
        == expected
    )
    assert (
        AlgorithmNNFlightSource(
            model_path=tmp_path / "model.pt"
        ).adsb_read_interval_for_source(target_date, adsb_source)
        == expected
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


def test_get_flights_range_gets_each_day_and_excludes_end_date(monkeypatch):
    source = DummyADSBFlightSource()
    calls = []

    def get_day(source_arg, target_date, adsb_src, use_cache, batch_workers):
        calls.append(
            (source_arg, target_date, adsb_src, use_cache, batch_workers)
        )
        return pl.DataFrame({"date": [target_date]})

    monkeypatch.setattr(flight_source_module, "get_flights", get_day)

    result = get_flights_range(
        source,
        date(2026, 3, 1),
        date(2026, 3, 3),
        ADSBSource.ADSBX,
        use_cache=False,
        batch_workers=4,
    )

    assert result.get_column("date").to_list() == [
        date(2026, 3, 1),
        date(2026, 3, 2),
    ]
    assert calls == [
        (source, date(2026, 3, 1), ADSBSource.ADSBX, False, 4),
        (source, date(2026, 3, 2), ADSBSource.ADSBX, False, 4),
    ]


def test_get_flights_reads_bounded_complete_aircraft_batches(monkeypatch, tmp_path):
    target_date = date(2026, 3, 1)
    batches = []
    monkeypatch.setattr(flight_source_module, "OUTPUT_DIR", tmp_path)

    def read_adsb(start, end, *, source, columns=None, lazy=False, icaos=None):
        if lazy:
            assert columns == ["icao"]
            assert icaos is None
            return pl.DataFrame(
                {"icao": ["eee555", "aaa111", None, "ccc333", "bbb222", "ddd444"]}
            ).lazy()
        assert icaos is not None
        assert 0 < len(icaos) <= 2
        batches.append(icaos)
        return pl.DataFrame({"icao": icaos})

    monkeypatch.setattr(flight_source_module, "read_adsb", read_adsb)

    result = get_flights(
        DummyBatchedADSBFlightSource(),
        target_date,
        ADSBSource.ADSBX,
        use_cache=False,
    )

    assert batches == [
        ["aaa111", "bbb222"],
        ["ccc333", "ddd444"],
        ["eee555"],
    ]
    assert result.get_column("icao").to_list() == [
        "aaa111",
        "bbb222",
        "ccc333",
        "ddd444",
        "eee555",
    ]


def test_get_flights_can_process_aircraft_batches_in_parallel(
    monkeypatch,
    tmp_path,
):
    target_date = date(2026, 3, 1)
    batches = []
    pool_settings = []
    monkeypatch.setattr(flight_source_module, "OUTPUT_DIR", tmp_path)

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

    def read_adsb(start, end, *, source, columns=None, lazy=False, icaos=None):
        if lazy:
            return pl.DataFrame(
                {"icao": ["eee555", "aaa111", "ccc333", "bbb222", "ddd444"]}
            ).lazy()
        batches.append(icaos)
        return pl.DataFrame({"icao": icaos})

    monkeypatch.setattr(flight_source_module, "ProcessPoolExecutor", InlinePool)
    monkeypatch.setattr(flight_source_module, "read_adsb", read_adsb)

    result = get_flights(
        DummyBatchedADSBFlightSource(),
        target_date,
        ADSBSource.ADSBX,
        use_cache=False,
        batch_workers=2,
    )

    assert pool_settings == [(2, "spawn")]
    assert batches == [
        ["aaa111", "bbb222"],
        ["ccc333", "ddd444"],
        ["eee555"],
    ]
    assert result.get_column("icao").to_list() == [
        "aaa111",
        "bbb222",
        "ccc333",
        "ddd444",
        "eee555",
    ]


def test_get_flights_reads_only_source_icaos(monkeypatch, tmp_path):
    monkeypatch.setattr(flight_source_module, "OUTPUT_DIR", tmp_path)
    calls = []

    def read_adsb(start, end, *, source, icaos):
        calls.append(icaos)
        return pl.DataFrame({"icao": icaos})

    monkeypatch.setattr(flight_source_module, "read_adsb", read_adsb)

    result = get_flights(
        DummySelectedADSBFlightSource(),
        date(2026, 3, 1),
        ADSBSource.ADSBX,
        use_cache=False,
    )

    assert calls == [["abc123", "def456"]]
    assert result.get_column("icao").to_list() == ["abc123", "def456"]


@pytest.mark.parametrize("adsb_source", tuple(ADSBSource))
def test_algorithm_nn_reuses_cached_algorithm_nearest_flights_from_same_adsb_source(
    monkeypatch,
    tmp_path,
    adsb_source,
):
    target_date = date(2026, 3, 1)
    airport_catalog = fake_airport_catalog(target_date)
    nearest_source = AlgorithmNearestFlightSource()
    nn_source = AlgorithmNNFlightSource(model_path=tmp_path / "model.pt")
    nearest_flights = pl.DataFrame(
        {
            "icao": ["def456", "abc123"],
            "takeoff_airport_ident": ["OLD2", "OLD1"],
        }
    )
    monkeypatch.setattr(flight_source_module, "OUTPUT_DIR", tmp_path)
    for cached_adsb_source in ADSBSource:
        cached_flights = (
            nearest_flights
            if cached_adsb_source is adsb_source
            else pl.DataFrame(
                {
                    "icao": [f"wrong-{cached_adsb_source.value}"],
                    "takeoff_airport_ident": ["WRONG"],
                }
            )
        )
        nearest_path = flight_path(
            nearest_source,
            target_date,
            cached_adsb_source,
        )
        nearest_path.parent.mkdir(parents=True)
        cached_flights.write_parquet(nearest_path)
        FlightManifest.from_provenance(
            airport_catalog.provenance,
            adsb_source=cached_adsb_source,
        ).write(flight_manifest_path(nearest_path))

    reads = []

    def read_adsb(start, end, *, source, icaos, columns):
        reads.append((source, icaos, columns))
        return pl.DataFrame({"icao": icaos})

    monkeypatch.setattr(flight_source_module, "read_adsb", read_adsb)
    monkeypatch.setattr(
        nearest_source.__class__,
        "create_flights_for_day",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("nearest segmentation must not run")
        ),
    )
    monkeypatch.setattr(model, "load_model", lambda path: "loaded-model")
    prediction_calls = []
    monkeypatch.setattr(
        inference,
        "predict_flights",
        lambda flights, messages, loaded_model, catalog: (
            prediction_calls.append((flights, messages, loaded_model, catalog))
            or flights
        ),
    )

    result = flight_source_module.get_flights_for_catalog(
        nn_source,
        target_date,
        airport_catalog,
        adsb_source,
        use_cache=False,
    )

    assert result.equals(nearest_flights)
    assert reads == [
        (
            adsb_source,
            ["abc123", "def456"],
            AIRPORT_NN_COLUMNS,
        )
    ]
    assert len(prediction_calls) == 1
    reused, messages, loaded_model, catalog = prediction_calls[0]
    assert reused.equals(nearest_flights)
    assert messages.get_column("icao").to_list() == ["abc123", "def456"]
    assert loaded_model == "loaded-model"
    assert catalog is airport_catalog


def test_algorithm_nn_ignores_nearest_flights_from_a_stale_catalog(
    monkeypatch,
    tmp_path,
):
    target_date = date(2026, 3, 1)
    airport_catalog = fake_airport_catalog(target_date)
    nearest_source = AlgorithmNearestFlightSource()
    nn_source = AlgorithmNNFlightSource(model_path=tmp_path / "model.pt")
    monkeypatch.setattr(flight_source_module, "OUTPUT_DIR", tmp_path)
    nearest_path = flight_path(
        nearest_source,
        target_date,
        ADSBSource.ADSBLOL,
    )
    nearest_path.parent.mkdir(parents=True)
    pl.DataFrame({"icao": ["stale1"]}).write_parquet(nearest_path)
    stale_catalog = FakeAirportCatalog(
        AirportCatalogProvenance(
            flight_date=target_date,
            observed_commit_sha="stale-observed-commit",
            knowledge_cutoff_date=date(2026, 4, 1),
            knowledge_commit_sha="stale-knowledge-commit",
            catalog_available_date=date(2026, 4, 1),
            policy_version="test",
        )
    )
    FlightManifest.from_provenance(
        stale_catalog.provenance,
        adsb_source=ADSBSource.ADSBLOL,
    ).write(flight_manifest_path(nearest_path))

    reads = []

    def read_adsb(start, end, *, source, columns=None, lazy=False, icaos=None):
        reads.append((lazy, icaos))
        frame = pl.DataFrame({"icao": ["abc123"]})
        return frame.lazy() if lazy else frame

    monkeypatch.setattr(flight_source_module, "read_adsb", read_adsb)
    expected = pl.DataFrame({"icao": ["fresh"]})
    monkeypatch.setattr(
        AlgorithmNNFlightSource,
        "create_flights_for_day",
        lambda self, dt, df, catalog: expected,
    )

    result = flight_source_module.get_flights_for_catalog(
        nn_source,
        target_date,
        airport_catalog,
        ADSBSource.ADSBLOL,
        use_cache=False,
    )

    assert result.equals(expected)
    assert reads == [(True, None), (False, ["abc123"])]


def test_get_flights_rejects_empty_batched_adsb_day(monkeypatch, tmp_path):
    monkeypatch.setattr(flight_source_module, "OUTPUT_DIR", tmp_path)

    def read_adsb(start, end, *, source, columns=None, lazy=False, icaos=None):
        frame = pl.DataFrame(schema={"icao": pl.String})
        assert lazy
        assert icaos is None
        return frame.lazy()

    monkeypatch.setattr(flight_source_module, "read_adsb", read_adsb)

    with pytest.raises(
        AssertionError,
        match=(
            "No ADS-B ICAOs found for dummy-adsb on 2026-03-01 from adsbx"
        ),
    ):
        get_flights(
            DummyBatchedADSBFlightSource(),
            date(2026, 3, 1),
            ADSBSource.ADSBX,
            use_cache=False,
        )


def test_get_flights_uses_cache_before_creating(monkeypatch, tmp_path):
    target_date = date(2026, 3, 1)
    source = DummyADSBFlightSource()
    monkeypatch.setattr(flight_source_module, "OUTPUT_DIR", tmp_path)
    path = flight_path(source, target_date, ADSBSource.ADSBX)
    path.parent.mkdir(parents=True)
    expected = pl.DataFrame({"icao": ["cached"]})
    expected.write_parquet(path)
    airport_catalog = fake_airport_catalog(target_date)
    monkeypatch.setattr(
        flight_source_module,
        "airport_catalog_for_date",
        lambda _dt: airport_catalog,
    )
    FlightManifest.from_provenance(
        airport_catalog.provenance,
        adsb_source=ADSBSource.ADSBX,
    ).write(flight_manifest_path(path))

    def fail_read(*args, **kwargs):
        raise AssertionError("cached flights should not read ADS-B")

    monkeypatch.setattr(flight_source_module, "read_adsb", fail_read)

    result = get_flights(source, target_date, ADSBSource.ADSBX)

    assert result.equals(expected)


def test_get_flights_rebuilds_cache_after_adsb_parquet_version_changes(
    monkeypatch,
    tmp_path,
):
    target_date = date(2026, 3, 1)
    source = DummyADSBFlightSource()
    adsb_source = ADSBSource.ADSBX
    airport_catalog = fake_airport_catalog(target_date)
    monkeypatch.setattr(flight_source_module, "OUTPUT_DIR", tmp_path)
    path = flight_path(source, target_date, adsb_source)
    path.parent.mkdir(parents=True)
    pl.DataFrame({"icao": ["cached"]}).write_parquet(path)
    FlightManifest.from_provenance(
        airport_catalog.provenance,
        adsb_source=adsb_source,
    ).write(flight_manifest_path(path))

    next_version = adsb_canonical.ADSB_PARQUET_VERSIONS[adsb_source] + 1
    monkeypatch.setitem(
        adsb_canonical.ADSB_PARQUET_VERSIONS,
        adsb_source,
        next_version,
    )
    reads = []

    def read_adsb(start, end, *, source):
        reads.append(source)
        return pl.DataFrame({"icao": ["fresh"]})

    monkeypatch.setattr(flight_source_module, "read_adsb", read_adsb)

    result = get_flights(source, target_date, adsb_source)

    assert result.get_column("icao").to_list() == ["fresh"]
    assert reads == [adsb_source]
    manifest = FlightManifest.read(flight_manifest_path(path))
    assert manifest.adsb_source == adsb_source.value
    assert manifest.adsb_parquet_version == next_version


@pytest.mark.parametrize("cache_state", ["missing-manifest", "stale-manifest"])
def test_get_flights_persists_rebuilt_invalid_cache(
    monkeypatch,
    tmp_path,
    cache_state,
):
    target_date = date(2026, 3, 1)
    source = DummyNonADSBFlightSource()
    current_catalog = fake_airport_catalog(target_date)
    monkeypatch.setattr(flight_source_module, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(
        flight_source_module,
        "airport_catalog_for_date",
        lambda _dt: current_catalog,
    )
    path = flight_path(source, target_date)
    path.parent.mkdir(parents=True)
    pl.DataFrame({"icao": ["stale"]}).write_parquet(path)
    if cache_state == "stale-manifest":
        stale_catalog = FakeAirportCatalog(
            AirportCatalogProvenance(
                flight_date=target_date,
                observed_commit_sha="stale-observed-commit",
                knowledge_cutoff_date=date(2026, 4, 1),
                knowledge_commit_sha="stale-knowledge-commit",
                catalog_available_date=date(2026, 4, 1),
                policy_version="test",
            )
        )
        FlightManifest.from_provenance(stale_catalog.provenance).write(
            flight_manifest_path(path)
        )

    create_calls = []

    def create_day(self, dt, catalog):
        create_calls.append((dt, catalog))
        return pl.DataFrame({"icao": ["fresh"]})

    monkeypatch.setattr(
        DummyNonADSBFlightSource,
        "create_flights_for_day",
        create_day,
    )

    result = get_flights(source, target_date)

    assert result.get_column("icao").to_list() == ["fresh"]
    assert pl.read_parquet(path).get_column("icao").to_list() == ["fresh"]
    assert FlightManifest.read(
        flight_manifest_path(path)
    ) == FlightManifest.from_provenance(current_catalog.provenance)
    assert create_calls == [(target_date, current_catalog)]

    create_calls.clear()
    cached_result = get_flights(source, target_date)

    assert cached_result.get_column("icao").to_list() == ["fresh"]
    assert create_calls == []


def test_get_flights_use_cache_false_does_not_persist(monkeypatch, tmp_path):
    target_date = date(2026, 3, 1)
    source = DummyNonADSBFlightSource()
    monkeypatch.setattr(flight_source_module, "OUTPUT_DIR", tmp_path)

    result = get_flights(source, target_date, use_cache=False)

    assert result.get_column("icao").to_list() == ["abc123"]
    assert not flight_path(source, target_date).exists()


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
