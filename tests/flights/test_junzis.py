import pickle
from dataclasses import replace
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from airports.matcher import AirportMatcher
from airports.models import (
    Airport,
    AirportCatalogProvenance,
    AirportMatchProvenance,
    AirportType,
)
from data_engineering.adsb.read_adsb import ADSBSource
from flights import cli, flight_source
from flights.evaluation import evaluation
from flights.flight_manifest import FlightManifest, flight_manifest_path
from flights.flight_type import FLIGHT_POLARS_SCHEMA
from flights.junzis import JunzisFlightSource
from flights.sources import FLIGHT_SOURCES_BY_NAME, JUNZIS_SOURCE


DAY = date(2026, 3, 1)
START = datetime(2026, 3, 1)


@pytest.fixture
def catalog():
    airports = [
        Airport(1, "FIRST", AirportType.SMALL_AIRPORT, "First", 0.0, 0.0),
        Airport(2, "LAST", AirportType.SMALL_AIRPORT, "Last", 0.0, 1.0),
    ]
    return SimpleNamespace(
        matcher=AirportMatcher(
            airports,
            {a.id: AirportMatchProvenance.KNOWN_ON_FLIGHT_DATE for a in airports},
            {a.id: "test" for a in airports},
        ),
        provenance=AirportCatalogProvenance(
            DAY, "observed", DAY, "knowledge", DAY, "test",
        ),
    )


def observations(seconds, icao="abc123"):
    seconds = list(seconds)
    return pl.DataFrame({
        "icao": [icao] * len(seconds),
        "time": [START + timedelta(seconds=float(t)) for t in seconds],
        "lat": [0.0] * len(seconds),
        "lon": np.linspace(0, 1, len(seconds)),
    })


def bounds(flights):
    return flights.select("takeoff_time", "landing_time").rows()


@pytest.mark.parametrize("size, expected", [(0, 0), (99, 0), (100, 0), (101, 1)])
def test_strict_upstream_observation_threshold(catalog, size, expected):
    result = JUNZIS_SOURCE.create_flights_for_day(DAY, observations(range(size)), catalog)
    assert result.schema == FLIGHT_POLARS_SCHEMA
    assert result.height == expected


def test_separated_clusters_and_noise(catalog):
    frame = observations([*range(101), *range(4000, 4101), 8000])
    result = JUNZIS_SOURCE.create_flights_for_day(DAY, frame, catalog)
    assert bounds(result) == [
        (START, START + timedelta(seconds=100)),
        (START + timedelta(seconds=4000), START + timedelta(seconds=4100)),
    ]


def test_sparse_chain_is_noise_despite_short_gaps(catalog):
    # More than 100 total rows and every gap below 30 minutes are insufficient:
    # no 30-minute neighborhood has the 100 points needed to be a DBSCAN core.
    result = JUNZIS_SOURCE.create_flights_for_day(
        DAY, observations(np.arange(200) * 300), catalog,
    )
    assert result.is_empty()


def test_cluster_size_is_strict_even_when_aircraft_has_enough_rows(catalog):
    result = JUNZIS_SOURCE.create_flights_for_day(
        DAY, observations([*range(100), *range(4000, 4101)]), catalog,
    )
    assert bounds(result) == [
        (START + timedelta(seconds=4000), START + timedelta(seconds=4100)),
    ]


def test_identical_timestamps_on_different_aircraft_are_preserved(catalog):
    first = observations(range(101), "aaa111")
    second = observations(range(101), "bbb222")
    for frames in ([first, second], [second, first]):
        result = JUNZIS_SOURCE.create_flights_for_day(
            DAY, pl.concat(frames), catalog,
        ).sort("icao")
        assert result["icao"].to_list() == ["aaa111", "bbb222"]
        assert bounds(result) == [(START, START + timedelta(seconds=100))] * 2


def test_duplicate_rows_do_not_satisfy_minimum_density(catalog):
    frame = observations(range(60))
    assert JUNZIS_SOURCE.create_flights_for_day(
        DAY, pl.concat([frame, frame]), catalog,
    ).is_empty()


def test_input_order_controls_ambiguous_border_point(catalog):
    # Each dense component has 101 points. The point at t=2100 is within
    # epsilon of one edge point of each component, but is not itself core.
    # DBSCAN assigns it to whichever component is visited first.
    left = observations([*range(100), 400])
    right = observations([3800, *range(4100, 4200)])
    border = observations([2100])
    earlier_first = JUNZIS_SOURCE.create_flights_for_day(
        DAY, pl.concat([left, right, border]), catalog,
    ).sort("takeoff_time")
    later_first = JUNZIS_SOURCE.create_flights_for_day(
        DAY, pl.concat([right, left, border]), catalog,
    ).sort("takeoff_time")
    assert bounds(earlier_first) == [
        (START, START + timedelta(seconds=2100)),
        (START + timedelta(seconds=3800), START + timedelta(seconds=4199)),
    ]
    assert bounds(later_first) == [
        (START, START + timedelta(seconds=400)),
        (START + timedelta(seconds=2100), START + timedelta(seconds=4199)),
    ]


def test_ground_observations_and_metadata_do_not_split_clusters(catalog):
    frame = observations(range(150)).with_columns(
        pl.lit(True).alias("on_ground"),
        pl.lit(0).alias("baro_altitude_ft"),
        pl.lit(50).alias("geom_altitude_ft"),
        pl.Series("callsign", ["DOMINANT"] * 100 + ["OTHER"] * 50),
        pl.lit("N123").alias("registration"),
        pl.lit(True).alias("pia"),
        pl.lit("Owner").alias("owner"),
        pl.lit("C172").alias("aircraft_type"),
    ).reverse()
    result = JUNZIS_SOURCE.create_flights_for_day(DAY, frame, catalog)
    assert result.height == 1
    row = result.row(0, named=True)
    assert row["takeoff_airport_id"] == 1
    assert row["landing_airport_id"] == 2
    assert row["takeoff_airport_ident"] == "FIRST"
    assert row["landing_airport_ident"] == "LAST"
    assert row["first_message_time"] == START
    assert row["last_message_time"] == START + timedelta(seconds=149)
    assert row["first_baro_altitude_ft"] == 0
    assert row["last_geom_altitude_ft"] == 50
    assert row["callsign"] == "DOMINANT"
    assert row["registration"] == "N123"
    assert row["pia"] is True
    assert row["owner"] == "Owner"
    assert row["aircraft_type"] == "C172"


@pytest.mark.parametrize("invalid_lat", [None, float("nan"), float("inf"), 91.0])
def test_invalid_endpoint_positions_preserve_clustering(catalog, invalid_lat):
    frame = observations(range(101)).with_columns(
        pl.Series("lat", [invalid_lat] + [0.0] * 100, dtype=pl.Float64),
    )
    row = JUNZIS_SOURCE.create_flights_for_day(DAY, frame, catalog).row(0, named=True)
    assert row["takeoff_time"] == START
    assert row["takeoff_airport_id"] is None
    assert row["landing_airport_id"] == 2


def test_overnight_clusters_belong_only_to_start_day(catalog):
    frame = observations([*range(-50, 51), *range(86350, 86451)])
    result = JUNZIS_SOURCE.create_flights_for_day(DAY, frame, catalog)
    assert bounds(result) == [
        (START + timedelta(seconds=86350), START + timedelta(seconds=86450)),
    ]


def test_many_aircraft_with_simultaneous_observations(catalog):
    frame = pl.concat([
        observations(range(101), f"{i:06x}")
        for i in range(51)
    ])
    result = JUNZIS_SOURCE.create_flights_for_day(DAY, frame, catalog)
    assert result.height == 51
    assert result["icao"].n_unique() == 51


def test_aircraft_results_are_independent_of_other_aircraft_and_batches(catalog):
    # Include an exact epsilon gap and unrelated aircraft with very different
    # time ranges. Their timestamps must not participate in the same scaler.
    target = observations([*range(101), *range(1900, 2001)], "abc123")
    others = [
        observations(np.arange(101) + i * 1800, f"{i:06x}")
        for i in range(52)
    ]
    expected = pl.concat([
        JUNZIS_SOURCE.create_flights_for_day(DAY, frame, catalog)
        for frame in [target, *others]
    ]).sort("icao", "takeoff_time")
    for frames in ([target, *others], [*others, target]):
        actual = JUNZIS_SOURCE.create_flights_for_day(
            DAY, pl.concat(frames), catalog,
        ).sort("icao", "takeoff_time")
        assert actual.equals(expected)


def test_source_public_import_and_worker_serialization():
    # Keep the public type identity stable when moving the implementation:
    # evaluation fingerprints and multiprocessing workers both depend on it.
    assert JunzisFlightSource.__module__ == "flights.junzis"
    restored = pickle.loads(pickle.dumps(JUNZIS_SOURCE))
    assert type(restored) is JunzisFlightSource
    assert restored == JUNZIS_SOURCE


def test_source_registration_and_evaluation_dispatch(monkeypatch):
    assert FLIGHT_SOURCES_BY_NAME["junzis"] is JUNZIS_SOURCE
    assert isinstance(JUNZIS_SOURCE, flight_source.ADSBFlightSource)
    assert JUNZIS_SOURCE.adsb_icao_batch_size() == 500
    assert JUNZIS_SOURCE.version == 2
    assert JUNZIS_SOURCE.reusable_flight_source() is None
    assert cli.parse_args(["2026-03-01", "2026-03-02", "junzis"]).source == "junzis"
    args = evaluation.parse_args(["--test-source", "junzis", "--adsb-source", "opensky"])
    assert args.test_source == ["junzis"]
    calls = []
    monkeypatch.setattr(evaluation, "get_flights", lambda *args: calls.append(args))
    evaluation.source_flights_reader("junzis", ADSBSource.OPENSKY)(DAY)
    assert calls == [(JUNZIS_SOURCE, DAY, ADSBSource.OPENSKY)]


@pytest.mark.parametrize("provider", list(ADSBSource))
@pytest.mark.parametrize("target_day", [False, True])
def test_pipeline_aircraft_batches_windows_and_cache(
    monkeypatch, tmp_path, catalog, provider, target_day,
):
    source = JunzisFlightSource(target_day_adsb=target_day)
    monkeypatch.setattr(flight_source, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(flight_source, "airport_catalog_for_date", lambda dt: catalog)
    monkeypatch.setattr(JunzisFlightSource, "adsb_icao_batch_size", lambda self: 1)
    frame = pl.concat([observations(range(101)), observations(range(101), "def456")])
    reads = []

    def read(start, end, *, source, columns, lazy=False, icaos=None):
        reads.append((start, end, source, lazy, icaos))
        if lazy:
            assert columns == ["icao"]
            return frame.select("icao").lazy()
        assert "time" in columns and "icao" in columns
        assert len(icaos) == 1
        return frame.filter(pl.col("icao").is_in(icaos))

    monkeypatch.setattr(flight_source, "read_adsb", read)
    # The earlier global-deduplication output must remain untouched and unused.
    old_source = replace(source, version=1)
    old_path = flight_source.flight_path(old_source, DAY, provider)
    old_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"icao": ["old-cache"]}).write_parquet(old_path)
    old_source.flight_manifest(catalog.provenance, provider).write(
        flight_manifest_path(old_path)
    )
    result = flight_source.get_flights(source, DAY, provider)
    assert result["icao"].to_list() == ["abc123", "def456"]
    context = timedelta(days=int(not target_day and provider is not ADSBSource.ADSBX))
    window = (START - context, START + timedelta(days=1) + context, provider)
    assert reads == [
        (*window, True, None),
        (*window, False, ["abc123"]),
        (*window, False, ["def456"]),
    ]
    path = flight_source.flight_path(source, DAY, provider)
    assert f"source=junzis/adsb_source={provider.value}/" in str(path)
    assert ("target_day_adsb=true" in str(path)) is target_day
    assert "/version=2/" in str(path)
    manifest = FlightManifest.read(flight_manifest_path(path))
    assert manifest == source.flight_manifest(catalog.provenance, provider)
    assert manifest.target_day_adsb is target_day
    assert flight_source.get_flights(source, DAY, provider).equals(result)
    assert len(reads) == 3
    assert pl.read_parquet(old_path)["icao"].to_list() == ["old-cache"]


def test_single_day_cli_uses_within_day_aircraft_workers(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        cli, "write_flights",
        lambda *args, **kwargs: calls.append((args, kwargs)) or pl.DataFrame(),
    )
    monkeypatch.setattr(cli, "flight_path", lambda *args: tmp_path / "data.parquet")
    monkeypatch.setattr(cli, "default_worker_count", lambda: 4)
    assert cli.default_workers_for_source(JUNZIS_SOURCE) == 4
    cli.write_flights_range(DAY, DAY + timedelta(days=1), JUNZIS_SOURCE, workers=4)
    assert calls == [
        ((JUNZIS_SOURCE, DAY, ADSBSource.ADSBLOL), {"batch_workers": 4})
    ]


def test_target_day_cli_planning():
    source = replace(JUNZIS_SOURCE, target_day_adsb=True)
    assert cli.adsb_dates_for_run(source, [DAY], ADSBSource.OPENSKY) == [DAY]
