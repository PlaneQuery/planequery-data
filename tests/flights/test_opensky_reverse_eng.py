from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from airports.matcher import AirportMatcher
from airports.models import (
    FIXED_WING_AIRPORT_TYPES,
    Airport,
    AirportCatalogProvenance,
    AirportMatchProvenance,
    AirportType,
)
from data_engineering.adsb.read_adsb import ADSBSource
from data_engineering.opensky import reverse_eng_states as reader
from flights import flight_source as pipeline
from flights.cli import adsb_dates_for_run, parse_args
from flights.flight_manifest import FlightManifest, flight_manifest_path
from flights.flight_type import FLIGHT_POLARS_SCHEMA
from flights.opensky_reverse_eng import (
    EndpointAirports,
    OpenSkyReverseEngFlightSource,
    _clean_batch,
    reconstruct_arrival_day,
)
from flights.opensky_reverse_eng_nearest import OpenSkyReverseEngNearestFlightSource
from flights.sources import FLIGHT_SOURCES_BY_NAME

DAY = date(2026, 3, 1)
EPOCH = int(datetime(2026, 3, 1, tzinfo=timezone.utc).timestamp())


def catalog(airports=None):
    airports = airports or [
        Airport(
            1,
            "CANONICAL",
            AirportType.SMALL_AIRPORT,
            "Test",
            0.0,
            0.0,
            elevation_ft=0,
            gps_code="GPS",
        )
    ]
    result = SimpleNamespace(
        airports=tuple(airports),
        airport_provenance={
            a.id: AirportMatchProvenance.KNOWN_ON_FLIGHT_DATE for a in airports
        },
        source_commit_by_airport_id={a.id: "test" for a in airports},
        provenance=AirportCatalogProvenance(
            DAY, "observed", date(2026, 4, 1), "knowledge", date(2026, 4, 1), "test"
        ),
    )
    result.matcher = AirportMatcher(
        tuple(a for a in airports if a.type in FIXED_WING_AIRPORT_TYPES),
        result.airport_provenance,
        result.source_commit_by_airport_id,
    )
    return result


def states(seconds, *, icao="abcdef", longitude=None, altitude=1000.0):
    seconds = np.asarray(seconds)
    count = len(seconds)
    return pl.DataFrame(
        {
            "icao": [icao] * count,
            "time": EPOCH + seconds + 1,
            "lastposupdate": EPOCH + seconds + 0.25,
            "lat": np.zeros(count),
            "lon": seconds * 100 / 111195 if longitude is None else longitude,
            "baroaltitude": np.full(count, altitude),
            "geoaltitude": np.full(count, altitude + 50),
            "callsign": ["TEST123 "] * count,
        },
        schema=reader.STATE_SCHEMA,
    )


def epochs(frame):
    return frame.select(
        pl.col("takeoff_time").dt.epoch("s"), pl.col("landing_time").dt.epoch("s")
    ).rows()


@pytest.mark.parametrize("name", ["opensky-reverse-eng", "opensky-reverse-eng-nearest"])
def test_registration_input_window_and_provider_restriction(name):
    source = FLIGHT_SOURCES_BY_NAME[name]
    assert isinstance(source, pipeline.ADSBFlightSource)
    assert source.version == 2
    assert (
        parse_args(
            ["2026-03-01", "2026-03-02", source.name, "--adsb-source", "opensky"]
        ).source
        == source.name
    )
    assert adsb_dates_for_run(source, [DAY], ADSBSource.OPENSKY) == [
        DAY + timedelta(days=i) for i in [-1, 0, 1]
    ]
    with pytest.raises(ValueError, match="requires --adsb-source opensky"):
        source.adsb_read_interval_for_source(DAY, ADSBSource.ADSBLOL)


@pytest.mark.parametrize("duration, expected", [(600, 0), (601, 1)])
def test_strict_duration_and_fractional_position_clock(duration, expected):
    # Three leading and six trailing rows are trimmed; final bin stays dense.
    data = states(np.arange(100, 100 + duration + 10))
    result = reconstruct_arrival_day(DAY, data, catalog())
    assert len(result) == expected
    if expected:
        assert epochs(result) == [(EPOCH + 103, EPOCH + 103 + duration)]
        assert result["first_message_time"].equals(
            result["takeoff_time"].rename("first_message_time")
        )
        assert result["callsign"].item() == "TEST123"
        assert result.schema == pl.Schema(FLIGHT_POLARS_SCHEMA)


def test_sparse_bins_and_trim_before_median_outlier_rejection():
    data = states(np.arange(30)).with_columns(
        pl.when(pl.col("time") == EPOCH + 26)
        .then(9000.0)
        .otherwise(pl.col("baroaltitude"))
        .alias("baroaltitude")
    )
    clean = _clean_batch(data)
    assert clean["lastposupdate"][0] == EPOCH + 3.25
    assert clean["lastposupdate"][-1] == EPOCH + 23.25
    sparse = states([0, 1, 2, *range(30, 60)])
    assert _clean_batch(sparse)["lastposupdate"][0] == EPOCH + 33.25


def test_long_gap_uses_displacement_speed_not_reported_velocity():
    times = np.r_[np.arange(100, 1101), np.arange(2101, 3102)]
    cruising = states(times)
    landed = states(
        times, longitude=np.r_[np.arange(1001), np.arange(1000, 2001)] * 100 / 111195
    )
    assert reconstruct_arrival_day(DAY, cruising, catalog()).height == 1
    assert reconstruct_arrival_day(DAY, landed, catalog()).height == 2


@pytest.mark.parametrize("gap, expected_flights", [(600, 1), (601, 2)])
def test_gap_threshold_is_strict(gap, expected_flights):
    times = np.r_[np.arange(100, 1101), np.arange(1100 + gap, 2101 + gap)]
    data = states(
        times, longitude=np.r_[np.arange(1001), np.arange(1000, 2001)] * 100 / 111195
    )
    assert reconstruct_arrival_day(DAY, data, catalog()).height == expected_flights


def test_median_removes_an_interior_altitude_outlier():
    data = states(np.arange(30)).with_columns(
        pl.when(pl.col("time") == EPOCH + 16)
        .then(9000.0)
        .otherwise(pl.col("baroaltitude"))
        .alias("baroaltitude")
    )
    clean = _clean_batch(data)
    assert EPOCH + 15.25 not in clean["lastposupdate"]
    assert len(clean["lastposupdate"]) == 20


def test_raw_pending_context_and_empty_following_day_are_replayed():
    # A cruise segment is closed by a low-speed gap; its final six raw rows
    # must survive into the next daily batch even when that batch is empty.
    times = np.r_[np.arange(-3000, -1999), np.arange(2000, 3001)]
    longitude = np.r_[np.arange(1001), np.arange(1000, 2001)] * 100 / 111195
    result = reconstruct_arrival_day(
        DAY - timedelta(days=1),
        states(times, longitude=longitude, altitude=10000.0),
        catalog(),
    )
    assert epochs(result) == [(EPOCH - 2994, EPOCH - 2000)]
    # At low altitude a near-midnight tail closes after reprocessing an empty day.
    near_midnight = states(np.arange(85000, 86301))
    result = reconstruct_arrival_day(DAY, near_midnight, catalog())
    assert epochs(result) == [(EPOCH + 85006, EPOCH + 86294)]


def test_takeoff_partition_includes_overnight_and_excludes_prior_day():
    source = OpenSkyReverseEngFlightSource()
    previous = states(np.arange(-1000, 1001), icao="000001", longitude=np.zeros(2001))
    overnight = states(np.arange(85000, 87401), icao="000002", longitude=np.zeros(2401))
    result = source.create_flights_for_day(
        DAY, pl.concat([previous, overnight]), catalog()
    )
    assert result["icao"].to_list() == ["000002"]
    assert result["takeoff_time"].dt.date().item() == DAY
    assert result["landing_time"].dt.date().item() == DAY + timedelta(days=1)
    assert result["takeoff_airport_id"].item() == 1
    assert result["takeoff_airport_ident"].item() == "CANONICAL"
    assert source.create_flights_for_day(
        DAY + timedelta(days=1), overnight, catalog()
    ).is_empty()


def test_source_does_not_replay_unavailable_fourth_day():
    # This overnight track is still pending near the end of D+1. Replaying
    # an imaginary empty D+2 would incorrectly finalize it.
    times = np.r_[np.arange(86300, 86400), np.arange(172000, 172701)]
    data = states(times, longitude=np.zeros(len(times)))
    assert reconstruct_arrival_day(DAY + timedelta(days=1), data, catalog()).height == 1
    source = OpenSkyReverseEngFlightSource()
    assert source.create_flights_for_day(DAY, data, catalog()).is_empty()

    # Even callers supplying extra context get the same three-day behavior.
    later = states(np.arange(172900, 174001), longitude=np.zeros(1101))
    assert source.create_flights_for_day(
        DAY, pl.concat([data, later]), catalog()
    ).is_empty()


def test_replay_is_invariant_to_utc_date_across_year_boundary():
    data = states(np.arange(-1000, 1001))
    original = reconstruct_arrival_day(DAY, data, catalog())
    target = date(2027, 1, 1)
    shift = (target - DAY).days * 86400
    moved = data.with_columns(pl.col("time") + shift, pl.col("lastposupdate") + shift)
    shifted = reconstruct_arrival_day(target, moved, catalog())
    assert original.height == shifted.height == 1
    restored = shifted.with_columns(
        pl.col(column) - pl.duration(seconds=shift)
        for column in (
            "takeoff_time",
            "landing_time",
            "first_message_time",
            "last_message_time",
        )
    )
    assert restored.equals(original)


def test_airport_gates_and_eligibility():
    base = catalog().airports[0]
    candidates = [
        replace(base, id=2, gps_code=None),
        replace(base, id=3, type=AirportType.HELIPORT),
        replace(base, id=4, elevation_ft=None),
        replace(base, id=5, elevation_ft=20000),
        replace(base, id=6, lon=0.18),  # about 20 km away
    ]
    airports = EndpointAirports(catalog(candidates))
    assert airports.candidate(0.0, 0.0, 100.0, departure=True) is None
    assert airports.candidate(0.0, 0.0, 100.0, departure=False).id == 6
    assert airports.candidate(0.0, 0.18, 2500.0, departure=False) is None
    assert not airports.close_last(0.0, 0.18, 2500.0)


def test_airports_rank_horizontal_distance_after_vertical_gate():
    base = catalog().airports[0]
    airports = EndpointAirports(
        catalog(
            [
                replace(base, id=2, gps_code="NEAR", lon=0.009, elevation_ft=0),
                replace(base, id=3, gps_code="LEVEL", lon=0.01, elevation_ft=6234),
                replace(base, id=4, gps_code="HIGH", lon=0.0, elevation_ft=20000),
            ]
        )
    )
    # Both NEAR and LEVEL pass the altitude gate; horizontal distance wins.
    assert airports.candidate(0.0, 0.0, 1900.0, departure=True).id == 2


AIRPORT_COLUMNS = (
    "takeoff_airport_id",
    "takeoff_airport_ident",
    "landing_airport_id",
    "landing_airport_ident",
)


def test_nearest_preserves_segments_and_ignores_distance_and_altitude_gates():
    base = catalog().airports[0]
    airports = catalog(
        [
            replace(base, id=2, ident="NEAR", lon=0, elevation_ft=-4000),
            replace(base, id=3, ident="LEVEL", lon=0.02, elevation_ft=0),
        ]
    )
    data = states(np.arange(100, 1101), longitude=np.linspace(0, 1, 1001))
    original = OpenSkyReverseEngFlightSource().create_flights_for_day(
        DAY, data, airports
    )
    nearest = OpenSkyReverseEngNearestFlightSource().create_flights_for_day(
        DAY, data, airports
    )
    assert original.height == nearest.height == 1
    assert original.select(AIRPORT_COLUMNS).row(0) == (3, "LEVEL", None, None)
    # Departure selects the closer airport despite its altitude difference;
    # arrival selects an airport even though it is more than 100 km away.
    assert nearest.select(AIRPORT_COLUMNS).row(0) == (2, "NEAR", 3, "LEVEL")
    assert nearest.drop(AIRPORT_COLUMNS).equals(original.drop(AIRPORT_COLUMNS))


def test_nearest_uses_fixed_wing_catalog_without_gps_or_elevation_requirement():
    base = catalog().airports[0]
    airports = catalog(
        [
            replace(base, lon=0.02),
            replace(
                base, id=2, ident="LOCAL", lon=0.001, gps_code=None, elevation_ft=None
            ),
            replace(base, id=3, ident="HELI", type=AirportType.HELIPORT),
        ]
    )
    data = states(np.arange(100, 1101), longitude=np.zeros(1001))
    original = OpenSkyReverseEngFlightSource().create_flights_for_day(
        DAY, data, airports
    )
    nearest = OpenSkyReverseEngNearestFlightSource().create_flights_for_day(
        DAY, data, airports
    )
    assert original.select(AIRPORT_COLUMNS).row(0) == (1, "CANONICAL", 1, "CANONICAL")
    assert nearest.select(AIRPORT_COLUMNS).row(0) == (2, "LOCAL", 2, "LOCAL")
    assert nearest.drop(AIRPORT_COLUMNS).equals(original.drop(AIRPORT_COLUMNS))


@pytest.mark.parametrize("empty", [False, True])
def test_nearest_reuses_base_cache_without_reading_or_segmenting(
    tmp_path, monkeypatch, empty
):
    base = OpenSkyReverseEngFlightSource()
    source = OpenSkyReverseEngNearestFlightSource()
    airports = catalog()
    data = pl.concat(
        [
            states(np.arange(100, 1101), icao="000001"),
            states(np.arange(2000, 3101), icao="000002"),
        ]
    )
    original = base.create_flights_for_day(DAY, data, airports)
    if empty:
        original = original.clear()
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", tmp_path)
    path = pipeline.flight_path(base, DAY, ADSBSource.OPENSKY)
    path.parent.mkdir(parents=True)
    original.write_parquet(path)
    base.flight_manifest(airports.provenance, ADSBSource.OPENSKY).write(
        flight_manifest_path(path)
    )
    monkeypatch.setattr(
        OpenSkyReverseEngFlightSource,
        "read_adsb",
        lambda *a, **kw: pytest.fail("cached endpoints do not need state vectors"),
    )
    monkeypatch.setattr(
        OpenSkyReverseEngFlightSource,
        "create_flights_for_day",
        lambda *a, **kw: pytest.fail("cached segments must be reused"),
    )
    monkeypatch.setattr(
        OpenSkyReverseEngFlightSource, "adsb_icao_batch_size", lambda self: 1
    )
    nearest = pipeline.get_flights_for_catalog(
        source, DAY, airports, ADSBSource.OPENSKY
    )
    assert nearest.drop(AIRPORT_COLUMNS).equals(original.drop(AIRPORT_COLUMNS))
    assert nearest.schema == original.schema
    if not empty:
        assert (
            nearest.select(AIRPORT_COLUMNS).rows()
            == [(1, "CANONICAL", 1, "CANONICAL")] * 2
        )
    nearest_path = pipeline.flight_path(source, DAY, ADSBSource.OPENSKY)
    assert nearest_path != path
    assert pl.read_parquet(nearest_path).equals(nearest)
    assert pl.read_parquet(path).equals(original)
    assert pipeline.get_flights_for_catalog(
        source, DAY, airports, ADSBSource.OPENSKY
    ).equals(nearest)


def write_raw_days(tmp_path, monkeypatch, data, days):
    by_day = {}
    raw = data.rename({"icao": "icao24"}).with_columns(pl.lit(True).alias("onground"))
    for day in days:
        start = int(
            datetime.combine(day, datetime.min.time(), timezone.utc).timestamp()
        )
        paths = []
        for hour in range(24):
            path = tmp_path / "raw" / str(day) / f"{hour:02d}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            raw.filter(
                pl.col("time").is_between(
                    start + hour * 3600, start + (hour + 1) * 3600, closed="left"
                )
            ).write_parquet(path)
            paths.append(path)
        by_day[day] = paths
    monkeypatch.setattr(reader, "STATE_CACHE_ROOT", tmp_path / "states-cache")
    monkeypatch.setattr(
        reader, "complete_opensky_state_vector_files", lambda day: by_day[day]
    )


def test_raw_reader_preserves_samples_both_clocks_and_ground_altitude(
    tmp_path, monkeypatch
):
    data = states(np.arange(5))
    stale = data.with_columns(pl.col("lastposupdate") - 1)
    missing = data.with_columns(pl.lit(None, dtype=pl.Float64).alias("baroaltitude"))
    raw = pl.concat([data, data.head(1), stale, missing])
    write_raw_days(tmp_path, monkeypatch, raw, [DAY])
    result = reader.read_reverse_eng_states(DAY, icaos=["abcdef"])
    assert result.height == 6  # valid duplicate snapshots are retained
    assert result["lastposupdate"].min() == EPOCH + 0.25
    assert result["time"].min() == EPOCH + 1
    assert result["baroaltitude"].unique().to_list() == [1000.0]
    assert result["callsign"].unique().to_list() == ["TEST123 "]
    assert reader.read_reverse_eng_states(DAY, icaos=["000000"]).is_empty()
    assert reader.read_reverse_eng_states(DAY, icaos=[]).schema == pl.Schema(
        reader.STATE_SCHEMA
    )
    # A completed derived cache no longer needs access to the original hours.
    monkeypatch.setattr(
        reader,
        "complete_opensky_state_vector_files",
        lambda day: pytest.fail("cache should be reused"),
    )
    assert (
        reader.read_reverse_eng_states(DAY, columns=["icao"], lazy=True)
        .collect()
        .height
        == 6
    )


@pytest.mark.parametrize(
    "source_type", [OpenSkyReverseEngFlightSource, OpenSkyReverseEngNearestFlightSource]
)
def test_pipeline_batches_raw_reader_and_writes_honest_manifest(
    tmp_path, monkeypatch, source_type
):
    data = pl.concat(
        [
            states(np.arange(100, 1101), icao="000001", longitude=np.zeros(1001)),
            states(np.arange(85800, 87001), icao="abcdef", longitude=np.zeros(1201)),
        ]
    )
    write_raw_days(
        tmp_path, monkeypatch, data, [DAY + timedelta(days=i) for i in [-1, 0, 1]]
    )
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(
        pipeline,
        "read_adsb",
        lambda *a, **kw: pytest.fail("canonical reader loses required inputs"),
    )
    monkeypatch.setattr(
        OpenSkyReverseEngFlightSource, "adsb_icao_batch_size", lambda self: 1
    )
    source = source_type()
    result = pipeline.get_flights_for_catalog(
        source, DAY, catalog(), ADSBSource.OPENSKY
    )
    assert result["icao"].to_list() == ["000001", "abcdef"]
    assert result["landing_time"].dt.date().to_list() == [DAY, DAY + timedelta(days=1)]
    path = pipeline.flight_path(source, DAY, ADSBSource.OPENSKY)
    assert pl.read_parquet(path).equals(result)
    manifest = FlightManifest.read(flight_manifest_path(path))
    assert manifest.adsb_parquet_version is None
    assert manifest.adsb_input_format == "opensky-reverse-eng-states"
    assert manifest.adsb_input_version == reader.STATE_CACHE_VERSION
    assert pipeline.get_flights_for_catalog(
        source, DAY, catalog(), ADSBSource.OPENSKY
    ).equals(result)


@pytest.mark.parametrize(
    "source_type", [OpenSkyReverseEngFlightSource, OpenSkyReverseEngNearestFlightSource]
)
def test_evaluation_cache_tracks_only_input_versions_actually_used(source_type):
    from flights.evaluation.cache import EvaluationCacheKey
    from flights.sources import ALGORITHM_NEAREST_SOURCE, OPENSKY_SOURCE

    source = source_type()
    raw_only = EvaluationCacheKey.from_inputs(
        DAY, source, OPENSKY_SOURCE, ADSBSource.OPENSKY, catalog()
    )
    mixed = EvaluationCacheKey.from_inputs(
        DAY, source, ALGORITHM_NEAREST_SOURCE, ADSBSource.OPENSKY, catalog()
    )
    assert raw_only.adsb_source == "opensky"
    assert raw_only.adsb_parquet_version is None
    assert mixed.adsb_parquet_version is not None
    manifest = source.flight_manifest(catalog().provenance, ADSBSource.OPENSKY)
    assert (
        manifest.flight_cache_key
        != replace(
            manifest, adsb_input_version=manifest.adsb_input_version + 1
        ).flight_cache_key
    )


def test_incomplete_raw_context_does_not_leave_success_marker(tmp_path, monkeypatch):
    from data_engineering.opensky import read_opensky_trino_states as raw_reader

    monkeypatch.setattr(reader, "STATE_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(raw_reader, "opensky_state_vector_files", lambda day: [])
    with pytest.raises(FileNotFoundError, match="No OpenSky state vector"):
        reader.read_reverse_eng_states(DAY)
    assert not list((tmp_path / "cache").rglob("_SUCCESS"))


def test_reader_checks_later_context_before_building_any_day(tmp_path, monkeypatch):
    monkeypatch.setattr(reader, "STATE_CACHE_ROOT", tmp_path)
    checked = []

    def complete(day):
        checked.append(day)
        if day > DAY:
            raise FileNotFoundError("later context day missing")
        return []

    monkeypatch.setattr(reader, "complete_opensky_state_vector_files", complete)
    monkeypatch.setattr(
        reader,
        "ensure_reverse_eng_states",
        lambda day: pytest.fail("must validate all context before preparing caches"),
    )
    with pytest.raises(FileNotFoundError, match="later context day missing"):
        reader.read_reverse_eng_states(DAY, datetime(2026, 3, 3, tzinfo=timezone.utc))
    assert checked == [DAY, DAY + timedelta(days=1)]
