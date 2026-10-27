from dataclasses import replace
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from data_engineering.adsb.read_adsb import ADSBSource
from flights.flight_type import FLIGHT_POLARS_SCHEMA
from flights.openap import OpenAPFlightSource
from flights.openap import segmentation as openap


DAY = date(2026, 3, 1)
START = datetime(2026, 3, 1, 12)


def observations(seconds, *, icao="abc123"):
    n = len(seconds)
    return pl.DataFrame({
        "icao": [icao] * n,
        "time": [START + timedelta(seconds=s) for s in seconds],
        "lat": [40.0] * n,
        "lon": [-75.0] * n,
        "callsign": ["TEST"] * n,
        "registration": ["N123"] * n,
        "baro_altitude_ft": [10000] * n,
        "geom_altitude_ft": pl.Series([None] * n, dtype=pl.Int64),
        "ground_speed_kt": [250.0] * n,
        "baro_rate_fpm": [1000] * n,
        "geom_rate_fpm": pl.Series([None] * n, dtype=pl.Int64),
        "on_ground": [False] * n,
    })


def test_labels_equal_upstream_and_preserve_unlabeled_final_window():
    phase = pytest.importorskip("openap.phase")
    frame = observations(list(range(0, 301, 10)))
    frame = frame.with_columns(
        pl.when(pl.col("time") < START + timedelta(seconds=60))
        .then(0).otherwise(pl.col("baro_altitude_ft")).alias("baro_altitude_ft"),
        pl.when(pl.col("time") < START + timedelta(seconds=60))
        .then(0).otherwise(pl.col("ground_speed_kt")).alias("ground_speed_kt"),
        pl.when(pl.col("time") < START + timedelta(seconds=60))
        .then(0).otherwise(pl.col("baro_rate_fpm")).alias("baro_rate_fpm"),
    )
    detector = phase.FlightPhase()
    detector.set_trajectory(
        frame["time"].dt.epoch("ms").to_numpy() / 1000,
        frame["baro_altitude_ft"].to_numpy(),
        frame["ground_speed_kt"].to_numpy(),
        frame["baro_rate_fpm"].to_numpy(),
    )
    labeled = openap.label_phases(frame.reverse())
    assert labeled["phase"].to_list() == detector.phaselabel()
    assert labeled["phase"].to_list() == ["GND"] * 6 + ["CL"] * 24 + ["NA"]
    flights = openap.flights_from_phases(labeled, DAY)
    assert flights.schema == FLIGHT_POLARS_SCHEMA
    assert flights["takeoff_time"].to_list() == [START + timedelta(seconds=60)]
    assert flights["landing_time"].to_list() == [START + timedelta(seconds=290)]


def test_phase_changes_remain_one_flight_but_ground_and_na_separate():
    frame = observations(list(range(0, 120, 10))).with_columns(
        pl.Series("phase", ["GND", "CL", "CR", "DE", "LVL", "CL",
                            "GND", "CL", "DE", "NA", "LVL", "LVL"])
    )
    flights = openap.flights_from_phases(frame, DAY)
    assert flights["takeoff_time"].to_list() == [
        START + timedelta(seconds=s) for s in (10, 70, 100)
    ]
    assert flights["landing_time"].to_list() == [
        START + timedelta(seconds=s) for s in (50, 80, 110)
    ]


def test_gap_rule_and_aircraft_are_independent():
    frame = observations([0, 600, 1201, 1211]).with_columns(pl.lit("CL").alias("phase"))
    second = observations([0, 10], icao="def456").with_columns(pl.lit("CR").alias("phase"))
    flights = openap.flights_from_phases(pl.concat([second, frame]).reverse(), DAY)
    assert flights["icao"].to_list() == ["abc123", "abc123", "def456"]
    assert flights["landing_time"][0] == START + timedelta(seconds=600)


def test_detector_resets_time_windows_after_gap(monkeypatch):
    calls = []

    class Detector:
        def set_trajectory(self, times, altitude, speed, vertical_rate):
            calls.append(times)
            self.n = len(times)

        def phaselabel(self):
            return ["CL"] * self.n

    monkeypatch.setattr(openap, "_phase_detector", Detector)
    labeled = openap.label_phases(observations([0, 10, 611, 621]))
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[1] - calls[0], [611, 611])
    assert openap.flights_from_phases(labeled, DAY).height == 2


def test_input_missing_values_ground_encoding_and_duplicates():
    frame = observations([0, 10, 20, 30, 40, 40]).with_columns(
        pl.Series("baro_altitude_ft", [None, None, None, 10000, 10000, 10000]),
        pl.Series("geom_altitude_ft", [None, 12000, None, None, None, None], dtype=pl.Int64),
        pl.Series("baro_rate_fpm", [None, None, None, 1000, 1000, 1000]),
        pl.Series("geom_rate_fpm", [None, -500, None, None, None, None], dtype=pl.Int64),
        pl.Series("on_ground", [True, False, False, False, False, False]),
        pl.Series("ground_speed_kt", [0.0, 200, 200, float("nan"), 250, 250]),
    )
    prepared = openap.prepare_observations(frame)
    assert prepared["_altitude"].to_list() == [0, 12000, 10000]
    assert prepared["_vertical_rate"].to_list() == [0, -500, 1000]
    assert prepared.height == 3


def test_takeoff_day_filter_preserves_overnight_flight_and_rejects_singleton():
    frame = observations([-43210, -43200, 43190, 43200, 43210]).with_columns(
        pl.Series("phase", ["CL", "DE", "CL", "CR", "DE"])
    )
    singleton = observations([100], icao="def456").with_columns(pl.lit("CL").alias("phase"))
    flights = openap.flights_from_phases(pl.concat([frame, singleton]), DAY)
    assert flights.height == 1
    assert flights["takeoff_time"][0] == datetime(2026, 3, 1, 23, 59, 50)
    assert flights["landing_time"][0] == datetime(2026, 3, 2, 0, 0, 10)


def test_empty_and_short_inputs():
    pytest.importorskip("openap.phase")
    for frame in (observations([]), observations([0, 10])):
        flights = openap.create_flights_from_adsb(frame, DAY)
        assert flights.is_empty()
        assert flights.schema == FLIGHT_POLARS_SCHEMA


def test_source_integrates_registry_cli_airports_and_input_windows():
    pytest.importorskip("openap.phase")
    from flights.cli import parse_args
    from flights.evaluation.evaluation import resolve_source

    source = resolve_source("openap")
    assert isinstance(source, OpenAPFlightSource)
    assert parse_args(["2026-03-01", "2026-03-02", "openap"]).source == "openap"
    assert source.adsb_input_interval(DAY, ADSBSource.ADSBLOL) == (
        datetime(2026, 2, 28), datetime(2026, 3, 3),
    )
    assert replace(source, target_day_adsb=True).adsb_input_interval(DAY, ADSBSource.ADSBLOL) == (
        datetime(2026, 3, 1), datetime(2026, 3, 2),
    )
    airport = SimpleNamespace(id=1, ident="KTEST")
    catalog = SimpleNamespace(matcher=SimpleNamespace(nearest=lambda lat, lon: airport))
    flights = source.create_flights_for_day(DAY, observations([0, 30, 60, 90, 120]), catalog)
    assert flights["takeoff_airport_id"].to_list() == [1]
    assert flights["landing_airport_ident"].to_list() == ["KTEST"]
