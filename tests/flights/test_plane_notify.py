from dataclasses import replace
from datetime import date, datetime, timedelta
from copy import deepcopy
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import polars as pl
import pytest

from data_engineering.adsb.read_adsb import ADSBSource
from flights import flight_source
from flights.flight_type import FLIGHT_POLARS_SCHEMA
from flights.plane_notify import PlaneNotifyFlightSource, replay_flights
from flights.plane_notify.core import PlaneNotify
from flights.sources import FLIGHT_SOURCES_BY_NAME


DAY = date(2026, 3, 1)
START = datetime(2026, 3, 1)


def catalog():
    return SimpleNamespace(matcher=SimpleNamespace(nearest=lambda lat, lon: SimpleNamespace(
        id=1 if lon < 1 else 2,
        ident="ORIGIN" if lon < 1 else "DEST",
        airport=SimpleNamespace(elevation_ft=0),
    )))


def positions(points):
    """Points are (seconds, altitude/None for ground[, speed])."""
    return pl.DataFrame([
        {
            "icao": "abcdef", "time": START + timedelta(seconds=seconds),
            "lat": 0.0, "lon": index * 0.5,
            "on_ground": altitude is None,
            "baro_altitude_ft": altitude,
            "ground_speed_kt": speed[0] if speed else 100.0,
            "callsign": "TEST1", "registration": "N123",
        }
        for index, (seconds, altitude, *speed) in enumerate(points)
    ], schema_overrides={"baro_altitude_ft": pl.Int64})


def replay(points, end=3600):
    return replay_flights(positions(points), START, START + timedelta(seconds=end), catalog())


def seconds(frame):
    return [tuple((time - START).total_seconds() for time in row)
            for row in frame.select("takeoff_time", "landing_time").iter_rows()]


def test_registration_ground_transitions_and_schema():
    assert isinstance(FLIGHT_SOURCES_BY_NAME["plane-notify"], flight_source.ADSBFlightSource)
    frame = replay([(0, None), (60, 1000), (120, 5000), (180, None)])
    assert seconds(frame) == [(60, 180)]
    assert frame.schema == pl.Schema(FLIGHT_POLARS_SCHEMA)
    assert frame["first_message_time"].item() == START
    assert frame["takeoff_airport_ident"].item() == "ORIGIN"
    assert frame["landing_airport_ident"].item() == "DEST"


def test_first_airborne_poll_requires_prior_empty_poll():
    assert replay([(0, 1000), (60, 2000), (120, None)]).is_empty()
    assert seconds(replay([(60, 1000), (120, 2000), (180, None)])) == [(60, 180)]


def test_timeout_backdates_landing_and_reacquisition_starts_next_flight():
    frame = replay([(60, 1000), (120, 2000), (600, 1000), (660, None)])
    assert seconds(frame) == [(60, 120), (600, 660)]


def test_timeout_uses_position_age_and_landing_uses_position_time():
    assert replay([(1, 1000), (121, 2000)], end=480).is_empty()
    assert seconds(replay([(1, 1000), (121, 2000)], end=540)) == [(60, 121)]


@pytest.mark.parametrize("next_time, count", [(360, 1), (420, 1), (480, 2)])
def test_five_minute_timeout_requires_an_empty_poll(next_time, count):
    # At t=420 the returning observation wins over an empty timeout poll.
    frame = replay([(60, 1000), (120, 2000), (next_time, 1000), (next_time + 60, None)])
    assert frame.height == count


def test_high_altitude_gap_does_not_land_and_missing_landing_is_not_fabricated():
    assert replay([(60, 1000), (120, 20000)]).is_empty()
    # Upstream reacquisition overwrites its pending takeoff timestamp, even
    # when the preceding high-altitude gap did not produce a landing.
    assert seconds(replay([(60, 1000), (120, 20000), (600, 1000), (660, None)])) == [(600, 660)]
    assert replay([(60, 1000), (120, 2000)], end=420).is_empty()
    assert seconds(replay([(60, 1000), (120, 2000)], end=480)) == [(60, 120)]


def test_last_airborne_altitude_must_be_low_for_ground_landing():
    assert replay([(60, 1000), (120, 20000), (180, None)]).is_empty()


def test_fifteen_thousand_equality_and_short_gap_state_are_preserved():
    state = PlaneNotify(lambda lat, lon: {"elevation_ft": 0})
    state.feeding = False
    state.check(START)
    state.feeding = True
    state.on_ground = False
    state.last_pos_datetime = START
    state.alt_ft = 15000
    assert state.check(START) is None
    assert state.below_desired_ft is False
    state.alt_ft = 14999
    state.check(START)
    state.alt_ft = 15000
    state.check(START)
    assert state.below_desired_ft is True
    state.feeding = False
    state.check(START + timedelta(seconds=60))
    assert state.landing_plausible is True
    state.feeding = True
    state.check(START + timedelta(seconds=120))
    assert state.landing_plausible is True


def test_minute_poll_uses_latest_observation_and_rejects_slow_airborne_decode():
    assert seconds(replay([(0, None), (1, 1000), (59, None), (61, 1000), (181, None)])) == [(120, 240)]
    assert seconds(replay([(0, None), (60, 1000, 10), (120, 1000, 11), (180, None)])) == [(120, 180)]


def test_sorted_independent_aircraft_and_empty_input():
    data = positions([(0, None), (60, 1000), (120, None)])
    two = pl.concat([data, data.with_columns(pl.lit("123abc").alias("icao"))]).reverse()
    result = replay_flights(two, START, START + timedelta(hours=1), catalog())
    assert result["icao"].to_list() == ["123abc", "abcdef"]
    assert replay_flights(data.head(0), START, START + timedelta(hours=1), catalog()).schema == result.schema


def test_overnight_flights_use_takeoff_day_and_target_day_cannot_force_close():
    data = positions([(86340, None), (86399, 1000), (86460, 2000), (86520, None)])
    source = PlaneNotifyFlightSource()
    # Minute polling puts this takeoff at precisely the following midnight.
    assert source.create_flights_for_day(DAY, data, catalog()).is_empty()
    result = source.create_flights_for_day(DAY + timedelta(days=1), data, catalog())
    assert seconds(result) == [(86400, 86520)]
    data = positions([(86220, None), (86280, 1000), (86340, 2000), (86520, None)])
    assert seconds(source.create_flights_for_day(DAY, data, catalog())) == [(86280, 86520)]
    assert replace(source, target_day_adsb=True).create_flights_for_day(DAY, data, catalog()).is_empty()


def test_read_preserves_requested_window_without_guessing_from_positions(monkeypatch):
    source = PlaneNotifyFlightSource()
    data = positions([(86220, None), (86280, 1000), (86340, 2000)])
    monkeypatch.setattr(flight_source.ADSBFlightSource, "read_adsb", lambda *args, **kwargs: data)
    start, end = source.adsb_read_interval_for_source(DAY, ADSBSource.ADSBX)
    tagged = source.read_adsb(start, end, source=ADSBSource.ADSBX)
    assert source.create_flights_for_day(DAY, tagged, catalog()).is_empty()
    assert source.adsb_read_interval_for_source(DAY, ADSBSource.ADSBLOL) == (
        START - timedelta(days=1), START + timedelta(days=2)
    )


def test_detection_matches_unmodified_upstream_branches(monkeypatch):
    # Verbatim planeClass.py:313-372 at e1aa8c668896cd6eaa80c42db813eb0487fb02d9.
    # GPL-3.0; see src/flights/plane_notify/LICENSE. Execute the
    # original snippet independently, retaining its imports, prints and lookup.
    original = Path(__file__).with_name("fixtures").joinpath("plane_notify_checks.txt").read_text()
    namespace = {"print": lambda *args: None}
    exec("def original_check(self, time_since_contact):\n" + original + '\n        return locals().get("trigger_type")\n', namespace)
    upstream_check = namespace["original_check"]
    lookup = lambda lat, lon, *args: {"elevation_ft": "" if lat is None else 5000}
    monkeypatch.setitem(sys.modules, "defAirport", SimpleNamespace(getClosestAirport=lookup))
    adapter = PlaneNotify(lookup)
    upstream = deepcopy(adapter)
    upstream.config = SimpleNamespace(get=lambda *args: "[small_airport, medium_airport, large_airport]")
    rng = random.Random(20260301)
    for tick in range(5000):
        now = START + timedelta(seconds=tick * 60)
        if rng.random() < 0.4:
            values = dict(
                feeding=True,
                on_ground=rng.choice([True, False]),
                alt_ft=rng.choice([0, 1000, 14999, 15000, 15001, 20000]),
                last_pos_datetime=now,
                latitude=rng.choice([None, 0.0]),
                longitude=0.0,
            )
        else:
            values = {"feeding": False}
        adapter.__dict__.update(values)
        upstream.__dict__.update(values)
        elapsed = None if upstream.last_pos_datetime is None else now - upstream.last_pos_datetime
        expected = upstream_check(upstream, elapsed)
        upstream.last_feeding = upstream.feeding
        upstream.last_on_ground = upstream.on_ground
        upstream.last_below_desired_ft = upstream.below_desired_ft
        assert adapter.check(now) == expected
        for field in ("tookoff", "landed", "on_ground", "below_desired_ft", "landing_plausible", "last_feeding", "last_on_ground", "last_below_desired_ft"):
            assert getattr(adapter, field) == getattr(upstream, field), (tick, field)


def test_skipping_idle_polls_matches_minute_by_minute_replay():
    for seed in range(10):
        rng = random.Random(seed)
        points = [
            (tick * 60, rng.choice([None, 1000, 14999, 15000, 20000]))
            for tick in range(360) if rng.random() < 0.2
        ]
        data = positions(points)
        by_time = {row["time"]: row for row in data.iter_rows(named=True)}
        detector = PlaneNotify(lambda lat, lon: {"elevation_ft": 0})
        pending = None
        expected = []
        for tick in range(360):
            now = START + timedelta(seconds=tick * 60)
            observation = by_time.get(now)
            detector.feeding = observation is not None
            if observation is not None:
                detector.on_ground = observation["on_ground"]
                detector.alt_ft = 0 if detector.on_ground else observation["baro_altitude_ft"]
                detector.latitude = observation["lat"]
                detector.longitude = observation["lon"]
                detector.last_pos_datetime = now
            event = detector.check(now)
            if event in ("no longer on ground", "data acquisition"):
                pending = now
            elif event in ("now on ground", "data loss") and pending is not None:
                landed = now if event == "now on ground" else detector.last_pos_datetime
                if landed > pending:
                    expected.append(((pending - START).total_seconds(), (landed - START).total_seconds()))
                pending = None
        assert seconds(replay_flights(data, START, START + timedelta(hours=6), catalog())) == expected
