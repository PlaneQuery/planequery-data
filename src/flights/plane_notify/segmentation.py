"""Replay plane-notify's event detector over canonical ADS-B observations."""

from datetime import datetime, timedelta
import math

import polars as pl

from airports.catalog import AirportCatalogView
from flights.flight_type import FLIGHT_POLARS_SCHEMA, with_flight_schema_columns
from .core import PlaneNotify


UPSTREAM_COMMIT = "e1aa8c668896cd6eaa80c42db813eb0487fb02d9"
POLL_INTERVAL = timedelta(seconds=60)
DATA_LOSS_INTERVAL = timedelta(minutes=5)
INPUT_COLUMNS = (
    "icao", "time", "lat", "lon", "baro_altitude_ft", "on_ground",
    "ground_speed_kt", "callsign", "registration",
)


def _ceil_poll(time: datetime) -> datetime:
    minute = time.replace(second=0, microsecond=0)
    return minute if time == minute else minute + POLL_INTERVAL


def _replay_aircraft(observations, start, end, airport_catalog):
    def closest_airport(lat, lon):
        match = airport_catalog.matcher.nearest(lat, lon)
        elevation = match.airport.elevation_ft
        return {"elevation_ft": "" if elevation is None else elevation}

    detector = PlaneNotify(closest_airport)
    flights = []
    pending = None
    last_observation = None
    last_checked = None

    def check(now):
        nonlocal pending, last_checked
        trigger = detector.check(now)
        if trigger in ("no longer on ground", "data acquisition"):
            # Upstream sets takeoff_time to its clock on every takeoff event.
            origin = last_checked if trigger == "no longer on ground" else last_observation
            pending = (now, origin)
        elif trigger in ("now on ground", "data loss") and pending is not None:
            takeoff_time, origin = pending
            # Upstream subtracts time_since_contact from data-loss duration.
            landing_time = detector.last_pos_datetime if trigger == "data loss" else now
            if landing_time > takeoff_time:
                row = {
                    "icao": origin["icao"],
                    "callsign": origin["callsign"],
                    "registration": origin["registration"],
                    "takeoff_time": takeoff_time,
                    "landing_time": landing_time,
                }
                for endpoint, prefix, observation in (
                    ("takeoff", "first", origin),
                    ("landing", "last", last_observation),
                ):
                    match = airport_catalog.matcher.nearest(observation["lat"], observation["lon"])
                    row.update({
                        f"{endpoint}_airport_id": match.id,
                        f"{endpoint}_airport_ident": match.ident,
                        f"{prefix}_message_time": observation["time"],
                        f"{prefix}_lat": observation["lat"],
                        f"{prefix}_lon": observation["lon"],
                        f"{prefix}_baro_altitude_ft": observation["baro_altitude_ft"],
                    })
                flights.append(row)
            pending = None
        # run_check also updates coordinates on empty polls.
        last_checked = last_observation

    def empty_polls(first, last):
        if first > last:
            return
        detector.feeding = False
        check(first)
        if detector.landing_plausible:
            deadline = max(
                first + POLL_INTERVAL,
                _ceil_poll(detector.last_pos_datetime + DATA_LOSS_INTERVAL),
            )
            if deadline <= last:
                check(deadline)
        # Other empty polls cannot change detection state or produce events.

    previous_poll = start - POLL_INTERVAL
    for observation in observations:
        now = observation["_poll_time"]
        empty_polls(previous_poll + POLL_INTERVAL, now - POLL_INTERVAL)
        previous_poll = now
        lat, lon = observation["lat"], observation["lon"]
        altitude = observation["baro_altitude_ft"]
        on_ground = observation["on_ground"]
        speed = observation["ground_speed_kt"]
        if (
            lat is None or lon is None
            or not math.isfinite(lat) or not math.isfinite(lon)
            or not -90 <= lat <= 90 or not -180 <= lon <= 180
            or on_ground is None
            or (not on_ground and (altitude is None or speed is None))
        ):
            # Invalid API records are skipped by run_adsbx_v2, not run_empty.
            continue
        detector.latitude, detector.longitude = lat, lon
        detector.on_ground = on_ground
        detector.alt_ft = 0 if on_ground else int(altitude)
        detector.last_pos_datetime = observation["time"]
        last_observation = observation
        # Preserve run_adsbx_v2's airborne <=10 kt decode-error guard.
        if not on_ground and speed <= 10:
            continue
        detector.feeding = True
        check(now)
    empty_polls(previous_poll + POLL_INTERVAL, end - POLL_INTERVAL)
    # A pending takeoff remains incomplete if no landing event occurred.
    return flights


def replay_flights(
    df_adsb: pl.DataFrame,
    start: datetime,
    end: datetime,
    airport_catalog: AirportCatalogView,
) -> pl.DataFrame:
    """Poll at UTC minute boundaries, taking the latest position in (t-60s, t].

    Archived positions do not encode API aircraft-retention behavior. An empty
    interval is therefore an empty API response. No polls are synthesized past
    the available input window. Only complete takeoff/landing pairs are emitted.
    """
    if df_adsb.is_empty():
        return pl.DataFrame(schema=FLIGHT_POLARS_SCHEMA)
    observations = (
        df_adsb.filter(
            pl.col("icao").is_not_null()
            & (pl.col("time") >= start) & (pl.col("time") < end)
        )
        .select(INPUT_COLUMNS)
        .sort(["icao", "time"], maintain_order=True)
        .with_columns(
            (
                pl.col("time").dt.truncate("1m")
                + pl.when(pl.col("time") == pl.col("time").dt.truncate("1m"))
                .then(pl.duration(seconds=0))
                .otherwise(pl.duration(seconds=60))
            ).alias("_poll_time")
        )
        .filter(pl.col("_poll_time") < end)
        .unique(subset=["icao", "_poll_time"], keep="last", maintain_order=True)
    )
    rows = []
    for aircraft in observations.partition_by("icao", maintain_order=True):
        rows.extend(_replay_aircraft(aircraft.iter_rows(named=True), start, end, airport_catalog))
    if not rows:
        return pl.DataFrame(schema=FLIGHT_POLARS_SCHEMA)
    return with_flight_schema_columns(pl.DataFrame(rows)).sort(["icao", "takeoff_time"])
