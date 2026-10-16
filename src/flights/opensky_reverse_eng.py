"""Empirical OpenSky flights_data4 reconstruction from original state vectors.

Rules were inferred on 2024-06-01 and validated unchanged on 2026-03-01.
This is an approximation, not recovered OpenSky production source. In
particular, the shared OurAirports catalog is not OpenSky's private catalog.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from itertools import pairwise

import numpy as np
import polars as pl
from scipy.ndimage import median_filter

from airports.catalog import AirportCatalogView
from airports.matcher import AirportMatcher
from airports.models import (
    EARTH_RADIUS_KM,
    FIXED_WING_AIRPORT_TYPES,
    Airport,
    AirportCatalogProvenance,
)
from data_engineering.adsb.read_adsb import ADSBSource
from data_engineering.opensky.reverse_eng_states import (
    STATE_CACHE_VERSION,
    STATE_SCHEMA,
    read_reverse_eng_states,
    valid_position,
)
from flights.flight_manifest import FlightManifest
from flights.flight_source import ADSBFlightSource
from flights.flight_type import FLIGHT_POLARS_SCHEMA, with_flight_schema_columns

EARTH_RADIUS_M = 6_371_000.8
SECONDS_PER_DAY = 86_400


class EndpointAirports:
    """Apply the inferred gates using the existing catalog/spatial matcher."""

    def __init__(self, catalog: AirportCatalogView):
        airports = tuple(
            airport
            for airport in catalog.airports
            if airport.type in FIXED_WING_AIRPORT_TYPES
            and airport.gps_code is not None
            and airport.elevation_ft is not None
            and all(
                math.isfinite(value)
                for value in (
                    airport.lat,
                    airport.lon,
                    airport.elevation_ft,
                )
            )
        )
        if not airports:
            raise ValueError(
                "OpenSky reconstruction needs fixed-wing airports with GPS codes and elevations"
            )
        self.matcher = AirportMatcher(
            airports,
            catalog.airport_provenance,
            catalog.source_commit_by_airport_id,
        )

    def close_last(self, lat: float, lon: float, altitude_m: float) -> bool:
        airport = self.matcher.nearest(lat, lon).airport
        return abs(altitude_m - airport.elevation_ft * 0.3048) < 2500

    def candidate(
        self, lat: float, lon: float, altitude_m: float, *, departure: bool
    ) -> Airport | None:
        horizontal_m, vertical_m = (15_000, 2000) if departure else (30_000, 2500)
        # AirportMatcher reports kilometres using the shared radius; convert
        # its gate/distances to the radius used by the inferred implementation.
        radius_km = horizontal_m * EARTH_RADIUS_KM / EARTH_RADIUS_M
        candidates = [
            (distance_km, airport.gps_code, airport.id, airport)
            for airport, distance_km in self.matcher.airports_within_radius(
                lat, lon, radius_km
            )
            if distance_km * EARTH_RADIUS_M / EARTH_RADIUS_KM < horizontal_m
            and abs(altitude_m - airport.elevation_ft * 0.3048) < vertical_m
        ]
        return min(candidates, key=lambda item: item[:3])[3] if candidates else None


def _clean_batch(states: pl.DataFrame) -> dict[str, np.ndarray]:
    states = states.filter(valid_position()).sort("lastposupdate", maintain_order=True)
    values = {
        column: states[column].to_numpy() for column in STATE_SCHEMA if column != "icao"
    }
    clock = values["lastposupdate"]
    _, inverse, counts = np.unique(
        np.floor(clock / 30), return_inverse=True, return_counts=True
    )
    values = {column: data[counts[inverse] > 3] for column, data in values.items()}
    count = len(values["time"])
    keep = np.ones(count, dtype=bool)
    if count >= 7:
        for column, threshold in (("baroaltitude", 50), ("lat", 0.5), ("lon", 0.5)):
            medians = median_filter(values[column], size=7, mode="nearest")
            # Match R runmed(endrule='constant'), including its edge values.
            medians[:3] = np.median(values[column][:7])
            medians[-3:] = np.median(values[column][-7:])
            keep &= np.abs(values[column] - medians) <= threshold
    # Trim positions in the dense batch, before applying the median mask.
    keep[:3] = False
    keep[max(0, count - 6) :] = False
    return {column: data[keep] for column, data in values.items()}


def _split_points(values: dict[str, np.ndarray]) -> np.ndarray:
    elapsed = np.diff(values["lastposupdate"])
    lat = np.radians(values["lat"])
    lon = np.radians(values["lon"])
    haversine = (
        np.sin(np.diff(lat) / 2) ** 2
        + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(np.diff(lon) / 2) ** 2
    )
    distance = 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))
    speed = np.divide(
        distance, elapsed, out=np.full(len(elapsed), np.nan), where=elapsed > 0
    )
    median_speed = np.nanmedian(speed) if np.any(np.isfinite(speed)) else np.nan
    split = (elapsed > 600) & (speed < median_speed / 4)
    return np.r_[0, np.flatnonzero(split) + 1, len(values["time"])]


def _epoch(day: date) -> int:
    return int(datetime.combine(day, datetime.min.time(), timezone.utc).timestamp())


def _record(
    icao: str,
    values: dict[str, np.ndarray],
    start: int,
    end: int,
    airports: EndpointAirports,
) -> dict:
    callsigns = values["callsign"][start:end]
    callsigns = callsigns[callsigns != None]
    mode = ""
    if len(callsigns):
        unique, counts = np.unique(callsigns, return_counts=True)
        # Count original strings first; normalize only the final flight label.
        mode = str(unique[np.argmax(counts)]).strip()
    row = {"icao": icao, "callsign": mode}
    for prefix, endpoint, index in (
        ("first", "takeoff", start),
        ("last", "landing", end - 1),
    ):
        clock = int(values["lastposupdate"][index])
        timestamp = datetime.fromtimestamp(clock, timezone.utc).replace(tzinfo=None)
        lat, lon = float(values["lat"][index]), float(values["lon"][index])
        altitude = float(values["baroaltitude"][index])
        airport = airports.candidate(
            lat, lon, altitude, departure=endpoint == "takeoff"
        )
        row.update(
            {
                f"{endpoint}_time": timestamp,
                f"{endpoint}_airport_id": airport.id if airport else None,
                f"{endpoint}_airport_ident": airport.ident if airport else None,
                f"{prefix}_message_time": timestamp,
                f"{prefix}_lat": lat,
                f"{prefix}_lon": lon,
                f"{prefix}_baro_altitude_ft": int(altitude / 0.3048),
            }
        )
        geometric = values["geoaltitude"][index]
        row[f"{prefix}_geom_altitude_ft"] = (
            int(geometric / 0.3048) if np.isfinite(geometric) else None
        )
    return row


def reconstruct_arrival_day(
    day: date,
    states: pl.DataFrame,
    airport_catalog: AirportCatalogView,
    *,
    airports: EndpointAirports | None = None,
    replay_end: date | None = None,
) -> pl.DataFrame:
    """Reproduce one OpenSky lastseen day from three UTC snapshot days.

    Expects the original clocks/metre values, with ``icao24`` renamed to
    ``icao`` (the raw-input reader performs this rename). Missing context must
    be checked by the reader; an empty complete day is valid input.
    ``replay_end`` caps the exclusive UTC snapshot boundary, so unavailable
    later days are not replayed as if they contained no observations.
    """
    if states.is_empty():
        return pl.DataFrame(schema=FLIGHT_POLARS_SCHEMA)
    missing = set(STATE_SCHEMA) - set(states.columns)
    if missing:
        raise ValueError(
            f"OpenSky reconstruction requires raw state-vector columns: {sorted(missing)}"
        )
    airports = airports or EndpointAirports(airport_catalog)
    target = _epoch(day)
    replay_stop = target + 2 * SECONDS_PER_DAY
    if replay_end is not None:
        if replay_end <= day:
            raise ValueError("replay_end must be after the arrival day")
        replay_stop = min(replay_stop, _epoch(replay_end))
    states = (
        states.filter(
            valid_position()
            & (pl.col("time") >= target - SECONDS_PER_DAY)
            & (pl.col("time") < replay_stop)
        )
        .select(pl.col(c).cast(dtype) for c, dtype in STATE_SCHEMA.items())
        .sort(["icao", "time"], maintain_order=True)
    )
    output = []
    for (icao,), aircraft in states.partition_by("icao", as_dict=True).items():
        if icao is None:
            continue
        carry = None
        for snapshot_day in range(
            target - SECONDS_PER_DAY, replay_stop, SECONDS_PER_DAY
        ):
            new = aircraft.filter(
                (pl.col("time") >= snapshot_day)
                & (pl.col("time") < snapshot_day + SECONDS_PER_DAY)
            )
            combined = pl.concat([carry, new]) if carry is not None else new
            carry = None
            if combined.is_empty():
                continue
            values = _clean_batch(combined)
            if len(values["time"]) < 2:
                continue
            cuts = _split_points(values)
            last_clock = values["lastposupdate"][-1]
            close_last = (
                snapshot_day + SECONDS_PER_DAY - last_clock > 600
                and airports.close_last(
                    values["lat"][-1], values["lon"][-1], values["baroaltitude"][-1]
                )
            )
            for number, (start, end) in enumerate(pairwise(cuts)):
                first, last = (
                    int(values["lastposupdate"][i]) for i in (start, end - 1)
                )
                if number == len(cuts) - 2 and not close_last:
                    # Retain the unfiltered tail, including rolling-window
                    # context, even when the segment is still shorter than 10m.
                    carry = combined.filter(
                        pl.col("lastposupdate") >= values["lastposupdate"][start]
                    )
                elif last - first > 600 and target <= last < target + SECONDS_PER_DAY:
                    output.append(_record(icao, values, start, end, airports))
    if not output:
        return pl.DataFrame(schema=FLIGHT_POLARS_SCHEMA)
    return with_flight_schema_columns(
        pl.DataFrame(
            output,
            schema_overrides={
                column: dtype
                for column, dtype in FLIGHT_POLARS_SCHEMA.items()
                if column in output[0]
            },
        )
    ).sort(["icao", "takeoff_time", "landing_time"])


@dataclass(frozen=True)
class OpenSkyReverseEngFlightSource(ADSBFlightSource):
    name: str = "opensky-reverse-eng"
    version: int = 2

    def adsb_icao_batch_size(self) -> int:
        return 500

    def adsb_columns(self) -> Sequence[str]:
        return tuple(STATE_SCHEMA)

    def adsb_read_interval(self, dt: date) -> tuple[datetime, datetime]:
        midnight = datetime.combine(dt, datetime.min.time())
        # Read the departure day plus one day of context on each side.
        return midnight - timedelta(days=1), midnight + timedelta(days=2)

    @staticmethod
    def _validate_provider(source: ADSBSource | None) -> None:
        if source is not ADSBSource.OPENSKY:
            raise ValueError(
                "opensky-reverse-eng requires --adsb-source opensky and raw state_vectors_data4"
            )

    def adsb_read_interval_for_source(
        self, dt: date, adsb_src: ADSBSource
    ) -> tuple[datetime, datetime]:
        self._validate_provider(adsb_src)
        return self.adsb_read_interval(dt)

    def read_adsb(
        self, start, end=None, *, source, columns=None, icaos=None, lazy=False
    ):
        self._validate_provider(source)
        return read_reverse_eng_states(
            start, end, columns=columns, icaos=icaos, lazy=lazy
        )

    def flight_manifest(
        self, provenance: AirportCatalogProvenance, adsb_src: ADSBSource | None
    ) -> FlightManifest:
        self._validate_provider(adsb_src)
        return replace(
            super().flight_manifest(provenance, adsb_src),
            adsb_parquet_version=None,
            adsb_input_format="opensky-reverse-eng-states",
            adsb_input_version=STATE_CACHE_VERSION,
        )

    def create_flights_for_day(
        self, dt: date, df_adsb: pl.DataFrame, airport_catalog: AirportCatalogView
    ) -> pl.DataFrame:
        if df_adsb.is_empty():
            return pl.DataFrame(schema=FLIGHT_POLARS_SCHEMA)
        airports = EndpointAirports(airport_catalog)
        arrivals = pl.concat(
            [
                reconstruct_arrival_day(
                    day,
                    df_adsb,
                    airport_catalog,
                    airports=airports,
                    replay_end=dt + timedelta(days=2),
                )
                for day in (dt, dt + timedelta(days=1))
            ]
        )
        return arrivals.filter(pl.col("takeoff_time").dt.date() == dt).sort(
            ["icao", "takeoff_time"]
        )
