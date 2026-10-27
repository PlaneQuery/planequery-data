"""Derive flights from unmodified OpenAP fuzzy phase labels.

This is the detector used by traffic's Flight.phases(). A flight is a run of
CL/CR/DE/LVL observations; GND/NA and gaps over traffic's default ten minutes
separate runs. No airport, callsign, duration, or distance rules alter them.
"""

from datetime import date

import numpy as np
import polars as pl

from flights.flight_type import FLIGHT_POLARS_SCHEMA, with_flight_schema_columns


OPENAP_VERSION = "2.6.2"
MAX_GAP_SECONDS = 600
AIRBORNE_PHASES = ("CL", "CR", "DE", "LVL")
ADSB_COLUMNS = (
    "icao", "time", "lat", "lon", "callsign", "registration",
    "baro_altitude_ft", "geom_altitude_ft", "ground_speed_kt",
    "baro_rate_fpm", "geom_rate_fpm", "on_ground",
)


def _phase_detector():
    # Keep the optional dependency out of unrelated flight-source imports.
    from importlib.metadata import version

    try:
        from openap.phase import FlightPhase
    except ImportError as exc:
        raise ImportError(
            'OpenAP segmentation requires pip install -e ".[openap]"'
        ) from exc
    if version("openap") != OPENAP_VERSION:
        raise RuntimeError(
            f"This flight-source version requires openap=={OPENAP_VERSION}; "
            "changing the detector requires a flight-source version bump."
        )
    return FlightPhase()


def prepare_observations(df: pl.DataFrame) -> pl.DataFrame:
    """Map canonical units to OpenAP inputs without resampling/interpolation.

    readsb's literal ground altitude is stored as null plus on_ground=True.
    Restore zero only for missing ground altitude/rate. Otherwise prefer baro,
    fall back to geometric values, and omit incomplete/nonfinite observations.
    Groundspeed follows traffic's wrapper (OpenAP calls its argument TAS).
    """
    ground_default = pl.when(pl.col("on_ground")).then(0)
    return (
        df.with_columns(
            pl.coalesce("baro_altitude_ft", "geom_altitude_ft", ground_default)
            .cast(pl.Float64).alias("_altitude"),
            pl.coalesce("baro_rate_fpm", "geom_rate_fpm", ground_default)
            .cast(pl.Float64).alias("_vertical_rate"),
        )
        .filter(
            pl.col("icao").is_not_null(),
            pl.col("time").is_not_null(),
            pl.all_horizontal(
                pl.col("_altitude", "ground_speed_kt", "_vertical_rate")
                .is_finite()
            ),
        )
        .sort("icao", "time", maintain_order=True)
        .unique(subset=["icao", "time"], keep="first", maintain_order=True)
    )


def label_phases(df: pl.DataFrame) -> pl.DataFrame:
    """Label each aircraft and gap-delimited trace using upstream defaults."""
    if df.is_empty():
        return df.with_columns(pl.lit(None, dtype=pl.String).alias("phase"))
    observations = prepare_observations(df)
    if observations.is_empty():
        return observations.with_columns(pl.lit(None, dtype=pl.String).alias("phase"))
    detector = _phase_detector()
    parts = []
    for aircraft in observations.partition_by("icao", maintain_order=True):
        timestamps = aircraft["time"].dt.epoch("ms").to_numpy() / 1000
        boundaries = np.r_[0, np.flatnonzero(np.diff(timestamps) > MAX_GAP_SECONDS) + 1,
                           len(timestamps)]
        labels = []
        altitude = aircraft["_altitude"].to_numpy()
        speed = aircraft["ground_speed_kt"].to_numpy()
        vertical_rate = aircraft["_vertical_rate"].to_numpy()
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            detector.set_trajectory(
                timestamps[start:end], altitude[start:end],
                speed[start:end], vertical_rate[start:end],
            )
            # Upstream intentionally remains untouched, including its unlabeled
            # final window. NA is not interpreted as evidence of flight.
            labels.extend(detector.phaselabel())
        parts.append(aircraft.with_columns(pl.Series("phase", labels)))
    return pl.concat(parts)


def flights_from_phases(labeled: pl.DataFrame, target_date: date) -> pl.DataFrame:
    """Use first/last airborne observations as the flight endpoint estimates."""
    if labeled.is_empty():
        return pl.DataFrame(schema=FLIGHT_POLARS_SCHEMA)
    labeled = labeled.sort("icao", "time").with_columns(
        pl.col("phase").is_in(AIRBORNE_PHASES).fill_null(False).alias("_airborne")
    ).with_columns(
        (
            (pl.col("icao") != pl.col("icao").shift())
            | (pl.col("_airborne") != pl.col("_airborne").shift())
            | (pl.col("time").diff().dt.total_milliseconds() > MAX_GAP_SECONDS * 1000)
        ).fill_null(True).cum_sum().alias("_segment")
    )
    flights = labeled.filter("_airborne").group_by("_segment", maintain_order=True).agg(
        pl.col("icao").first(),
        pl.col("callsign", "registration").drop_nulls().first(),
        pl.col("time").first().alias("takeoff_time"),
        pl.col("time").last().alias("landing_time"),
        pl.col("time").first().alias("first_message_time"),
        pl.col("time").last().alias("last_message_time"),
        *[
            getattr(pl.col(column), aggregation)().alias(f"{endpoint}_{column}")
            for endpoint, aggregation in (("first", "first"), ("last", "last"))
            for column in ("lat", "lon", "baro_altitude_ft", "geom_altitude_ft")
        ],
    ).filter(
        pl.col("takeoff_time").dt.date() == target_date,
        pl.col("landing_time") > pl.col("takeoff_time"),
    )
    return with_flight_schema_columns(flights).sort("icao", "takeoff_time")


def create_flights_from_adsb(df: pl.DataFrame, target_date: date) -> pl.DataFrame:
    return flights_from_phases(label_phases(df), target_date)
