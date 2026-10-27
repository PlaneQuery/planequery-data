"""Adapt canonical observations to the unmodified Traffic.iterate() defaults."""

from datetime import date
from importlib.metadata import version

import polars as pl

from flights.flight_type import FLIGHT_POLARS_SCHEMA, with_flight_schema_columns


TRAFFIC_VERSION = "2.13"
ADSB_COLUMNS = (
    "icao", "time", "lat", "lon", "callsign", "registration",
    "baro_altitude_ft", "geom_altitude_ft",
)


def _traffic_class():
    # Registry/CLI imports must work without this optional, substantial package.
    try:
        from traffic.core import Traffic
    except ImportError as exc:
        raise ImportError(
            'Traffic segmentation requires pip install -e ".[traffic]"'
        ) from exc
    if version("traffic") != TRAFFIC_VERSION:
        raise RuntimeError(
            f"This flight-source version requires traffic=={TRAFFIC_VERSION}; "
            "changing the iterator requires a flight-source version bump."
        )
    return Traffic


def create_flights_from_adsb(df: pl.DataFrame, target_date: date) -> pl.DataFrame:
    """Return upstream trajectories whose first observation is on target_date.

    Traffic groups by (icao24, callsign), then splits at gaps > 10 minutes.
    Its own null-callsign and singleton handling is preserved. Ground messages
    are retained: endpoints are observation bounds, not detected liftoff/touchdown.
    """
    if df.is_empty():
        return pl.DataFrame(schema=FLIGHT_POLARS_SCHEMA)
    observations = df.select(ADSB_COLUMNS).drop_nulls(["icao", "time"])
    if observations.is_empty():
        return pl.DataFrame(schema=FLIGHT_POLARS_SCHEMA)
    Traffic = _traffic_class()
    # Canonical naive timestamps represent UTC. convert_time_zone also handles
    # explicitly zoned inputs, keeping target-day ownership in UTC.
    observations = observations.with_columns(pl.col("time").dt.convert_time_zone("UTC"))
    data = observations.rename({"icao": "icao24", "time": "timestamp"}).to_pandas()
    records = []
    for flight in Traffic(data).iterate():
        first = flight.data.iloc[0]
        last = flight.data.iloc[-1]
        start = first.timestamp.to_pydatetime().replace(tzinfo=None)
        stop = last.timestamp.to_pydatetime().replace(tzinfo=None)
        if start.date() != target_date:
            continue
        registrations = flight.data.registration.dropna()
        record = {
            "icao": first.icao24,
            "callsign": first.callsign,
            "registration": registrations.iloc[0] if len(registrations) else "",
            "takeoff_time": start,
            "landing_time": stop,
            "first_message_time": start,
            "last_message_time": stop,
        }
        for endpoint, row in (("first", first), ("last", last)):
            for column in ("lat", "lon", "baro_altitude_ft", "geom_altitude_ft"):
                record[f"{endpoint}_{column}"] = row[column]
        records.append(record)
    if not records:
        return pl.DataFrame(schema=FLIGHT_POLARS_SCHEMA)
    # pandas represents missing numeric observations as NaN; restore nulls
    # before converting nullable altitude columns back to canonical integers.
    flights = pl.from_dicts(records, infer_schema_length=None).with_columns(
        pl.col(pl.Float64).fill_nan(None)
    )
    return with_flight_schema_columns(flights).sort("icao", "takeoff_time", "callsign")
