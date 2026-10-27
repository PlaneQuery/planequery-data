"""Per-ICAO time-only DBSCAN extraction from Sun et al. (2017).

Reference: https://github.com/junzis/flight-data-processor/blob/12ed7b73d202b1d3c836323026725120501a40e5/flightextract.py
Sun, Ellerbroek and Hoekstra (2017), doi:10.2514/1.I010520.

The upstream extractor zeroes every feature except normalized time before
clustering. Its scaled radius is equivalent to 1,800 seconds, with 100 samples
per core neighborhood and strictly more than 100 observations per saved cluster.
Deduplication, scaling, density checks, and clustering operate independently
per ICAO. The upstream input-order DBSCAN traversal and strict output-size
threshold are preserved, without its cross-aircraft preprocessing.
"""

import math
from datetime import date

import numpy as np
import polars as pl
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

from airports.catalog import AirportCatalogView
from flights.flight_type import FLIGHT_POLARS_SCHEMA
from flights.segmentation import algorithm as shared


GAP_HOURS = 0.5
MIN_SAMPLES = 100
ADSB_COLUMNS = tuple(
    column for column in shared.ADSB_ALGORITHM_COLUMNS
    if column not in {"ground_speed_kt", "track_deg", "on_ground"}
)


def _endpoint_fields(
    message: shared.AdsbPositionMessage,
    endpoint: str,
    position: str,
    airport_catalog: AirportCatalogView,
) -> dict[str, object]:
    match = None
    if (
        message.lat is not None and message.lon is not None
        and math.isfinite(message.lat) and math.isfinite(message.lon)
        and -90 <= message.lat <= 90 and -180 <= message.lon <= 180
    ):
        match = airport_catalog.matcher.nearest(message.lat, message.lon)
    return {
        f"{endpoint}_time": message.time,
        f"{endpoint}_airport_id": match.id if match else None,
        f"{endpoint}_airport_ident": match.ident if match else None,
        f"{position}_message_time": message.time,
        f"{position}_lat": message.lat,
        f"{position}_lon": message.lon,
        f"{position}_baro_altitude_ft": shared._baro_altitude_int_ft(message),
        f"{position}_geom_altitude_ft": shared._geom_altitude_ft(message),
    }


def create_flights_from_adsb(
    df_adsb: pl.DataFrame,
    target_date: date,
    airport_catalog: AirportCatalogView,
) -> pl.DataFrame:
    """Extract continuous paths, partition by first observation, assign airports.

    First/last cluster observations supply the nominal takeoff/landing times.
    No airborne, duration, distance, or airport filter is applied by upstream.
    Each aircraft is processed independently, including timestamp deduplication
    and scaling. No observation of another aircraft can affect its flights.
    Missing positions do not change clusters; their airport fields stay null.
    """
    if df_adsb.is_empty():
        return pl.DataFrame(schema=FLIGHT_POLARS_SCHEMA)
    observations = df_adsb.filter(
        pl.col("time").is_not_null() & pl.col("icao").is_not_null()
    )
    day_start, day_end = shared._day_bounds(target_date)
    rows = []
    for _, aircraft in observations.group_by("icao", maintain_order=True):
        aircraft = aircraft.unique(
            subset=["time"], keep="first", maintain_order=True
        )
        if aircraft.height <= MIN_SAMPLES:
            continue
        timestamps = aircraft.get_column("time").dt.epoch("us").to_numpy() / 1_000_000
        scaler = MinMaxScaler(feature_range=(0, 1000))
        scaled = scaler.fit_transform(timestamps.reshape(-1, 1)).ravel()
        # Retain upstream's operation order, including floating-point rounding.
        radius = float(scaler.scale_[0] * GAP_HOURS * 60 * 60)
        # Upstream zeroes columns 1..8 before fitting. Keep the same shape
        # and row order, including order-dependent shared border points.
        features = np.zeros((aircraft.height, 9))
        features[:, 0] = scaled
        labels = DBSCAN(eps=radius, min_samples=MIN_SAMPLES).fit_predict(features)
        for label in np.unique(labels):
            if label == -1:
                continue
            cluster = aircraft.filter(pl.Series(labels == label)).sort("time")
            if cluster.height <= MIN_SAMPLES:
                continue
            if not day_start <= cluster[0, "time"] < day_end:
                continue
            messages = shared._rows_to_position_messages(cluster)
            flags, identity, metadata = shared._segment_counts_for_messages(
                messages
            )
            flight_flags = shared._db_flags_to_flight_flags(
                shared._db_flags_mode_from_counts(flags)
            )
            rows.append(
                {
                    "icao": messages[0].icao,
                    **shared._flight_identity_mode_from_counts(identity),
                    **shared._aircraft_metadata_mode_from_counts(metadata),
                    **flight_flags,
                    **_endpoint_fields(
                        messages[0], "takeoff", "first", airport_catalog
                    ),
                    **_endpoint_fields(
                        messages[-1], "landing", "last", airport_catalog
                    ),
                }
            )
    return pl.DataFrame(rows, schema=FLIGHT_POLARS_SCHEMA)
