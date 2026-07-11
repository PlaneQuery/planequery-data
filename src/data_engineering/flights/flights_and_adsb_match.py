from data_engineering.adsb.read_adsb import read_adsb
from airports.airport_lookup import AirportLookup
import polars as pl
import math
from datetime import datetime

MAX_AIRPORT_DISTANCE_KM = 16

_airport_lookup = None

def _get_airport_lookup() -> AirportLookup:
    global _airport_lookup
    if _airport_lookup is None:
        _airport_lookup = AirportLookup()
    return _airport_lookup

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) *
         math.sin(delta_lambda / 2.0) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _icao_has_perfect_match(df_flights: pl.DataFrame, df_adsb: pl.DataFrame) -> bool:
    airport_lookup = _get_airport_lookup()
    df_flights = (
        df_flights
        .select(["takeoff_time", "landing_time", "takeoff_airport_ident", "landing_airport_ident"])
        .sort("takeoff_time")
    )
    df_adsb = (
        df_adsb
        .select(["time", "lat", "lon"])
        .with_row_index("adsb_row_ix")
    )

    if df_flights.is_empty() or df_adsb.is_empty():
        return False

    def get_airport_coords(ident: str):
        return airport_lookup.get_airport_coordinates(ident)

    takeoff_coords = df_flights["takeoff_airport_ident"].map_elements(get_airport_coords, return_dtype=pl.List(pl.Float64))
    landing_coords = df_flights["landing_airport_ident"].map_elements(get_airport_coords, return_dtype=pl.List(pl.Float64))

    if takeoff_coords.is_null().any() or landing_coords.is_null().any():
        return False

    df_flights = df_flights.with_columns([
        takeoff_coords.list.get(0).alias("takeoff_airport_lat"),
        takeoff_coords.list.get(1).alias("takeoff_airport_lon"),
        landing_coords.list.get(0).alias("landing_airport_lat"),
        landing_coords.list.get(1).alias("landing_airport_lon"),
    ])

    df_flights_check = df_flights.with_columns(
        (pl.col("takeoff_airport_ident") == pl.col("landing_airport_ident").shift(1))
        .fill_null(True).alias("takeoff_same_as_previous_landing")
    )
    if not df_flights_check.get_column("takeoff_same_as_previous_landing").all():
        return False

    df_adsb = df_adsb.with_columns([
        pl.lit(False).alias("in_flight"),
        pl.lit(False).alias("boundary"),
        pl.lit(None).cast(pl.Float64).alias("airport_lat"),
        pl.lit(None).cast(pl.Float64).alias("airport_lon"),
    ])

    for row in df_flights.iter_rows(named=True):
        takeoff_airport_coords = (row["takeoff_airport_lat"], row["takeoff_airport_lon"])
        landing_airport_coords = (row["landing_airport_lat"], row["landing_airport_lon"])
        takeoff_time = row["takeoff_time"]
        landing_time = row["landing_time"]
        df = df_adsb.filter(~pl.col("in_flight"))

        df_takeoff_match = df.filter(pl.col("time") >= takeoff_time).sort("time").head(1)
        df_landing_match = df.filter(pl.col("time") <= landing_time).sort("time", descending=True).head(1)

        if df_takeoff_match.is_empty() or df_landing_match.is_empty():
            return False

        df_takeoff_idx = df_takeoff_match.get_column("adsb_row_ix")[0]
        df_landing_idx = df_landing_match.get_column("adsb_row_ix")[0]

        if df_takeoff_idx == df_landing_idx:
            return False

        df_flight = df.filter((pl.col("adsb_row_ix") >= df_takeoff_idx) & (pl.col("adsb_row_ix") <= df_landing_idx))
        df_adsb = df_adsb.with_columns(
            pl.when(pl.col("adsb_row_ix").is_in(df_flight.get_column("adsb_row_ix").implode()))
            .then(pl.lit(True)).otherwise(pl.col("in_flight")).alias("in_flight")
        )
        df_adsb = df_adsb.with_columns(
            pl.when((pl.col("adsb_row_ix") == df_takeoff_idx) | (pl.col("adsb_row_ix") == df_landing_idx))
            .then(pl.lit(True)).otherwise(pl.col("boundary")).alias("boundary")
        )
        df_adsb = df_adsb.with_columns([
            pl.when(pl.col("adsb_row_ix") == df_takeoff_idx).then(pl.lit(takeoff_airport_coords[0]))
              .when(pl.col("adsb_row_ix") == df_landing_idx).then(pl.lit(landing_airport_coords[0]))
              .otherwise(pl.col("airport_lat")).alias("airport_lat"),
            pl.when(pl.col("adsb_row_ix") == df_takeoff_idx).then(pl.lit(takeoff_airport_coords[1]))
              .when(pl.col("adsb_row_ix") == df_landing_idx).then(pl.lit(landing_airport_coords[1]))
              .otherwise(pl.col("airport_lon")).alias("airport_lon"),
        ])

    b = pl.col("boundary")
    df_grounded = df_adsb.with_columns([
        pl.when(b).then(pl.col("time")).otherwise(None).backward_fill().alias("next_boundary_time"),
        pl.when(b).then(pl.col("time")).otherwise(None).forward_fill().alias("previous_boundary_time"),
        pl.when(b).then(pl.col("airport_lat")).otherwise(None).backward_fill().alias("next_airport_lat"),
        pl.when(b).then(pl.col("airport_lon")).otherwise(None).backward_fill().alias("next_airport_lon"),
        pl.when(b).then(pl.col("airport_lat")).otherwise(None).forward_fill().alias("previous_airport_lat"),
        pl.when(b).then(pl.col("airport_lon")).otherwise(None).forward_fill().alias("previous_airport_lon"),
    ])
    df_grounded = df_grounded.filter(~b & ~pl.col("in_flight"))
    df_grounded = df_grounded.with_columns(
        (pl.col("time") - pl.col("previous_boundary_time")).alias("time_from_previous_boundary"),
        (pl.col("next_boundary_time") - pl.col("time")).alias("time_to_next_boundary"),
    )
    use_prev = pl.col("time_from_previous_boundary") <= pl.col("time_to_next_boundary")
    df_grounded = df_grounded.with_columns(
        pl.when(pl.col("previous_airport_lat").is_null()).then(pl.col("next_airport_lat"))
          .when(pl.col("next_airport_lat").is_null()).then(pl.col("previous_airport_lat"))
          .when(use_prev).then(pl.col("previous_airport_lat"))
          .otherwise(pl.col("next_airport_lat")).alias("closest_airport_lat"),
        pl.when(pl.col("previous_airport_lon").is_null()).then(pl.col("next_airport_lon"))
          .when(pl.col("next_airport_lon").is_null()).then(pl.col("previous_airport_lon"))
          .when(use_prev).then(pl.col("previous_airport_lon"))
          .otherwise(pl.col("next_airport_lon")).alias("closest_airport_lon"),
    )
    df_grounded = df_grounded.with_columns(
        pl.struct(["lat", "lon", "closest_airport_lat", "closest_airport_lon"])
        .map_elements(
            lambda r: haversine(r["lat"], r["lon"], r["closest_airport_lat"], r["closest_airport_lon"])
            if all(v is not None for v in [r["lat"], r["lon"], r["closest_airport_lat"], r["closest_airport_lon"]])
            else None,
            return_dtype=pl.Float64,
        )
        .alias("distance_from_airport_km")
    )

    return not df_grounded.get_column("distance_from_airport_km").gt(MAX_AIRPORT_DISTANCE_KM).any()


def get_icaos_with_perfect_match(df_flights_full: pl.DataFrame, df_adsb_full: pl.DataFrame) -> set[str]:
    perfect = set()
    for icao in df_flights_full.get_column("icao").unique():
        df_flights = df_flights_full.filter(pl.col("icao") == icao)
        df_adsb = df_adsb_full.filter(pl.col("icao") == icao)
        if _icao_has_perfect_match(df_flights, df_adsb):
            perfect.add(icao)
    return perfect
