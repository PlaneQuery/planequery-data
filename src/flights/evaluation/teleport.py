# in tests/ have examples of true teleports.
# This has function that detects teleports.
import polars as pl

def detect_teleports(df_flights: pl.DataFrame) -> pl.DataFrame:
    TELEPORT_MIN_DISTANCE_KM = 50.0
    EARTH_RADIUS_KM = 6371.0088

    prev_lat = pl.col("prev_last_lat").radians()
    prev_lon = pl.col("prev_last_lon").radians()
    curr_lat = pl.col("first_lat").radians()
    curr_lon = pl.col("first_lon").radians()
    haversine_a = (
        ((curr_lat - prev_lat) / 2).sin().pow(2)
        + prev_lat.cos() * curr_lat.cos() * ((curr_lon - prev_lon) / 2).sin().pow(2)
    )

    teleport_flights = (
        df_flights
        .sort(["icao", "takeoff_time", "landing_time"])
        .with_columns(
            pl.col("landing_time").shift(1).over("icao").alias("prev_landing_time"),
            pl.col("landing_airport_ident").shift(1).over("icao").alias("prev_landing_airport_ident"),
            pl.col("last_lat").shift(1).over("icao").alias("prev_last_lat"),
            pl.col("last_lon").shift(1).over("icao").alias("prev_last_lon"),
        )
        .filter(
            pl.all_horizontal(
                pl.col("prev_last_lat").is_not_null(),
                pl.col("prev_last_lon").is_not_null(),
                pl.col("first_lat").is_not_null(),
                pl.col("first_lon").is_not_null(),
            )
        )
        .filter(
            pl.col("prev_landing_airport_ident").is_null()
            | pl.col("takeoff_airport_ident").is_null()
            | (pl.col("prev_landing_airport_ident") != pl.col("takeoff_airport_ident"))
        )
        .with_columns(
            (2 * EARTH_RADIUS_KM * haversine_a.sqrt().clip(0, 1).arcsin()).alias("endpoint_jump_km"),
            ((pl.col("takeoff_time") - pl.col("prev_landing_time")).dt.total_seconds() / 3600).alias("time_gap_hours"),
        )
        .with_columns(
            (pl.col("endpoint_jump_km") / pl.col("time_gap_hours")).alias("required_ground_speed_kmh")
        )
        .filter(pl.col("endpoint_jump_km") >= TELEPORT_MIN_DISTANCE_KM)
        .select(
            "icao",
            "prev_landing_time",
            "takeoff_time",
            "time_gap_hours",
            "endpoint_jump_km",
            "required_ground_speed_kmh",
            "prev_landing_airport_ident",
            "takeoff_airport_ident",
            "prev_last_lat",
            "prev_last_lon",
            "first_lat",
            "first_lon",
            "pia",
            "ladd",
            "aircraft_type",
            "owner",
        )
        .sort("endpoint_jump_km", descending=True)
    )

    return teleport_flights