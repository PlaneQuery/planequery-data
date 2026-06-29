import math
import polars as pl
from airports.airportLookup import AirportLookup

AIRPORT_RADIUS_KM = 50.0
airport_lookup = None

TAKEOFF_AIRPORT_CANDIDATE_DTYPE = pl.List(
    pl.Struct(
        {
            "ident": pl.Utf8,
            "distance_km": pl.Float64,
            "elevation_ft": pl.Int64,
            "first_baro_altitude_minus_elevation_ft": pl.Float64,
        }
    )
)

def takeoff_airport_candidates(row, radius_km=AIRPORT_RADIUS_KM):
    global airport_lookup
    if airport_lookup is None:
        airport_lookup = AirportLookup()

    lat = row["lat"]
    lon = row["lon"]
    first_baro_altitude_ft = row["first_baro_altitude_ft"]
    if lat is None or lon is None or math.isnan(lat) or math.isnan(lon):
        return []

    lat = float(lat)
    lon = float(lon)
    airports = airport_lookup.getAirportsWithinRadius(lat, lon, radius_km=radius_km)

    if first_baro_altitude_ft is None or math.isnan(first_baro_altitude_ft):
        first_baro_altitude_ft = None
    else:
        first_baro_altitude_ft = float(first_baro_altitude_ft)

    return [
        {
            "ident": airport.ident,
            "distance_km": airport.distance_from_plane_point_in_km,
            "elevation_ft": airport.elevation_ft,
            "first_baro_altitude_minus_elevation_ft": (
                None
                if first_baro_altitude_ft is None
                else first_baro_altitude_ft - airport.elevation_ft
            ),
        }
        for airport in airports
    ]

def airport_ident_in_candidates(row):
    ident = row["ident"]
    candidates = row["candidates"]
    if ident is None or candidates is None:
        return False
    return ident in {candidate["ident"] for candidate in candidates}

def add_takeoff_airport_candidates(
    df: pl.DataFrame,
    radius_km: float = AIRPORT_RADIUS_KM,
) -> pl.DataFrame:
    return df.with_columns(
        pl.struct(
            [
                pl.col("first_lat").alias("lat"),
                pl.col("first_lon").alias("lon"),
                pl.col("first_baro_altitude_ft"),
            ]
        )
        .map_elements(
            lambda row: takeoff_airport_candidates(row, radius_km=radius_km),
            return_dtype=TAKEOFF_AIRPORT_CANDIDATE_DTYPE,
        )
        .alias("takeoff_airports")
    ).with_columns(
        pl.col("takeoff_airports")
        .list.len()
        .alias("takeoff_airports_count"),
        pl.struct(
            [
                pl.col("takeoff_airport_ident_df1").alias("ident"),
                pl.col("takeoff_airports").alias("candidates"),
            ]
        )
        .map_elements(airport_ident_in_candidates, return_dtype=pl.Boolean)
        .alias("takeoff_airport_ident_df1_in_takeoff_airports"),
    )

def explode_takeoff_airports(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.explode("takeoff_airports")
        .filter(pl.col("takeoff_airports").is_not_null())
        .unnest("takeoff_airports")
    )
