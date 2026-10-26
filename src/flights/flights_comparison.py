from datetime import timedelta

import polars as pl

from airports.catalog import AirportCatalogView
from flights.flight_type import (
    Flight,
    add_lat_lon_to_airport_columns,
    flights_df_to_flights,
)
from planequery_data.utils import haversine


COMPARISON_ENDPOINT_SCHEMA = {
    "first_message_time": pl.Datetime("ms"),
    "first_lat": pl.Float64,
    "first_lon": pl.Float64,
    "last_message_time": pl.Datetime("ms"),
    "last_lat": pl.Float64,
    "last_lon": pl.Float64,
}


def _with_comparison_endpoint_columns(df: pl.DataFrame) -> pl.DataFrame:
    missing_columns = [
        pl.lit(None, dtype=dtype).alias(column_name)
        for column_name, dtype in COMPARISON_ENDPOINT_SCHEMA.items()
        if column_name not in df.columns
    ]
    if missing_columns:
        return df.with_columns(missing_columns)
    return df

def _haversine_expr(
    lat_col: str,
    lon_col: str,
    airport_lat_col: str,
    airport_lon_col: str,
) -> pl.Expr:
    cols = [lat_col, lon_col, airport_lat_col, airport_lon_col]
    return pl.struct(cols).map_elements(
        lambda row: haversine(
            row[lat_col],
            row[lon_col],
            row[airport_lat_col],
            row[airport_lon_col],
        )
        if all(row[col] is not None for col in cols)
        else None,
        return_dtype=pl.Float64,
    )


MAX_FLIGHT_MATCH_TIME_DELTA = timedelta(hours=1)


def compare_flight(
    f0: Flight,
    f1: Flight,
    max_flight_match_time_delta: timedelta = MAX_FLIGHT_MATCH_TIME_DELTA,
) -> bool:
    if (
        abs(f0.takeoff_time - f1.takeoff_time) > max_flight_match_time_delta
        or abs(f0.landing_time - f1.landing_time) > max_flight_match_time_delta
    ):
        return False
    return True


def flights_comparison(
    flights_0: list[Flight],
    flights_1: list[Flight],
):
    """Compare flight lists assigned to the same UTC takeoff day."""
    f0_series: list[int | None] = [None] * len(flights_0)

    j1_start = 0
    
    for i in range(len(flights_0)):
        f0 = flights_0[i]
        for j in range(j1_start, len(flights_1)):
            f1 = flights_1[j]
            does_match = compare_flight(f0, f1)
            if does_match:
                f0_series[i] = j
                j1_start = j + 1
                break

    return f0_series


def df_flights_comparison(
    df_0: pl.DataFrame,
    df_1: pl.DataFrame,
    start_dt,
    end_dt,
    airport_catalog: AirportCatalogView,
):
    df_0 = _with_comparison_endpoint_columns(df_0)
    df_1 = add_lat_lon_to_airport_columns(
        _with_comparison_endpoint_columns(df_1),
        airport_catalog,
    )
    df_0_by_icao = df_0.sort(["icao", "takeoff_time"]).partition_by(
        "icao",
        as_dict=True,
    )
    df_1_by_icao = df_1.sort(["icao", "takeoff_time"]).partition_by(
        "icao",
        as_dict=True,
    )
    icaos = sorted(
        set(df_0.get_column("icao").unique().to_list())
        | set(df_1.get_column("icao").unique().to_list())
    )
    dfs = []
    for icao in icaos:
        key = (icao,)
        df0 = df_0_by_icao.get(key, df_0.clear())
        df1 = df_1_by_icao.get(key, df_1.clear())
        flights_0 = flights_df_to_flights(df0)
        flights_1 = flights_df_to_flights(df1)
        match_0 = flights_comparison(
            flights_0,
            flights_1,
            start_dt,
            end_dt,
        )
        df0 = df0.with_row_index("idx")

        row_map_1: list[int | None] = [None] * len(flights_1)
        for df0_index, df1_index in enumerate(match_0):
            if isinstance(df1_index, int):
                row_map_1[df1_index] = df0_index
        df1 = df1.with_columns(
            pl.Series("row_map", row_map_1, dtype=pl.UInt32),
        )
        df = df0.join(
            df1,
            left_on="idx",
            right_on="row_map",
            how="full",
            suffix="_df1"
        )
        df = df.rename(
            {
                "takeoff_airport_lat": "takeoff_airport_lat_df1",
                "takeoff_airport_lon": "takeoff_airport_lon_df1",
                "landing_airport_lat": "landing_airport_lat_df1",
                "landing_airport_lon": "landing_airport_lon_df1",
            }
        )
        df = df.with_columns(
            pl.when(pl.col("row_map").is_not_null())
            .then(pl.lit("both"))
            .when(pl.col("idx").is_not_null())
            .then(pl.lit("df0_only"))
            .otherwise(pl.lit("df1_only"))
            .alias("match_status")
        )
        df = df.drop(["idx", "row_map"])
        df = df.with_columns(pl.lit(icao).alias("icao"))

        dfs.append(df)

    df = pl.concat(dfs)
    df = df.with_columns(
        pl.when(
            (pl.col("match_status") == "both")
            & pl.col("takeoff_airport_id").is_not_null()
            & pl.col("takeoff_airport_id_df1").is_not_null()
            & (pl.col("takeoff_airport_id") == pl.col("takeoff_airport_id_df1"))
        )
        .then(True)
        .otherwise(False)
        .alias("same_takeoff_airport_id")
    )
    df = df.with_columns(
        pl.when(
            (pl.col("match_status") == "both")
            & pl.col("landing_airport_id").is_not_null()
            & pl.col("landing_airport_id_df1").is_not_null()
            & (pl.col("landing_airport_id") == pl.col("landing_airport_id_df1"))
        )
        .then(True)
        .otherwise(False)
        .alias("same_landing_airport_id")
    )
    df = df.with_columns(
        pl.when(
            (pl.col("match_status") == "both")
            & (
                pl.col("same_takeoff_airport_id")
                & pl.col("same_landing_airport_id")
            )
        )
        .then(True)
        .otherwise(False)
        .alias("same_airport_id")
    )

    # Compare observed ADS-B endpoints with the gold airports.
    expr_0 = pl.col("first_message_time") - pl.col("takeoff_time_df1")
    expr_1 = pl.col("last_message_time") - pl.col("landing_time_df1")
    df = df.with_columns(
        pl.when(pl.col("match_status") == "both")
        .then(expr_0)
        .otherwise(None)
        .alias("first_message_time_diff"),
        pl.when(pl.col("match_status") == "both")
        .then(expr_1)
        .otherwise(None)
        .alias("last_message_time_diff"),
    )

    df = df.with_columns(
        pl.when(pl.col("match_status") == "both")
        .then(
            _haversine_expr(
                "first_lat",
                "first_lon",
                "takeoff_airport_lat_df1",
                "takeoff_airport_lon_df1",
            )
        )
        .otherwise(None)
        .alias("first_message_distance_to_takeoff_airport_km"),
        pl.when(pl.col("match_status") == "both")
        .then(
            _haversine_expr(
                "last_lat",
                "last_lon",
                "landing_airport_lat_df1",
                "landing_airport_lon_df1",
            )
        )
        .otherwise(None)
        .alias("last_message_distance_to_landing_airport_km"),
    )
    return df
