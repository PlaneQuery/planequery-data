
from datetime import timedelta
from enum import Enum, unique
from data_engineering.flights.flight_type import Flight, add_lat_lon_to_airport_ident_columns, flights_df_to_flights
import polars as pl
from itertools import zip_longest
from utils import haversine


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


class Match(Enum):
    FALSE = 0
    TRUE = 1
    DISCARD = 2
# do I want to have seperate rejections for takeoff/landing or airport ident and maybe I can pick a particular one? 
# maybe I can return a list like [True, False, True, False] "matched", "matched", "airport_ident x != y",
MAX_TIME_DIFF = timedelta(hours=1)
def compare_flight(f0: Flight, f1: Flight, compare_airports = True):
    if abs(f0.takeoff_time - f1.takeoff_time) > MAX_TIME_DIFF or abs(f0.landing_time - f1.landing_time) > MAX_TIME_DIFF:
        return False
    if compare_airports:
        if f0.takeoff_airport_ident != f1.takeoff_airport_ident or f0.landing_airport_ident != f1.landing_airport_ident:
            return False
    return True

def flights_comparison(flights_0: list[Flight], flights_1: list[Flight], start_dt, end_dt, grace_period = timedelta(minutes=30), compare_airports= True):
    '''
    we assume flights are assigned to days by takeoff_time. 
    '''
    f0_series: list[int | str | None] = [None] * len(flights_0)
    f1_series: list[int | str | None] = [None] * len(flights_1)

    i1_start = 0

    for i0, f0 in enumerate(flights_0):
        for j in range(i1_start, len(flights_1)):
            if f1_series[j] is not None:
                continue

            if compare_flight(f0, flights_1[j], compare_airports):
                f0_series[i0] = j
                f1_series[j] = i0
                i1_start = j + 1
                break

    if f0_series and f0_series[0] is None and flights_0[0].takeoff_time - start_dt <= grace_period:
        f0_series[0] = "DISCARD"

    if f1_series and f1_series[0] is None and flights_1[0].takeoff_time - start_dt <= grace_period:
        f1_series[0] = "DISCARD"

    if f0_series and f0_series[-1] is None and end_dt - flights_0[-1].takeoff_time <= grace_period:
        f0_series[-1] = "DISCARD"

    if f1_series and f1_series[-1] is None and (end_dt - flights_1[-1].takeoff_time <= grace_period or flights_1[-1].landing_time > end_dt): # TODO: landing time is temporary since current ADSBX data is only for 1 day. 
        f1_series[-1] = "DISCARD"

    return (f0_series, f1_series)

# we return the maps of what they map to and then.... we create both, df0, df1. in the dataframe I'll keep the original order those rows will just haven null for i.e the df1 columns
def df_flights_comparision(df_0: pl.DataFrame, df_1: pl.DataFrame, start_dt, end_dt, grace_period = timedelta(minutes=30), compare_airports= True): # TODO: Add the distnace comaprision between the last_lat, last_lon and last_message_time of df_1 and the true SFDPS data. 
    df_0 = _with_comparison_endpoint_columns(df_0)
    df_1 = _with_comparison_endpoint_columns(df_1)
    icaos = sorted(set(df_0.get_column("icao").unique().to_list()) | set(df_1.get_column("icao").unique().to_list()))
    dfs = []
    for icao in icaos:
        df0 = df_0.filter(pl.col("icao") == icao)
        df1 = df_1.filter(pl.col("icao") == icao)
        df0 = df0.sort("takeoff_time")
        df1 = df1.sort("takeoff_time")
        flights_0 = flights_df_to_flights(df0)
        flights_1 = flights_df_to_flights(df1)
        match_0, match_1 = flights_comparison(flights_0, flights_1, start_dt, end_dt, grace_period, compare_airports)
        discard_0 = [x == "DISCARD" for x in match_0]

        df0 = (
            df0.with_row_index("idx")
            .with_columns(
                pl.Series("discard", discard_0, dtype=pl.Boolean),
            )
            .filter(~pl.col("discard"))
        )

        row_map_1 = [None if x == "DISCARD" else x for x in match_1]
        discard_1 = [x == "DISCARD" for x in match_1]
        df1 = (df1
            .with_columns(
                pl.Series("row_map", row_map_1, dtype=pl.UInt32),
                pl.Series("discard", discard_1, dtype=pl.Boolean),
            )
            .filter(~pl.col("discard"))
        )
        df1 = add_lat_lon_to_airport_ident_columns(df1)
        df = df0.join(
            df1,
            left_on="idx",
            right_on="row_map",
            how="full",
            suffix="_df1"
        )
        df = df.rename({"takeoff_airport_lat": "takeoff_airport_lat_df1", "takeoff_airport_lon": "takeoff_airport_lon_df1", "landing_airport_lat": "landing_airport_lat_df1", "landing_airport_lon": "landing_airport_lon_df1"})
        df = df.with_columns(
            pl.when(pl.col("row_map").is_not_null())
            .then(pl.lit("both"))
            .when(pl.col("idx").is_not_null())
            .then(pl.lit("df0_only"))
            .otherwise(pl.lit("df1_only"))
            .alias("match_status")
        )   
        df = df.drop(["idx", "row_map", "discard"])
        df = df.with_columns(pl.lit(icao).alias("icao"))

        dfs.append(df)

    df = pl.concat(dfs)
    df = df.with_columns(
    pl.when(
        (pl.col("match_status") == "both")
        & (pl.col("takeoff_airport_ident") == pl.col("takeoff_airport_ident_df1"))
    )
    .then(True)
    .otherwise(False)
    .alias("same_takeoff_airport_ident")
    )
    df = df.with_columns(
    pl.when(
        (pl.col("match_status") == "both")
        & (pl.col("landing_airport_ident") == pl.col("landing_airport_ident_df1"))
    )
    .then(True)
    .otherwise(False)
    .alias("same_landing_airport_ident")
    )
    df = df.with_columns(
    pl.when(
        (pl.col("match_status") == "both")
        & (pl.col("same_takeoff_airport_ident") & pl.col("same_landing_airport_ident"))
    )
    .then(True)
    .otherwise(False)
    .alias("same_airport_ident")
    )

        # last/first message diffrence and takeoff/landing airport ident distance from last/first lat/lon from df1 (gold) 
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

def df_flights_comparison_stats(df: pl.DataFrame) -> dict[str, int | float]:
    true_positive = df.filter(pl.col("match_status") == "both").height
    false_positive = df.filter(pl.col("match_status") == "df0_only").height
    false_negative = df.filter(pl.col("match_status") == "df1_only").height

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative

    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    takeoff_airport_ident_match = df.filter(pl.col("same_takeoff_airport_ident")).height
    landing_airport_ident_match = df.filter(pl.col("same_landing_airport_ident")).height
    airport_ident_match = df.filter(pl.col("same_airport_ident")).height
    takeoff_airport_ident_match_pct = takeoff_airport_ident_match / true_positive if true_positive else 0.0
    landing_airport_ident_match_pct = landing_airport_ident_match / true_positive if true_positive else 0.0
    airport_ident_match_pct = airport_ident_match / true_positive if true_positive else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        # Keep the underlying counts alongside the rates so evaluation output
        # shows how many matched flights had the correct airport assignment.
        "takeoff_airport_ident_total_count": true_positive,
        "takeoff_airport_ident_match_count": takeoff_airport_ident_match,
        "takeoff_airport_ident_incorrect_match_count": (
            true_positive - takeoff_airport_ident_match
        ),
        "takeoff_airport_ident_match_pct": takeoff_airport_ident_match_pct,
        "landing_airport_ident_total_count": true_positive,
        "landing_airport_ident_match_count": landing_airport_ident_match,
        "landing_airport_ident_incorrect_match_count": (
            true_positive - landing_airport_ident_match
        ),
        "landing_airport_ident_match_pct": landing_airport_ident_match_pct,
        "airport_ident_total_count": true_positive,
        "airport_ident_match_count": airport_ident_match,
        "airport_ident_incorrect_match_count": (
            true_positive - airport_ident_match
        ),
        "airport_ident_match_pct": airport_ident_match_pct,
    }
