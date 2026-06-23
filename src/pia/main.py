# get flights that fullfill constraints that can be put straight into resolve.
# Also, get candiate icaos

from datetime import date, datetime, time, timedelta
from pathlib import Path
import polars as pl

from data_engineering.flights.flight_type import flights_df_to_flights, get_flights
from pia.ladd_aircraft_pia_candidates import get_ladd_pia_candidates
from pia.resolver import get_canidate_icaos, resolve_pia_mappings

def keep_start_window_with_prior_anchor(df_flights: pl.DataFrame, start_date: date) -> pl.DataFrame:
    start_datetime = datetime.combine(start_date, time.min)
    flight_key = ["icao", "takeoff_time"]

    flights_in_window = df_flights.filter(pl.col("landing_time") >= start_datetime)
    prior_anchor_flights = (
        df_flights
        .filter(
            (pl.col("ladd"))
            & (~pl.col("pia"))
            & (pl.col("landing_time") <= start_datetime)
        )
        .sort(["icao", "takeoff_time"], descending=[False, True])
        .unique(subset=["icao"], keep="first")
    )

    return (
        pl.concat([prior_anchor_flights, flights_in_window])
        .unique(subset=flight_key, keep="first")
        .sort(flight_key)
    )

def run_pia_algorithm(start_date: date, end_date: date, icaos: list[str] | None = None):
    '''
    start_date inclusive
    end_date exclusive
    '''
    # canidate_start_date = max(date(2024,1,1), start_date - timedelta(days=365))
    canidate_start_date = start_date
    df_flights = get_flights(canidate_start_date, end_date)
    if icaos:
        df_flights = df_flights.filter(pl.col("icao").is_in(icaos))
    else:
        df_flights = df_flights.filter(pl.col("ladd") | pl.col("pia"))
    print("read df flights")
    ladd_pia_candidates = get_ladd_pia_candidates(df_flights)
    # print(f"{len(ladd_pia_candidates)} ladd pia candidates")
    df_flights = df_flights.filter(pl.col("icao").is_in(ladd_pia_candidates) | pl.col("pia"))
    df_flights = keep_start_window_with_prior_anchor(df_flights, start_date)
    df_flights = fix_route_mismatches(df_flights)
    # df_flights_2 = get_flights(end_date, end_date + timedelta(days=1)) # We proboably don't need most of this just pree ladd flight is what's required.
    # df_flights_2 = df_flights_2.filter(pl.col("icao").is_in(ladd_pia_candidates) | pl.col("pia"))
    # df_flights = pl.concat([df_flights, df_flights_2]) # We still have pia source nodes present. 
    flights = flights_df_to_flights(df_flights)
    print("start resolver")
    mappings,g = resolve_pia_mappings(flights)
    return mappings,g

if __name__ == "__main__":
    start_date = date(2025,5,20)
    end_date = date(2025,5,21)
    run_pia_algorithm(start_date, end_date)

def fix_route_mismatches(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df
        .with_row_index("_route_fix_row_nr")
        .sort(["icao", "takeoff_time"])
        .with_columns(
            pl.col("takeoff_airport_ident")
            .shift(-1)
            .over("icao")
            .alias("_next_takeoff_airport_ident")
        )
        .with_columns(
            pl.when(
                pl.col("_next_takeoff_airport_ident").is_not_null()
                & (pl.col("landing_airport_ident") != pl.col("_next_takeoff_airport_ident"))
            )
            .then(pl.col("_next_takeoff_airport_ident"))
            .otherwise(pl.col("landing_airport_ident"))
            .alias("landing_airport_ident")
        )
        .sort("_route_fix_row_nr")
        .drop(["_route_fix_row_nr", "_next_takeoff_airport_ident"])
    )

