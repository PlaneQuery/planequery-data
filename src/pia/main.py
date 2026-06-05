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

    flights_in_window = df_flights.filter(pl.col("takeoff_time") >= start_datetime)
    prior_anchor_flights = (
        df_flights
        .filter(
            (pl.col("ladd"))
            & (~pl.col("pia"))
            & (pl.col("takeoff_time") <= start_datetime)
        )
        .sort(["icao", "takeoff_time"], descending=[False, True])
        .unique(subset=["icao"], keep="first")
    )

    return (
        pl.concat([prior_anchor_flights, flights_in_window])
        .unique(subset=flight_key, keep="first")
        .sort(flight_key)
    )

def main(start_date: date, end_date: date):
    '''
    start_date inclusive
    end_date exclusive
    '''
    canidate_start_date = max(date(2024,1,1), start_date - timedelta(days=365))
    df_flights = get_flights(canidate_start_date, end_date)
    df_flights = df_flights.filter(pl.col("ladd") | pl.col("pia"))
    ladd_pia_candidates = get_ladd_pia_candidates(df_flights)
    print(f"{len(ladd_pia_candidates)} ladd pia candidates")
    df_flights = df_flights.filter(pl.col("icao").is_in(ladd_pia_candidates) | pl.col("pia"))
    df_flights = keep_start_window_with_prior_anchor(df_flights, start_date)
    df_flights_2 = get_flights(end_date, end_date + timedelta(days=10))
    df_flights = pl.concat([df_flights, df_flights_2])
    flights = flights_df_to_flights(df_flights)
    mappings,g = resolve_pia_mappings(flights)
    print(mappings)

if __name__ == "__main__":
    start_date = date(2025,5,20)
    end_date = date(2025,5,21)
    main(start_date, end_date)
