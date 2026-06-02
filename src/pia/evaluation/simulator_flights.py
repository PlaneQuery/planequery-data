# Assume flights are perfect. I'm going to take from my flights table and see what percentage of pia flights can be identified.
from datetime import date, timedelta
import polars as pl
from data_engineering.adsb.read_adsb import read_adsb, read_icaos_from_adsb
from data_engineering.flights.flight_type import get_flights, flights_df_to_flights
from pia.simulator.main import simulate_pia_flights
from data_engineering.flights.flights_and_adsb_match import get_icaos_with_perfect_match


def get_simulated_flights_0():
    start_date = date(2026, 2, 28)
    end_date = date(2026, 3, 1)

    flight_dfs = []
    target_date = start_date
    while target_date < end_date:
        flight_dfs.append(get_flights(target_date))
        target_date += timedelta(days=1)

    df_flights = pl.concat(flight_dfs)
    df_flights = df_flights.filter(pl.col("ladd"))
    icaos = df_flights.get_column("icao").unique().to_list()
    df_adsbs = []
    target_date = start_date
    while target_date < end_date:
        df_adsb = read_icaos_from_adsb(target_date, icaos)
        df_adsbs.append(df_adsb)
        target_date += timedelta(days=1)
    df_adsb = pl.concat(df_adsbs)
    perfect_icaos = get_icaos_with_perfect_match(df_flights, df_adsb)
    df_flights = df_flights.filter(pl.col("icao").is_in(perfect_icaos))
    df_flights_sim, gold_mapping = simulate_pia_flights(df_flights)
    return df_flights_sim, gold_mapping

if __name__ == "__main__":
    df_flights_sim, gold_mapping = get_simulated_flights_0()
    print(df_flights_sim)