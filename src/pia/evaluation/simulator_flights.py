# Assume flights are perfect. I'm going to take from my flights table and see what percentage of pia flights can be identified.
from datetime import date, datetime, timedelta
import polars as pl
from data_engineering.adsb.read_adsb import ADSBSource, read_icaos_from_adsb
from flights.algorithm import AlgorithmNearestFlightSource
from flights.flight_source import get_flights
from flights.flight_type import flights_df_to_flights
from pia.simulator.main import simulate_pia_flights
from flights.flights_match_adsb import get_matching_icaos_in_flights


def get_simulated_flights_0():
    start_date = date(2026, 5, 20)
    end_date = date(2026, 5, 22)

    flight_dfs = []
    target_date = start_date
    while target_date < end_date:
        flight_dfs.append(
            get_flights(
                AlgorithmNearestFlightSource(),
                target_date,
                ADSBSource.ADSBLOL,
            )
        )
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
    df_flights = get_matching_icaos_in_flights(
        df_flights, df_adsb, datetime(start_date.year, start_date.month, start_date.day)
    )
    df_flights_sim, gold_mapping = simulate_pia_flights(df_flights)
    return df_flights_sim, gold_mapping

if __name__ == "__main__":
    df_flights_sim, gold_mapping = get_simulated_flights_0()
    print(df_flights_sim)
