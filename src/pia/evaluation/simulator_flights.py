# Assume flights are perfect. I'm going to take from my flights table and see what percentage of pia flights can be identified.
from datetime import date, timedelta
import polars as pl
from data_engineering.flights.flight_type import get_flights, flights_df_to_flights
from pia.simulator.main import simulate_pia_flights

start_date = date(2026, 2, 28)
end_date = date(2026, 3, 1)

flight_dfs = []
target_date = start_date
while target_date < end_date:
    flight_dfs.append(get_flights(target_date))
    target_date += timedelta(days=1)

df_flights = pl.concat(flight_dfs)
df_flights = df_flights.filter(pl.col("ladd"))
df_flights_sim, gold_mapping = simulate_pia_flights(df_flights)