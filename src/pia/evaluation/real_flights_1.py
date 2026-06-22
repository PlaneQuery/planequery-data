from datetime import date, timedelta
from pathlib import Path
import sys

import polars as pl
from data_engineering.flights.flight_type import get_flights
from data_engineering.openairframes.read import scan_latest_faa_registration_icao_for_join
from data_engineering.flights.flight_type import flights_df_to_flights
from pia.main import run_pia_algorithm
from pia.evaluation.main import evaluation

mappings_path = Path("/Users/jonahgoode/Documents/PlaneQuery/r-piaicao jacksweeney mappings.csv")

df_mappings = pl.read_csv(mappings_path).rename(
    {
        "PIA ICAO": "pia_icao",
        "Tail Number": "tailnumber",
        "Date": "date",
    }
).with_columns(
    pl.col("pia_icao").str.strip_chars().str.to_lowercase(),
    pl.col("tailnumber").str.strip_chars().str.to_uppercase(),
    pl.col("date").str.strptime(pl.Date, strict=False),
)
df_real_icaos = scan_latest_faa_registration_icao_for_join().collect()

df_mappings = df_mappings.join(
    df_real_icaos,
    left_on="tailnumber",
    right_on="registration",
    how="left",
)

df_mappings = df_mappings.filter(pl.col("real_icao").is_not_null())
df_flights = get_flights(date(2026, 5, 1), date(2026, 6, 9))
real_icaos_avaliable = df_flights.filter(pl.col("icao").is_in(df_mappings.get_column("real_icao").unique().to_list())).get_column("icao").unique().to_list()
df_mappings = df_mappings.filter(pl.col("real_icao").is_in(real_icaos_avaliable))
pia_icaos = df_mappings.get_column("pia_icao").unique().to_list()
icaos = set(real_icaos_avaliable) | set(pia_icaos)
df_flights = df_flights.filter(pl.col("icao").is_in(icaos))

pia_to_real_icao_map = dict(zip(df_mappings["pia_icao"], df_mappings["real_icao"]))
gold_dict = dict()
pia_flights = flights_df_to_flights(df_flights.filter(pl.col("pia")))
for flight in pia_flights:
    gold_dict[flight.flight_id] = pia_to_real_icao_map[flight.icao]

mappings, g = run_pia_algorithm(date(2026, 5, 1), date(2026, 6, 9), icaos=list(gold_dict.values()) + pia_icaos )
stats = evaluation(mappings, gold_dict)
print(stats)