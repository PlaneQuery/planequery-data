from datetime import date, timedelta
from pathlib import Path
import sys

import polars as pl
from data_engineering.flights.flight_type import get_flights
from data_engineering.openairframes.read import scan_latest_faa_registration_icao_for_join
from data_engineering.flights.flight_type import flights_df_to_flights
from data_engineering.utils import OUTPUT_DIR
from pia.evaluation.ground_control import get_ground_control_mappings
from pia.main import run_pia_algorithm
from pia.evaluation.main import evaluation, visualize_graph

DF_FLIGHTS_PATH = OUTPUT_DIR / "tmp" / "real_flights_1_df_flights.parquet"
df_mappings = get_ground_control_mappings()
start_date = date(2026,6,1)
df_mappings = df_mappings.filter(pl.col("date") > start_date)
if DF_FLIGHTS_PATH.exists():
    df_flights = pl.read_parquet(DF_FLIGHTS_PATH)
else:
    df_flights = get_flights(date(2024, 1, 1), date(2026, 6, 9))
    df_flights.write_parquet(DF_FLIGHTS_PATH)


real_icaos = df_mappings.get_column("icao").unique().to_list()
pia_icaos = df_mappings.get_column("pia_icao").unique().to_list()
icaos = set(real_icaos) | set(pia_icaos)

df_flights = df_flights.filter(pl.col("icao").is_in(icaos))

pia_to_real_icao_map = dict(zip(df_mappings["pia_icao"], df_mappings["icao"]))
gold_dict = dict()
pia_flights = flights_df_to_flights(df_flights.filter((pl.col("icao").is_in(pia_icaos)) & (pl.col("takeoff_time").dt.date() >= start_date)))
for flight in pia_flights:
    gold_dict[flight] = pia_to_real_icao_map[flight.icao]

mappings, g = run_pia_algorithm(df_flights, start_date)
mappings = {
    flight: mapped_icao
    for flight, mapped_icao in mappings.items()
    if flight in gold_dict
}
stats = evaluation(mappings, gold_dict)
visualize_graph(g, OUTPUT_DIR/ "tmp" / "pia_graph_flights_real_flights_1.png")
print(stats)
