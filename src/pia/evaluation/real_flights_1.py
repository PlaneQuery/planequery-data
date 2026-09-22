from datetime import date, timedelta
from pathlib import Path
import sys

import polars as pl
from flights.algorithm import get_flights_range
from data_engineering.openairframes.read import scan_latest_faa_registration_icao_for_join
from flights.flight_type import flights_df_to_flights
from data_engineering.utils import OUTPUT_DIR
from data_engineering.ground_control import load_ground_control_mappings
from pia.main import run_pia_algorithm
from pia.evaluation.main import evaluation, visualize_graph

start_date = date(2026,5,1)
end_date = date(2026,6,9)
DF_FLIGHTS_PATH = OUTPUT_DIR / "tmp" / f"real_flights_1_df_flights_{start_date}_{end_date}.parquet"
if DF_FLIGHTS_PATH.exists():
    df_flights = pl.read_parquet(DF_FLIGHTS_PATH)
else:
    df_flights = get_flights_range(start_date, end_date)
    df_flights.write_parquet(DF_FLIGHTS_PATH)


df_mappings = load_ground_control_mappings()
df_mappings = df_mappings.filter(pl.col("date") > start_date)
real_icaos = df_mappings.get_column("icao").unique().to_list()
pia_icaos = df_mappings.get_column("pia_icao").unique().to_list()
icaos = set(real_icaos) | set(pia_icaos)

df_flights = df_flights.filter(pl.col("icao").is_in(icaos) | pl.col("pia"))

pia_to_real_icao_map = dict(zip(df_mappings["pia_icao"], df_mappings["icao"]))
gold_dict = dict()
pia_flights = flights_df_to_flights(df_flights.filter((pl.col("pia"))))
for flight in pia_flights:
    if flight.icao in pia_to_real_icao_map:
        gold_dict[flight] = pia_to_real_icao_map[flight.icao]

mappings, g = run_pia_algorithm(df_flights)
mappings = {
    flight: mapped_icao
    for flight, mapped_icao in mappings.items()
    if flight in gold_dict
}
stats = evaluation(mappings, gold_dict)
visualize_graph(g, OUTPUT_DIR/ "tmp" / "pia_graph_flights_real_flights_1.png")
print(stats)
