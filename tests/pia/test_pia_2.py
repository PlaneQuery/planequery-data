import polars as pl

from data_engineering.flights.flight_type import flights_df_to_flights
from data_engineering.utils import OUTPUT_DIR
from pia.main import run_pia_algorithm
from datetime import date
from src.pia.evaluation.main import evaluation, visualize_graph
def test_0():
    from src.pia.evaluation.real_flights_1 import df_flights, gold_dict, pia_icaos
    mappings, g = run_pia_algorithm(date(2026, 6, 1), date(2026, 6, 8), icaos=list(gold_dict.values()) + pia_icaos )
    stats = evaluation(mappings, gold_dict)
    visualize_graph(g, OUTPUT_DIR/"tmp"/"pia_graph_test_0.png")
    print(stats)
    