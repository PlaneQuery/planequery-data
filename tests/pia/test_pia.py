from datetime import date
import os
from pathlib import Path
import polars as pl
import pytest

from data_engineering.ground_control import load_ground_control_mappings

os.environ.setdefault("OUTPUT_DIR", str(Path("output").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp/matplotlib").resolve()))

from data_engineering.flights.flight_type import flights_df_to_flights, get_flights, pia_or_american_ladd_icao_filter
from pia.resolver import resolve_pia_mappings
from pia.evaluation.pia_graph_0 import flights_0, flights_2, flights_3, test_mapping_dict_3, test_mapping_dict_0
from pia.main import main
from pia.evaluation.real_flights_0 import a039a2_flights

def test_flights_0_mapping():
    mapping_dict, _graph = resolve_pia_mappings(flights_0)
    assert mapping_dict == test_mapping_dict_0
    

def test_flights_2_mapping():
    mapping_dict, _graph = resolve_pia_mappings(flights_2)
    assert mapping_dict == test_mapping_dict_0


def test_flights_3_mapping():
    mapping_dict, _graph = resolve_pia_mappings(flights_3)
    assert mapping_dict == test_mapping_dict_3


def test_simulator_flights_0():
    # TODO: What is the random assignment percentage
    from pia.evaluation.simulator_flights import get_simulated_flights_0

    df_flights_sim, gold_mapping = get_simulated_flights_0()
    print(len(df_flights_sim))
    mapping_dict = resolve_pia_mappings(flights_df_to_flights(df_flights_sim))
    correct = 0
    total = 0
    for flight, real_icao in gold_mapping.items():
        if flight in mapping_dict:
            total += 1
            if mapping_dict[flight] == real_icao:
                correct += 1
    print(mapping_dict)
    print(f"Accuracy: {correct}/{total} = {correct/total:.2%}")

def test_real_flights_0():
    df_groundcontrol = load_ground_control_mappings()
    start_date = date(2026,5,14)
    end_date = date(2026,6,10)
    df_groundcontrol = df_groundcontrol.filter((pl.col("date") > start_date) & (pl.col("date") < end_date)).sort("date")
    pia_to_real_icao = dict(zip(df_groundcontrol["pia_icao"], df_groundcontrol["real_icao"]))
    print(f"len(pia_to_real_icao): {len(pia_to_real_icao)}")
    df_flights = get_flights(start_date, end_date)
    df_flights = df_flights.filter(pia_or_american_ladd_icao_filter())
    df_flights = df_flights.filter(pl.col("icao").str.len_chars() == 6)
    df_flights = df_flights.filter(pl.col("icao").is_in(pia_to_real_icao.keys() | pia_to_real_icao.values()))
    mappings = resolve_pia_mappings(flights_df_to_flights(df_flights))
    correct = 0
    total = 0
    for flight, real_icao in mappings.items():
        if flight.icao in pia_to_real_icao.keys():
            correct_real_icao = pia_to_real_icao[flight.icao]
            if real_icao == correct_real_icao:
                correct += 1
            total += 1

    print(f"Accuracy: {correct}/{total} = {correct/total:.2%}")
