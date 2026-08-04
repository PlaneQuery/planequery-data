from datetime import date
import os
from pathlib import Path
import polars as pl

os.environ.setdefault("OUTPUT_DIR", str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp/matplotlib").resolve()))

from data_engineering.ground_control import load_ground_control_mappings
from data_engineering.flights.flight_type import FlightIndex, flights_df_to_flights, get_flights, pia_or_american_ladd_icao_filter
from pia.resolver import build_candidates, check_candidate_mapping_paths, resolve_pia_mappings
from pia.evaluation.pia_graph_0 import flights_0, flights_2, flights_3, test_mapping_dict_3, test_mapping_dict_0
from pia.main import main
from pia.evaluation.real_flights_0 import a039a2_flights

def test_flights_0_mapping():
    mapping_dict = resolve_pia_mappings(flights_0)
    assert mapping_dict == test_mapping_dict_0
    

def test_flights_2_mapping():
    mapping_dict = resolve_pia_mappings(flights_2)
    assert mapping_dict == test_mapping_dict_0


def test_flights_3_mapping():
    mapping_dict = resolve_pia_mappings(flights_3)
    assert mapping_dict == test_mapping_dict_3


def test_simulator_flights_0():
    # TODO: What is the random assignment percentage
    from pia.evaluation.simulator_flights import get_simulated_flights_0

    df_flights_sim, gold_mapping = get_simulated_flights_0()
    mapping_dict = resolve_pia_mappings(flights_df_to_flights(df_flights_sim))
    correct = 0
    total = 0
    for flight, real_icao in gold_mapping.items():
        if flight in mapping_dict:
            total += 1
            if mapping_dict[flight] == real_icao:
                correct += 1
    print(f"Accuracy: {correct}/{total} = {correct/total:.2%}")

def test_real_flights_0():
    start_date = date(2025,4,1)
    end_date = date(2025,7,16)
    correct_number = 35
    total_pia_known_flights, correct = pia_tester(start_date, end_date, correct_number)
    print(f"Accuracy: {correct}/{total_pia_known_flights} = {correct/total_pia_known_flights:.2%}")

def test_real_flights_1():
    start_date = date(2025,4,1)
    end_date = date(2025,7,16)
    correct_number = 35
    total_pia_known_flights, correct = pia_tester(start_date, end_date, correct_number, filter=False)
    print(f"Accuracy: {correct}/{total_pia_known_flights} = {correct/total_pia_known_flights:.2%}")

def test_real_flights_2():
    start_date = date(2025,4,1)
    end_date = date(2025,7,16)
    total_pia_known_flights, correct = pia_tester(start_date, end_date, correct_number=0)
    print(f"Accuracy: {correct}/{total_pia_known_flights} = {correct/total_pia_known_flights:.2%}")
def test_real_flights_3():
    start_date = date(2025,4,1)
    end_date = date(2025,7,16)
    total_pia_known_flights, correct = pia_tester(start_date, end_date, correct_number=0, filter=False)
    print(f"Accuracy: {correct}/{total_pia_known_flights} = {correct/total_pia_known_flights:.2%}")
    
def pia_tester(start_date: date, end_date: date, correct_number: int, filter=True):
    df_groundcontrol = load_ground_control_mappings()
    df_groundcontrol = df_groundcontrol.filter((pl.col("date") > start_date) & (pl.col("date") < end_date)).sort("date")
    pia_to_real_icao = dict(zip(df_groundcontrol["pia_icao"], df_groundcontrol["real_icao"]))
    df_flights = get_flights(start_date, end_date)
    for pia_icao, real_icao in list(pia_to_real_icao.items()):
        n_pia_flights = len(df_flights.filter(pl.col("icao") == pia_icao))
        n_real_flights = len(df_flights.filter(pl.col("icao") == real_icao))
        if n_pia_flights == 0 or n_real_flights == 0:
            print(f"Warning: No flights found for pia_icao {pia_icao} or real_icao {real_icao}")
            pia_to_real_icao.pop(pia_icao) 
    print(f"len(pia_to_real_icao): {len(pia_to_real_icao)}")
    if filter:
        df_flights = df_flights.filter(pl.col("icao").is_in(pia_to_real_icao.keys() | pia_to_real_icao.values()))
    else:
        df_flights = df_flights.filter(pl.col("icao").str.len_chars() == 6)
        df_flights = df_flights.filter(pia_or_american_ladd_icao_filter())
    flights = flights_df_to_flights(df_flights)
    flight_index = FlightIndex(flights)
    candidates = build_candidates(flight_index)
    candidate_path_results = check_candidate_mapping_paths(
        candidates,
        flight_index,
        pia_to_real_icao,
    )
    possible_mappings = sum(candidate_path_results.values())
    print(
        "Ground-control mappings with a path in the candidate graph: "
        f"{possible_mappings}/{len(candidate_path_results)}"
    )
    for pia_icao, path_exists in candidate_path_results.items():
        if not path_exists:
            print(
                "No candidate path for ground-control mapping: "
                f"{pia_icao} -> {pia_to_real_icao[pia_icao]}"
            )

    mappings = resolve_pia_mappings(flights)
    total_pia_known_flights = len(df_flights.filter(pl.col("icao").is_in(pia_to_real_icao.keys())))
    correct = 0
    for flight, real_icao in mappings.items():
        if flight.icao in pia_to_real_icao.keys():
            correct_real_icao = pia_to_real_icao[flight.icao]
            if real_icao == correct_real_icao:
                correct += 1

    # assert correct == total_pia_known_flights == correct_number
    return total_pia_known_flights,correct
