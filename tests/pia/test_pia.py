from datetime import date
import os
from pathlib import Path

import pytest

os.environ.setdefault("OUTPUT_DIR", str(Path("output").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp/matplotlib").resolve()))

from data_engineering.flights.flight_type import flights_df_to_flights
from pia.resolver import resolve_pia_mappings
from pia.evaluation.pia_graph_0 import flights_0, flights_2, flights_3, test_mapping_dict_3, test_mapping_dict_0
from pia.main import run_pia_algorithm
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
    mapping_dict, _graph = resolve_pia_mappings(flights_df_to_flights(df_flights_sim))
    correct = 0
    total = 0
    for flight, real_icao in gold_mapping.items():
        if flight in mapping_dict:
            total += 1
            if mapping_dict[flight] == real_icao:
                correct += 1
    print(mapping_dict)
    print(f"Accuracy: {correct}/{total} = {correct/total:.2%}")


@pytest.mark.skip(reason="skipping real-data PIA tests for now")
def test_real_flights_0():
    from pia.evaluation.real_flights_0 import flights_0, mapping_dict_0

    print(len(flights_0))
    mapping_dict, _graph = run_pia_algorithm(date(2025, 5, 15), date(2025, 5, 21))
    correct = 0
    total = 0
    for flight, real_icao in mapping_dict_0.items():
        if flight in mapping_dict:
            total += 1
            if mapping_dict[flight] == real_icao:
                correct += 1
    print(f"Accuracy: {correct}/{total} = {correct/total:.2%}")


@pytest.mark.skip(reason="skipping real-data PIA tests for now")
def test_real_flights_1():
    start_date = date(2025, 5, 14)
    end_date = date(2025, 5, 21)
    mappings, graph = run_pia_algorithm(start_date, end_date)
    check_value = None
    keys = sorted(mappings, key=lambda flight: flight.flight_id)
    for key in keys:
        if key.flight_id.startswith(a039a2_flights[0].icao):
            print(key, mappings[key])
    for node in keys:
        if node.flight_id.startswith(a039a2_flights[0].icao):
            incoming = list(graph.in_edges(node.flight_id, data=True))
            outgoing = list(graph.out_edges(node.flight_id, data=True))
            all_edges = incoming + outgoing
            print(node, all_edges)

    for key in keys:
        if key.flight_id.startswith(a039a2_flights[0].icao):
            value = mappings[key]
            if check_value is None:
                check_value = value
            else:
                assert value == check_value, "multiple icaos used"
