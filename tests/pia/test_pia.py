from datetime import date
import os
from pathlib import Path

import pytest

os.environ.setdefault("OUTPUT_DIR", str(Path("output").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp/matplotlib").resolve()))

from data_engineering.flights.flight_type import flights_df_to_flights
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
    mapping_dict, g = main(date(2026,5,14), date(2026,6,1))
    correct = 0
    total = 0
    for flight, real_icao in mapping_dict_0.items():
        if flight in mapping_dict:
            total += 1
            if mapping_dict[flight] == real_icao:
                correct += 1
    print(f"Accuracy: {correct}/{total} = {correct/total:.2%}")

