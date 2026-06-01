import contextlib
import io
import os
import unittest
from pathlib import Path

os.environ.setdefault("OUTPUT_DIR", str(Path("output").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp/matplotlib").resolve()))

from data_engineering.flights.flight_type import flights_df_to_flights
from pia.evaluation.main import visualize_graph
from pia.resolver import resolve_pia_mappings
from pia.evaluation.pia_graph_0 import flights_0, test_mapping_dict_0

class TestPIA(unittest.TestCase):
    def test_flights_0_mapping(self):
        mapping_dict, graph = resolve_pia_mappings(flights_0)
        self.assertEqual(mapping_dict, test_mapping_dict_0)

    def test_simulator_flights_0(self):
        from pia.evaluation.simulator_flights import df_flights_sim, gold_mapping
        mapping_dict, graph = resolve_pia_mappings(flights_df_to_flights(df_flights_sim))
        correct = 0
        total = 0
        for flight_id, real_icao in gold_mapping.items():
            if flight_id in mapping_dict:
                total += 1
                if mapping_dict[flight_id] == real_icao:
                    correct += 1
        print(f"Accuracy: {correct}/{total} = {correct/total:.2%}")
if __name__ == "__main__":
    unittest.main()
