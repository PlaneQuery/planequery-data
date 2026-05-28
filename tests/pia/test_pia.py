import contextlib
import io
import os
import unittest
from pathlib import Path

os.environ.setdefault("OUTPUT_DIR", str(Path("output").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp/matplotlib").resolve()))

from pia.evaluation.main import build_graph, process_graph
from pia.evaluation.pia_graph_0 import flights_0, test_mapping_dict_0


class TestPIA(unittest.TestCase):
    def test_flights_0_mapping(self):
        graph = build_graph(flights_0)
        _, mapping_dict = process_graph(graph)

        self.assertEqual(test_mapping_dict_0, mapping_dict)
