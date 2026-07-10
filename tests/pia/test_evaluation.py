import os
from pathlib import Path


os.environ.setdefault("OUTPUT_DIR", str(Path("output").resolve()))

from pia.evaluation.main import evaluation
from pia.evaluation.pia_graph_0 import test_flight as make_test_flight


def test_evaluation_accepts_unresolved_pia_mapping():
    predicted_flight = make_test_flight("pia_0")
    expected_flight = make_test_flight("pia_1")

    stats = evaluation(
        {predicted_flight: "unresolved_pia:a0398c"},
        {expected_flight: "a0398c"},
    )

    assert stats == {
        "expected_mappings": 1,
        "predicted_mappings": 1,
        "correct_mappings": 0,
        "missing_predictions": 1,
        "extra_predictions": 1,
        "accuracy": 0.0,
    }
