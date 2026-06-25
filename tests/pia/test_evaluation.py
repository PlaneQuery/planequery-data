import os
from pathlib import Path


os.environ.setdefault("OUTPUT_DIR", str(Path("output").resolve()))

from pia.evaluation.main import evaluation


def test_evaluation_accepts_unresolved_pia_mapping():
    stats = evaluation(
        {"pia_0_2026-01-01_05-00": "unresolved_pia:a0398c"},
        {"pia_1_2026-01-01_05-00": "a0398c"},
    )

    assert stats == {
        "expected_mappings": 1,
        "predicted_mappings": 1,
        "correct_mappings": 0,
        "missing_predictions": 1,
        "extra_predictions": 1,
        "accuracy": 0.0,
    }
