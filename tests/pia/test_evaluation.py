from datetime import timedelta
import os
from pathlib import Path


os.environ.setdefault("OUTPUT_DIR", str(Path("output").resolve()))

import pytest

from pia.evaluation.main import evaluation, traceability_stats
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
        "expected_aircraft": 1,
        "traceable_aircraft": 0,
        "lost_aircraft": 1,
        "traceability_index": 0.0,
    }


def test_traceability_requires_all_mappings_for_an_aircraft_to_be_correct():
    aircraft_a_flight_1 = make_test_flight("pia_a_1")
    aircraft_a_flight_2 = make_test_flight(
        "pia_a_2",
        takeoff_time=aircraft_a_flight_1.takeoff_time + timedelta(days=1),
    )
    aircraft_b_flight = make_test_flight("pia_b")
    gold_mapping = {
        aircraft_a_flight_1: "A00001",
        aircraft_a_flight_2: "A00001",
        aircraft_b_flight: "B00002",
    }
    predicted_mapping = {
        aircraft_a_flight_1: "A00001",
        aircraft_a_flight_2: "incorrect",
        aircraft_b_flight: "B00002",
    }

    stats = evaluation(predicted_mapping, gold_mapping)

    assert stats["accuracy"] == pytest.approx(2 / 3)
    assert stats["expected_aircraft"] == 2
    assert stats["traceable_aircraft"] == 1
    assert stats["lost_aircraft"] == 1
    assert stats["traceability_index"] == 0.5


def test_traceability_counts_a_missing_mapping_as_a_lost_aircraft():
    aircraft_a_flight = make_test_flight("pia_a")
    aircraft_b_flight = make_test_flight("pia_b")

    stats = traceability_stats(
        {aircraft_a_flight: "A00001"},
        {
            aircraft_a_flight: "A00001",
            aircraft_b_flight: "B00002",
        },
    )

    assert stats == {
        "expected_aircraft": 2,
        "traceable_aircraft": 1,
        "lost_aircraft": 1,
        "traceability_index": 0.5,
    }


def test_traceability_of_empty_cohort_is_zero():
    assert traceability_stats({}, {}) == {
        "expected_aircraft": 0,
        "traceable_aircraft": 0,
        "lost_aircraft": 0,
        "traceability_index": 0,
    }
