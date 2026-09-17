from flights_ai_model_airport.llm_benchmark import (
    _public_case,
    evaluate_predictions,
)


def _case(
    case_id: str,
    endpoint: str,
    *,
    baseline_correct: bool,
    target_in_candidates: bool = True,
) -> dict:
    target = f"K{case_id}"
    baseline = target if baseline_correct else "KBASE"
    return {
        "case_id": case_id,
        "date": "2026-02-01",
        "flight_id": f"flight-{case_id}",
        "endpoint": endpoint,
        "offset_meaning": "minutes after the first flight message",
        "observations": [],
        "candidates": [
            {"ident": baseline, "baseline": True},
            {"ident": target, "baseline": False},
        ],
        "baseline_airport_ident": baseline,
        "target_airport_ident": target,
        "baseline_correct": baseline_correct,
        "target_in_candidates": target_in_candidates,
        "label_source": "sfdps",
    }


def test_public_case_removes_all_evaluation_labels():
    case = _case("001", "takeoff", baseline_correct=False)

    public = _public_case(case)

    assert "target_airport_ident" not in public
    assert "baseline_correct" not in public
    assert "target_in_candidates" not in public
    assert "label_source" not in public
    assert public["baseline_airport_ident"] == "KBASE"


def test_evaluation_reports_preservation_and_correction_rates():
    cases = [
        _case("001", "takeoff", baseline_correct=True),
        _case("002", "takeoff", baseline_correct=False),
        _case("003", "landing", baseline_correct=True),
        _case("004", "landing", baseline_correct=False),
    ]
    predictions = [
        {
            "case_id": case["case_id"],
            "airport_ident": case["target_airport_ident"],
            "confidence": 0.9,
            "rationale": "test",
        }
        for case in cases
    ]

    _rows, metrics = evaluate_predictions(cases, predictions)

    assert metrics["balanced_sample_accuracy"] == 1.0
    assert metrics["baseline_preservation_rate"] == 1.0
    assert metrics["baseline_error_correction_rate"] == 1.0
    assert metrics["correction_rate_when_target_is_candidate"] == 1.0
