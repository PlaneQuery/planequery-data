import polars as pl

from flights.evaluation import airport_benchmark
from flights.evaluation.airport_benchmark import shared_quality, summarize


def test_network_quality_is_shared_and_independent_of_predictions():
    frame = pl.DataFrame(
        {
            "reference_id": ["one", "one", "two", "two"],
            "endpoint": ["takeoff"] * 4,
            "network": ["adsblol", "adsbx"] * 2,
            "ground_supported": [False, True, False, False],
            "eligible": [True] * 4,
            "reported": [False] * 4,
            "nn_correct": [False, True, True, True],
            "nearest_correct": [False, True, True, True],
        }
    )
    actual = shared_quality(frame).sort("reference_id", "network")
    assert actual["quality_eligible"].to_list() == [True, True, False, False]
    changed = shared_quality(
        frame.with_columns(~pl.col("nn_correct"), ~pl.col("nearest_correct"))
    ).sort("reference_id", "network")
    assert actual["quality_eligible"].equals(changed["quality_eligible"])


def test_flight_accuracy_requires_both_endpoints_and_keeps_retrieval_misses():
    frame = pl.DataFrame(
        {
            "reference_id": ["one", "one", "two", "two", "three"],
            "endpoint": ["takeoff", "landing", "takeoff", "landing", "takeoff"],
            "truth": ["A", "B", "A", "C", "A"],
            "airport_type": ["small_airport"] * 5,
            "quality_eligible": [True] * 5,
            "eligible": [True] * 5,
            "nn_correct": [True, True, True, False, True],
            "nearest_correct": [True, False, True, False, True],
            "target": [0, 1, 0, -1, 0],
        }
    )
    result = summarize(frame)["primary"]
    assert result["endpoints"] == 5
    assert result["nn_pct"] == result["candidate_recall_pct"] == 80
    assert result["nearest_pct"] == 60
    assert result["complete_flights"] == 2
    assert result["nn_both_pct"] == 50
    assert result["nearest_both_pct"] == 0


def test_report_retains_unmatched_references_and_empty_cohorts(monkeypatch, tmp_path):
    from datetime import date

    day = date(2026, 3, 1)
    folder = tmp_path / str(day)
    folder.mkdir()
    part = folder / "part.npz"
    pl.DataFrame(
        {
            "reference_id": ["matched", "missing"],
            "source": ["broad"] * 2,
            "matched": [True, False],
            "time_conflict": [False] * 2,
            "observations_in_reference_window": [10, 0],
        }
    ).write_parquet(part.with_name("part-references.parquet"))
    frame = pl.DataFrame(
        {
            "reference_id": ["matched"] * 2,
            "endpoint": ["takeoff", "landing"],
            "network": ["adsblol"] * 2,
            "source": ["broad"] * 2,
            "truth": ["A", "B"],
            "airport_type": ["small_airport"] * 2,
            "target": [0, 0],
            "nn": ["A", "B"],
            "nearest": ["A", "C"],
            "nn_correct": [True, True],
            "nearest_correct": [True, False],
            "eligible": [True] * 2,
            "reported": [True] * 2,
            "ground_supported": [False] * 2,
            "takeoff_time": [1] * 2,
            "landing_time": [2] * 2,
        }
    )
    model = tmp_path / "mock.pt"
    model.write_bytes(b"mock checkpoint")
    monkeypatch.setattr(
        airport_benchmark.torch,
        "load",
        lambda *a, **kw: {
            "training": {
                "train_dates": ["2026-02-01"],
                "validation_dates": ["2026-02-21"],
            }
        },
    )
    monkeypatch.setattr(
        airport_benchmark, "prepare", lambda *a: (None, {"adsblol": [part]})
    )
    monkeypatch.setattr(airport_benchmark, "load_model", lambda *a: None)
    monkeypatch.setattr(airport_benchmark, "score", lambda *a: (frame, 0))
    result = airport_benchmark.run(day, ["adsblol"], model, tmp_path)
    assert result["results"]["adsblol"]["broad"]["match_pct"] == 50
    assert result["results"]["adsblol"]["original"]["match_pct"] is None
    refs = pl.read_parquet(folder / "references.parquet").sort("reference_id")
    assert refs["match_status"].to_list() == ["both", "reference_only"]
    assert refs["takeoff_nn"].to_list() == ["A", None]
