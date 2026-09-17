import json
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

import flights_airport_nn.evaluate_native_sources as evaluation
from flights_airport_nn.evaluate_native_sources import temporal_matches
from flights_airport_nn.report_native_evaluation import (
    accuracy,
    annotate,
    conflicting_reference_ids,
)


def test_temporal_matching_is_independent_of_labels_and_reference_source():
    start = datetime(2026, 3, 1, 10, tzinfo=UTC)
    segments = pl.DataFrame(
        {
            "flight_id": ["native"],
            "icao": ["abc123"],
            "takeoff_time": [start],
            "landing_time": [start + timedelta(hours=1)],
        }
    )
    references = pl.DataFrame(
        {
            "reference_id": ["faa", "bts", "missing"],
            "reference_source": ["FAA", "BTS", "FAA"],
            "icao": ["abc123", "abc123", "def456"],
            "takeoff_time": [start] * 3,
            "landing_time": [start + timedelta(hours=1)] * 3,
            "takeoff_truth": ["WRONG", "DIFFERENT", "OTHER"],
        }
    )
    matches = temporal_matches(segments, references)
    assert set(matches["reference_id"]) == {"faa", "bts"}
    assert matches["unique_match"].all()
    changed = references.with_columns(pl.lit("ARBITRARY").alias("takeoff_truth"))
    assert temporal_matches(segments, changed).equals(matches)


def test_ambiguous_same_source_references_are_not_cherry_picked():
    start = datetime(2026, 3, 1, 10, tzinfo=UTC)
    segments = pl.DataFrame(
        {
            "flight_id": ["native"],
            "icao": ["abc123"],
            "takeoff_time": [start],
            "landing_time": [start + timedelta(hours=1)],
        }
    )
    references = pl.DataFrame(
        {
            "reference_id": ["one", "two"],
            "reference_source": ["FAA", "FAA"],
            "icao": ["abc123"] * 2,
            "takeoff_time": [start, start + timedelta(minutes=1)],
            "landing_time": [start + timedelta(hours=1)] * 2,
        }
    )
    matches = temporal_matches(segments, references)
    assert matches.height == 2
    assert not matches["unique_match"].any()


def test_ground_corroboration_does_not_depend_on_predictions():
    frame = pl.DataFrame(
        {
            "reference_id": ["r", "r"],
            "endpoint": ["takeoff", "takeoff"],
            "reference_source": ["FAA", "FAA"],
            "source": ["adsblol", "adsbx"],
            "icao": ["abc123", "abc123"],
            "truth": ["TEST", "TEST"],
            "loss_seconds": [0, 0],
            "truth_ground": [False, True],
            "nn": ["WRONG", "TEST"],
            "nearest": ["TEST", "WRONG"],
            "candidates": [["TEST"], ["TEST"]],
        }
    )
    refs = pl.DataFrame(schema={"date": pl.String})
    catalog = {"TEST": {"type": "small_airport", "iso_country": "US"}}
    lineage = {"aircraft": [], "airports": []}
    result, _ = annotate(frame, refs, lineage, catalog)
    changed, _ = annotate(
        frame.with_columns(
            pl.lit("ARBITRARY").alias("nn"), pl.lit("ARBITRARY").alias("nearest")
        ),
        refs,
        lineage,
        catalog,
    )
    assert result["label_quality"].to_list() == ["faa_ground_corroborated"] * 2
    assert changed["label_quality"].equals(result["label_quality"])
    assert result["ground_adsbx"].all()


def test_missing_segment_stays_in_full_reference_denominator():
    frame = pl.DataFrame(
        {
            "reference_id": ["one", "two"],
            "icao": ["abc123", "abc123"],
            "truth": ["TEST", "TEST"],
            "match_status": ["matched", "unmatched"],
            "observed": [True, False],
            "nn_correct": [True, False],
            "nearest_correct": [True, False],
            "retrieved": [True, False],
        }
    )
    stats = accuracy(frame)
    assert stats["endpoints"] == 2
    assert stats["unmatched"] == 1
    assert stats["nn_accuracy"] == stats["nearest_accuracy"] == 0.5


def test_novelty_and_leakage_checks_include_pretraining(tmp_path, monkeypatch):
    initial = tmp_path / "initial" / "model.pt"
    final = tmp_path / "final" / "model.pt"
    initial.parent.mkdir()
    final.parent.mkdir()
    initial.with_name("config.json").write_text(
        json.dumps({"train_dates": ["2026-02-01"], "validation_dates": ["2026-02-25"]})
    )
    final.with_name("config.json").write_text(
        json.dumps(
            {
                "train_dates": ["2026-06-01"],
                "validation_dates": ["2026-06-03"],
                "initial_model": str(initial),
            }
        )
    )
    broad = tmp_path / "cache" / "2026-06-01"
    broad.mkdir(parents=True)
    np.savez(
        broad / "part-0.npz",
        flight_id=np.array(["abc123_2026-06-01_00-00"]),
        truth=np.array(["SMALL"]),
    )
    pretrain = tmp_path / "pretrain.npz"
    np.savez(
        pretrain,
        flight_id=np.array(["def456_2026-02-01_00-00"]),
        truth=np.array(["LARGE"]),
    )
    monkeypatch.setattr(evaluation, "CACHE_ROOT", broad.parent)
    monkeypatch.setattr(evaluation, "cache_path", lambda _: pretrain)
    lineage = evaluation.model_lineage(final)
    assert lineage["aircraft"] == ["abc123", "def456"]
    assert lineage["airports"] == ["LARGE", "SMALL"]
    assert lineage["train_dates"] == ["2026-02-01", "2026-06-01"]
    assert lineage["validation_dates"] == ["2026-02-25", "2026-06-03"]


def test_conflicting_reference_flights_flag_both_records_without_predictions():
    start = datetime(2026, 3, 1, 10, tzinfo=UTC)
    refs = pl.DataFrame(
        {
            "reference_id": ["one", "two", "independent"],
            "reference_source": ["FAA", "FAA", "BTS"],
            "date": ["2026-03-01"] * 3,
            "icao": ["abc123"] * 3,
            "takeoff_time": [start, start + timedelta(minutes=30), start],
            "landing_time": [
                start + timedelta(hours=1),
                start + timedelta(minutes=90),
                start + timedelta(hours=1),
            ],
        }
    )
    assert conflicting_reference_ids(refs) == {"one", "two"}
