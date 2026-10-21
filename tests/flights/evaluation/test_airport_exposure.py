import hashlib
import json
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from data_engineering.adsb.read_adsb import ADSBSource
from flights.evaluation import airport_exposure, evaluation
from flights.evaluation.airport_exposure import (
    AIRPORT_EXPOSURE_SCHEMA,
    airport_exposure_report,
    load_training_airports,
)
from flights.evaluation.models import EvaluationResult, EvaluationResults, EvaluationStats


@pytest.fixture
def training_metadata_dir(tmp_path):
    folder = tmp_path / "data/processed/flights_airport_nn/february-2026/2026-02-01"
    folder.mkdir(parents=True)
    return folder


@pytest.fixture
def training_manifest(tmp_path, monkeypatch, training_metadata_dir):
    monkeypatch.setattr(airport_exposure, "OUTPUT_DIR", tmp_path)
    part = training_metadata_dir / "part-0000.parquet"
    pl.DataFrame({
        "truth_id": [1, 1, 2, 3, 4, None],
        "endpoint": ["takeoff", "landing", "landing", "takeoff", "landing", "takeoff"],
        "eligible": [True, True, True, False, True, True],
        "target": [0, 1, 0, 0, -1, 0],
    }).write_parquet(part)
    # The manifest, rather than a glob or validation data, defines membership.
    pl.DataFrame({"truth_id": [99]}).write_parquet(training_metadata_dir / "unlisted.parquet")
    manifest = tmp_path / "training.json"
    manifest.write_text(json.dumps({
        "inputs": {
            "train": [
                [str(part.with_suffix(".npz").relative_to(tmp_path)), 0, 0],
                [str(part.relative_to(tmp_path)), part.stat().st_size, part.stat().st_mtime_ns],
            ],
            "2026-02-21": [["missing-validation.parquet", 0, 0]],
        },
    }))
    return manifest


def test_training_airports_follow_optimizer_filter_and_manifest(training_manifest):
    training = load_training_airports(training_manifest)
    assert training.airport_counts.to_dicts() == [
        {"airport_id": 1, "training_endpoint_count": 2},
        {"airport_id": 2, "training_endpoint_count": 1},
    ]
    assert training.manifest_path == training_manifest
    assert training.manifest_sha256 == hashlib.sha256(
        training_manifest.read_bytes()
    ).hexdigest()


def test_training_airport_codes_resolve_on_each_training_date(
    training_manifest, training_metadata_dir, monkeypatch,
):
    pl.DataFrame({
        "date": ["2026-02-01", "2026-02-01", "2026-02-02", "2026-02-01", "2026-02-01"],
        "truth": ["KAAA", "KAAA", "KAAA", "EXCLUDED", "RETRIEVAL_MISS"],
        "eligible": [True, True, True, False, True],
        "target": [0, 0, 0, 0, -1],
    }).write_parquet(training_metadata_dir / "part-0000.parquet")
    days = []

    def catalog_for_day(day):
        days.append(day)
        return SimpleNamespace(
            get_airport_by_ident=lambda ident: SimpleNamespace(id=day.day)
            if ident == "KAAA" else None,
        )

    monkeypatch.setattr(airport_exposure, "airport_catalog_for_date", catalog_for_day)
    original = training_manifest.read_bytes()
    training = load_training_airports(training_manifest)
    assert training.airport_counts.to_dicts() == [
        {"airport_id": 1, "training_endpoint_count": 2},
        {"airport_id": 2, "training_endpoint_count": 1},
    ]
    assert set(days) == {date(2026, 2, 1), date(2026, 2, 2)}
    assert training_manifest.read_bytes() == original


def test_unresolved_training_code_is_not_silently_omitted(
    training_manifest, training_metadata_dir, monkeypatch,
):
    pl.DataFrame({
        "date": ["2026-02-01"], "truth": ["MISSING"], "eligible": [True], "target": [0],
    }).write_parquet(training_metadata_dir / "part-0000.parquet")
    monkeypatch.setattr(
        airport_exposure, "airport_catalog_for_date",
        lambda day: SimpleNamespace(get_airport_by_ident=lambda ident: None),
    )
    with pytest.raises(ValueError, match="Cannot resolve training airport 'MISSING' on 2026-02-01"):
        load_training_airports(training_manifest)


@pytest.mark.parametrize("prefix", ["", "/mnt/storage1/planequery-data/", "/another/machine/"])
def test_training_paths_use_output_dir_not_recorded_root(tmp_path, monkeypatch, prefix):
    output_dir = tmp_path / "output"
    monkeypatch.setattr(airport_exposure, "OUTPUT_DIR", output_dir)
    relative = Path("data/processed/flights_airport_nn/february-2026/2026-02-01/part-0000.parquet")
    part = output_dir / relative
    part.parent.mkdir(parents=True)
    pl.DataFrame({"truth_id": [7], "eligible": [True], "target": [0]}).write_parquet(part)
    manifest = tmp_path / "separate-model-directory" / "training.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"inputs": {"train": [[prefix + str(relative), 0, 0]]}}))
    original = manifest.read_bytes()
    assert load_training_airports(manifest).airport_counts["airport_id"].to_list() == [7]
    assert manifest.read_bytes() == original


def test_manifest_argument_can_be_relative_to_output_dir(training_manifest):
    training = load_training_airports(Path("training.json"))
    assert training.manifest_path == training_manifest
    assert training.airport_counts.height == 2


def test_training_counts_combine_multiple_parts(training_manifest, training_metadata_dir):
    part = training_metadata_dir / "part-0001.parquet"
    pl.DataFrame({
        "truth_id": [2, 5], "endpoint": ["takeoff", "landing"],
        "eligible": [True, True], "target": [0, 1],
    }).write_parquet(part)
    manifest = json.loads(training_manifest.read_text())
    manifest["inputs"]["train"].append([str(part.relative_to(training_manifest.parent)), 0, 0])
    training_manifest.write_text(json.dumps(manifest))
    assert load_training_airports(training_manifest).airport_counts.to_dicts() == [
        {"airport_id": 1, "training_endpoint_count": 2},
        {"airport_id": 2, "training_endpoint_count": 2},
        {"airport_id": 5, "training_endpoint_count": 1},
    ]


@pytest.mark.parametrize("missing", [False, True])
def test_training_manifest_requires_metadata_parts(tmp_path, monkeypatch, missing):
    monkeypatch.setattr(airport_exposure, "OUTPUT_DIR", tmp_path)
    manifest = tmp_path / "training.json"
    manifest.write_text(json.dumps({"inputs": {"train": [
        ["february-2026/2026-02-01/missing.parquet" if missing else "part.npz", 0, 0],
    ]}}))
    with pytest.raises(
        FileNotFoundError if missing else ValueError,
        match="Training airport metadata not found" if missing else "No training metadata",
    ):
        load_training_airports(manifest)


def _results():
    statuses = ["both"] * 5 + ["df1_only", "df0_only", "both"]
    takeoffs = [datetime(2026, 3, 1) + timedelta(hours=2 * i) for i in range(8)]
    landings = [takeoff + timedelta(hours=1) for takeoff in takeoffs]
    comparison = pl.DataFrame({
        "match_status": statuses,
        "takeoff_time": [
            None if status == "df1_only" else value
            for status, value in zip(statuses, takeoffs)
        ],
        "landing_time": [
            None if status == "df1_only" else value
            for status, value in zip(statuses, landings)
        ],
        "takeoff_time_df1": [
            None if status == "df0_only" else value
            for status, value in zip(statuses, takeoffs)
        ],
        "landing_time_df1": [
            None if status == "df0_only" else value
            for status, value in zip(statuses, landings)
        ],
        "takeoff_airport_id": [1, 1, None, 99, 1, None, 3, None],
        "takeoff_airport_id_df1": [1, 1, 1, 2, 3, 1, None, None],
        "landing_airport_id": [3, 2, None, 1, None, None, 1, 2],
        "landing_airport_id_df1": [3, 1, 3, 4, None, 4, None, 2],
    })
    comparison = comparison.with_columns([
        (
            (pl.col("match_status") == "both")
            & (pl.col(f"{endpoint}_airport_id") == pl.col(f"{endpoint}_airport_id_df1"))
        ).fill_null(False).alias(f"same_{endpoint}_airport_id")
        for endpoint in ("takeoff", "landing")
    ]).with_columns(
        (pl.col("same_takeoff_airport_id") & pl.col("same_landing_airport_id"))
        .alias("same_airport_id")
    )
    stats = EvaluationStats.from_comparison(
        comparison,
        test_date=date(2026, 3, 1),
        adsb_source=ADSBSource.ADSBLOL,
        test_source="algorithm-nn",
        gold_source="sfdps",
        gold_icao_count=1,
        gold_icao_missing_from_test_count=0,
    )
    return EvaluationResults(entries=(EvaluationResult(comparison, stats),))


def _row(report, endpoint, group):
    return report.filter(
        (pl.col("endpoint") == endpoint) & (pl.col("airport_group") == group)
    ).row(0, named=True)


def test_exposure_uses_gold_endpoint_and_explicit_denominators(training_manifest):
    training = load_training_airports(training_manifest)
    results = _results()
    original = results.comparisons[0].clone()
    report = airport_exposure_report(results, training)
    assert report.schema == AIRPORT_EXPOSURE_SCHEMA
    assert report.height == 6
    assert results.comparisons[0].equals(original)
    row = _row(report, "takeoff", "training")
    expected = {
        "gold_endpoint_count": 5,
        "gold_airport_count": 2,
        "matched_endpoint_count": 4,
        "matched_airport_count": 2,
        "correct_count": 2,
        "incorrect_count": 2,
        "null_prediction_count": 1,
        "airport_accuracy": 0.5,
        "airport_macro_accuracy": 1 / 3,
        "flight_match_coverage": 0.8,
        "correct_endpoint_recall": 0.4,
    }
    assert {name: row[name] for name in expected} == pytest.approx(expected)
    assert row["training_manifest_sha256"] == training.manifest_sha256

    row = _row(report, "takeoff", "nontraining")
    assert row["gold_endpoint_count"] == 1
    assert row["incorrect_count"] == 1  # Predicting a training airport cannot change the group.
    assert row["airport_accuracy"] == 0.0

    row = _row(report, "landing", "training")
    assert row["matched_endpoint_count"] == 2
    assert row["airport_accuracy"] == 0.5
    row = _row(report, "landing", "nontraining")
    assert row["gold_endpoint_count"] == 4
    assert row["matched_endpoint_count"] == 3
    assert row["correct_count"] == 1
    assert row["incorrect_count"] == 2
    assert row["null_prediction_count"] == 1
    assert row["airport_accuracy"] == pytest.approx(1 / 3)
    assert row["airport_macro_accuracy"] == 0.25
    assert row["flight_match_coverage"] == 0.75
    assert row["correct_endpoint_recall"] == 0.25

    for endpoint in ("takeoff", "landing"):
        row = _row(report, endpoint, "unknown")
        assert row["gold_endpoint_count"] == 1  # Excludes the test-only row.
        assert row["gold_airport_count"] == 0
        assert row["null_prediction_count"] == 1
        assert row["flight_match_coverage"] == 1.0
        for name in ("correct_count", "incorrect_count", "airport_accuracy",
                     "airport_macro_accuracy", "correct_endpoint_recall"):
            assert row[name] is None


def test_empty_groups_and_unmatched_gold_have_no_accuracy(training_manifest):
    training = load_training_airports(training_manifest)
    entry = _results().entries[0]
    only_missed = replace(
        entry, comparison=entry.comparison.filter(pl.col("match_status") == "df1_only"),
    )
    report = airport_exposure_report(EvaluationResults((only_missed,)), training)
    missed = _row(report, "takeoff", "training")
    assert missed["matched_endpoint_count"] == 0
    assert missed["null_prediction_count"] == 0
    assert missed["airport_accuracy"] is None
    assert missed["airport_macro_accuracy"] is None
    assert missed["flight_match_coverage"] == 0.0
    assert missed["correct_endpoint_recall"] == 0.0
    empty = _row(report, "takeoff", "nontraining")
    assert empty["gold_endpoint_count"] == 0
    assert empty["flight_match_coverage"] is None
    assert empty["correct_endpoint_recall"] is None
    assert airport_exposure_report(EvaluationResults(()), training).schema == report.schema


def test_no_eligible_training_labels_leaves_known_airports_nontraining(training_manifest, training_metadata_dir):
    part = training_metadata_dir / "part-0000.parquet"
    pl.read_parquet(part).with_columns(pl.lit(False).alias("eligible")).write_parquet(part)
    training = load_training_airports(training_manifest)
    assert training.airport_counts.is_empty()
    report = airport_exposure_report(_results(), training)
    assert _row(report, "takeoff", "training")["gold_endpoint_count"] == 0
    assert _row(report, "takeoff", "nontraining")["gold_endpoint_count"] == 6


def test_report_keeps_dates_sources_and_input_windows_separate(training_manifest):
    training = load_training_airports(training_manifest)
    entry = _results().entries[0]
    other = replace(entry, stats=replace(
        entry.stats, test_date=date(2026, 3, 2), gold_source="bts",
        adsb_source=ADSBSource.ADSBX, target_day_adsb=True,
    ))
    report = airport_exposure_report(EvaluationResults((entry, other)), training)
    assert report.height == 12
    assert report.select("test_date", "gold_src", "adsb_src", "target_day_adsb").unique().sort(
        "test_date"
    ).rows() == [
        ("2026-03-01", "sfdps", "adsblol", False),
        ("2026-03-02", "bts", "adsbx", True),
    ]


@pytest.mark.parametrize("enabled", [False, True])
def test_cli_training_report_is_optional(monkeypatch, tmp_path, training_manifest, enabled):
    output = tmp_path / "evaluation.csv"
    monkeypatch.setattr(evaluation, "evaluate", lambda **kwargs: _results())
    monkeypatch.setattr(evaluation, "default_stats_path", lambda: output)
    args = ["--test-source", "algorithm-nn", "--columns", "test_date", "precision"]
    if enabled:
        args += ["--airport-nn-training", str(training_manifest)]
    else:
        def unexpected_load(*args):
            raise AssertionError("Training files should only be read when requested")
        monkeypatch.setattr(evaluation, "load_training_airports", unexpected_load)
    evaluation.main(args)
    assert pl.read_csv(output).columns == ["test_date", "precision"]
    exposure = tmp_path / "evaluation_airport_exposure.csv"
    membership = tmp_path / "evaluation_training_airports.csv"
    assert exposure.exists() is enabled
    assert membership.exists() is enabled
    if enabled:
        assert pl.read_csv(exposure).height == 6
        assert pl.read_csv(membership)["airport_id"].to_list() == [1, 2]


def test_cli_rejects_exposure_with_stats_only_cache_listing(training_manifest, capsys):
    with pytest.raises(SystemExit):
        evaluation.parse_args(["--list-cache", "--airport-nn-training", str(training_manifest)])
    assert "requires detailed comparisons" in capsys.readouterr().err
