import json
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from airports.models import AirportCatalogProvenance
from data_engineering.adsb.read_adsb import ADSBSource
from flights.airport_nn.source import AlgorithmNNFlightSource
from flights.evaluation import airport_exposure, evaluation
from flights.evaluation.airport_exposure import (
    airport_exposure_report,
    load_training_airports,
)
from flights.evaluation.cache import (
    CachedEvaluationStats,
    EvaluationCache,
    RESULT_FILENAME,
    STATS_FILENAME,
    default_evaluation_cache_dir,
)
from flights.flight_type import Flight, flights_to_flights_df
from flights.sources import (
    ALGORITHM_NEAREST_SOURCE,
    BTS_SOURCE,
    BTS_VERIFIED_SOURCE,
    EUROCONTROL_VERIFIED_SOURCE,
)


TARGET_DATE = date(2026, 3, 1)
NOON = datetime(2026, 3, 1, 12)  # noqa: DTZ001 - flight times are naive UTC


def _airport_catalog():
    catalog = SimpleNamespace(
        airports=(
            SimpleNamespace(id=1, lat=40.0, lon=-75.0),
            SimpleNamespace(id=2, lat=41.0, lon=-74.0),
        )
    )
    catalog.provenance = AirportCatalogProvenance(
        flight_date=TARGET_DATE,
        observed_commit_sha="observed-commit",
        knowledge_cutoff_date=date(2026, 4, 1),
        knowledge_commit_sha="knowledge-commit",
        catalog_available_date=date(2026, 4, 1),
        policy_version="test-policy",
    )
    return catalog


def _flight(
    *,
    takeoff_airport_id: int = 1,
    landing_airport_id: int = 2,
):
    return Flight(
        icao="abc123",
        takeoff_time=NOON,
        landing_time=NOON + timedelta(hours=1),
        takeoff_airport_ident="KAAA",
        landing_airport_ident="KBBB",
        takeoff_airport_id=takeoff_airport_id,
        landing_airport_id=landing_airport_id,
    )


def _algorithm_nn_source(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"test checkpoint")
    return AlgorithmNNFlightSource(model_path=checkpoint)


def _patch_inputs(monkeypatch, test_source, frame_for_source):
    reads = []

    def source_flights_reader(source, adsb_source=None):
        def read(target_date):
            reads.append((source.name, adsb_source, target_date))
            return frame_for_source(source).clone()

        return read

    monkeypatch.setattr(
        evaluation,
        "source_flights_reader",
        source_flights_reader,
    )
    monkeypatch.setattr(
        evaluation,
        "airport_catalog_for_date",
        lambda target_date: _airport_catalog(),
    )
    return reads


def _evaluate(test_source, cache, **kwargs):
    return evaluation.evaluate(
        test_source=test_source,
        gold_source=BTS_SOURCE,
        test_dates=TARGET_DATE,
        adsb_source=ADSBSource.ADSBLOL,
        evaluation_cache=cache,
        **kwargs,
    )


def test_default_cache_directory_is_under_output_dir(monkeypatch, tmp_path):
    from flights.evaluation import cache as evaluation_cache

    monkeypatch.setattr(evaluation_cache, "OUTPUT_DIR", tmp_path)

    assert default_evaluation_cache_dir() == (
        tmp_path / "data" / "cache" / "evaluation"
    )

    monkeypatch.setattr(evaluation, "OUTPUT_DIR", tmp_path)
    assert evaluation.default_evaluation_cache().root == (
        tmp_path / "data" / "cache" / "evaluation"
    )


def test_verified_source_cache_key_tracks_selected_adsb_provider(tmp_path):
    cache = EvaluationCache(tmp_path / "cache")
    catalog = _airport_catalog()

    opensky = cache.make_key(
        TARGET_DATE,
        BTS_SOURCE,
        BTS_VERIFIED_SOURCE,
        ADSBSource.OPENSKY,
        catalog,
    )
    adsblol = cache.make_key(
        TARGET_DATE,
        BTS_SOURCE,
        BTS_VERIFIED_SOURCE,
        ADSBSource.ADSBLOL,
        catalog,
    )

    assert opensky.adsb_source == "opensky"
    assert opensky.adsb_parquet_version == 4
    assert adsblol.adsb_source == "adsblol"
    assert adsblol.adsb_parquet_version == 6
    assert cache.path_for(opensky) != cache.path_for(adsblol)


def test_airport_exposure_reuses_cached_comparison(monkeypatch, tmp_path):
    monkeypatch.setattr(airport_exposure, "OUTPUT_DIR", tmp_path)
    source = _algorithm_nn_source(tmp_path)
    _patch_inputs(monkeypatch, source, lambda _: flights_to_flights_df([_flight()]))
    cache = EvaluationCache(tmp_path / "cache")
    fresh = _evaluate(source, cache)

    part = tmp_path / "data/processed/flights_airport_nn/february-2026/2026-02-01/part-0000.parquet"
    part.parent.mkdir(parents=True)
    pl.DataFrame({"truth_id": [1], "eligible": [True], "target": [0]}).write_parquet(part)
    manifest = tmp_path / "training.json"
    manifest.write_text(json.dumps({"inputs": {"train": [[str(part.relative_to(tmp_path)), 0, 0]]}}))
    training = load_training_airports(manifest)

    def unexpected_read(*args, **kwargs):
        raise AssertionError("Exposure reporting must reuse cached flights")

    monkeypatch.setattr(evaluation, "source_flights_reader", unexpected_read)
    cached = _evaluate(source, cache)
    assert airport_exposure_report(cached, training).equals(
        airport_exposure_report(fresh, training)
    )
    assert cached.stats == fresh.stats


def test_identical_evaluation_uses_cached_comparison_and_stats(
    monkeypatch,
    tmp_path,
):
    test_source = _algorithm_nn_source(tmp_path)
    cache = EvaluationCache(tmp_path / "cache")
    frame = flights_to_flights_df([_flight()])
    reads = _patch_inputs(monkeypatch, test_source, lambda source: frame)

    first_results = _evaluate(test_source, cache)
    first_read_count = len(reads)
    second_results = _evaluate(test_source, cache)

    assert first_read_count == 2
    assert len(reads) == first_read_count
    assert len(first_results.entries) == len(second_results.entries) == 1
    assert first_results.comparisons[0].equals(
        second_results.comparisons[0]
    )
    assert first_results.stats == second_results.stats
    assert first_results.stats[0].precision == 1.0


@pytest.mark.parametrize("verified", [False, True])
def test_target_day_adsb_isolates_evaluation_cache_and_labels_output(
    monkeypatch, tmp_path, verified,
):
    source = EUROCONTROL_VERIFIED_SOURCE if verified else _algorithm_nn_source(tmp_path)
    cache = EvaluationCache(tmp_path / "cache")
    frame = flights_to_flights_df([_flight()])
    configured_sources = []

    def frame_for_source(resolved):
        configured_sources.append(resolved)
        return frame

    reads = _patch_inputs(monkeypatch, source, frame_for_source)
    ordinary = _evaluate(source, cache)
    single_day = _evaluate(source, cache, target_day_adsb=True)
    repeated = _evaluate(evaluation._with_target_day_adsb([source], True)[0], cache)

    assert len(reads) == 4
    assert configured_sources[0].target_day_adsb is False
    assert configured_sources[2].target_day_adsb is True
    if not verified:
        assert configured_sources[2].model_path == source.model_path
    assert ordinary.stats[0].target_day_adsb is False
    assert single_day.stats[0].target_day_adsb is True
    assert repeated.stats == single_day.stats
    assert cache.list_entries().height == 2
    listed = evaluation.list_cached_evaluations(
        target_day_adsb=True, evaluation_cache=cache,
    )
    assert listed.height == 1
    assert listed["target_day_adsb"].to_list() == [True]


@pytest.mark.parametrize("gold_source", [ALGORITHM_NEAREST_SOURCE, EUROCONTROL_VERIFIED_SOURCE])
def test_target_day_adsb_applies_to_adsb_gold_source(monkeypatch, tmp_path, gold_source):
    source = _algorithm_nn_source(tmp_path)
    cache = EvaluationCache(tmp_path / "cache")
    frame = flights_to_flights_df([_flight()])
    sources = []
    _patch_inputs(monkeypatch, source, lambda resolved: sources.append(resolved) or frame)
    results = evaluation.evaluate(
        test_source=source, gold_source=gold_source,
        test_dates=TARGET_DATE, adsb_source=list(ADSBSource),
        target_day_adsb=True, evaluation_cache=cache,
    )
    assert len(sources) == 6
    assert all(resolved.target_day_adsb for resolved in sources)
    assert {stats.adsb_source for stats in results.stats} == set(ADSBSource)
    assert all(stats.target_day_adsb for stats in results.stats)


def test_target_day_adsb_is_an_optional_flag():
    assert evaluation.parse_args([]).target_day_adsb is False
    assert evaluation.parse_args(["--target-day-adsb"]).target_day_adsb is True


@pytest.mark.parametrize("value", ["0", "1", "true"])
def test_target_day_adsb_rejects_values(value):
    with pytest.raises(SystemExit):
        evaluation.parse_args(["--target-day-adsb", value])


@pytest.mark.parametrize("unreadable_file", [RESULT_FILENAME, STATS_FILENAME])
def test_unreadable_entry_is_a_cache_miss(
    monkeypatch,
    tmp_path,
    unreadable_file,
):
    test_source = _algorithm_nn_source(tmp_path)
    cache = EvaluationCache(tmp_path / "cache")
    frame = flights_to_flights_df([_flight()])
    reads = _patch_inputs(monkeypatch, test_source, lambda source: frame)

    _evaluate(test_source, cache)
    next(cache.root.rglob(unreadable_file)).write_bytes(b"not parquet")
    assert not cache.find_stats()
    _evaluate(test_source, cache)

    assert len(reads) == 4
    assert cache.find_stats()


def test_two_adsb_providers_are_cached_and_listed_with_filters(
    monkeypatch,
    tmp_path,
):
    test_source = _algorithm_nn_source(tmp_path)
    cache = EvaluationCache(tmp_path / "cache")
    frame = flights_to_flights_df([_flight()])
    _patch_inputs(monkeypatch, test_source, lambda source: frame)

    evaluation.evaluate(
        test_source=test_source,
        gold_source=BTS_SOURCE,
        test_dates=TARGET_DATE,
        adsb_source=[ADSBSource.ADSBLOL, ADSBSource.ADSBX],
        evaluation_cache=cache,
    )

    listed = evaluation.list_cached_evaluations(evaluation_cache=cache)
    assert listed.height == 2
    assert set(listed.get_column("adsb_src")) == {"adsblol", "adsbx"}

    filtered = evaluation.list_cached_evaluations(
        test_source=test_source.name,
        gold_source=BTS_SOURCE.name,
        adsb_source=ADSBSource.ADSBX,
        evaluation_cache=cache,
    )
    assert filtered.height == 1
    assert filtered.row(0, named=True)["adsb_src"] == "adsbx"

    assert evaluation.list_cached_evaluations(
        test_source=BTS_SOURCE.name,
        gold_source=BTS_SOURCE.name,
        adsb_source=ADSBSource.ADSBX,
        evaluation_cache=cache,
    ).is_empty()


def test_cache_listing_reads_stats_without_loading_full_comparisons(
    monkeypatch,
    tmp_path,
):
    test_source = _algorithm_nn_source(tmp_path)
    cache = EvaluationCache(tmp_path / "cache")
    frame = flights_to_flights_df([_flight()])
    _patch_inputs(monkeypatch, test_source, lambda source: frame)
    _evaluate(test_source, cache)

    read_parquet = pl.read_parquet

    def read_stats_only(path, *args, **kwargs):
        assert Path(path).name != RESULT_FILENAME
        return read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pl, "read_parquet", read_stats_only)

    cached_stats = cache.find_stats()
    assert len(cached_stats) == 1
    assert isinstance(cached_stats[0], CachedEvaluationStats)
    assert cached_stats[0].stats.precision == 1.0
    assert cache.list_entries().height == 1


def test_refresh_cache_recomputes_and_overwrites_entry(monkeypatch, tmp_path):
    test_source = _algorithm_nn_source(tmp_path)
    cache = EvaluationCache(tmp_path / "cache")
    state = {"changed": False}

    def frame_for_source(source):
        if source.name == test_source.name and state["changed"]:
            return flights_to_flights_df(
                [_flight(takeoff_airport_id=10, landing_airport_id=20)]
            )
        return flights_to_flights_df([_flight()])

    reads = _patch_inputs(monkeypatch, test_source, frame_for_source)

    original_results = _evaluate(test_source, cache)
    state["changed"] = True
    refreshed_results = _evaluate(test_source, cache, refresh_cache=True)

    assert len(reads) == 4
    assert original_results.stats[0].airport_id_match_pct == 1.0
    assert refreshed_results.stats[0].airport_id_match_pct == 0.0

    listed = cache.list_entries(
        test_source=test_source.name,
        gold_source=BTS_SOURCE.name,
        adsb_source=ADSBSource.ADSBLOL,
    )
    assert listed.height == 1
    assert listed.row(0, named=True)["airport_id_match_pct"] == 0.0


def test_parse_args_accepts_hyphenated_source_options():
    args = evaluation.parse_args(
        [
            "--test-source",
            "algorithm-nn",
            "--gold-source",
            "bts",
            "--adsb-source",
            "adsblol",
        ]
    )

    assert args.test_source == ["algorithm-nn"]
    assert args.gold_source == ["bts"]
    assert args.adsb_source == [ADSBSource.ADSBLOL]


def test_select_output_columns_defaults_to_all_columns_and_preserves_order():
    frame = pl.DataFrame(
        {
            "test_date": [TARGET_DATE.isoformat()],
            "test_src": ["algorithm-nn"],
            "precision": [1.0],
        }
    )

    assert evaluation.select_output_columns(frame).equals(frame)

    selected = evaluation.select_output_columns(
        frame,
        ["precision", "test_date"],
    )
    assert selected.columns == ["precision", "test_date"]
    assert selected.row(0) == (1.0, TARGET_DATE.isoformat())


@pytest.mark.parametrize(
    ("columns", "message"),
    [
        (["precision", "missing"], "Unknown output columns"),
        (["precision", "precision"], "Duplicate output columns"),
    ],
)
def test_select_output_columns_rejects_invalid_requests(columns, message):
    frame = pl.DataFrame({"precision": [1.0], "recall": [1.0]})

    with pytest.raises(ValueError, match=message):
        evaluation.select_output_columns(frame, columns)


def test_csv_writer_selects_columns_from_evaluation_results(
    monkeypatch,
    tmp_path,
):
    test_source = _algorithm_nn_source(tmp_path)
    cache = EvaluationCache(tmp_path / "cache")
    frame = flights_to_flights_df([_flight()])
    _patch_inputs(monkeypatch, test_source, lambda source: frame)
    output_csv = tmp_path / "reports" / "selected.csv"

    results = evaluation.evaluate(
        test_source=test_source,
        gold_source=BTS_SOURCE,
        test_dates=TARGET_DATE,
        adsb_source=ADSBSource.ADSBLOL,
        evaluation_cache=cache,
    )
    evaluation.write_evaluation_csv(
        results.to_frame(),
        output_csv,
        columns=["gold_src", "precision", "test_date"],
    )

    written = pl.read_csv(output_csv)
    assert written.columns == ["gold_src", "precision", "test_date"]
    assert written.row(0) == ("bts", 1.0, TARGET_DATE.isoformat())
    assert results.stats[0].recall == 1.0
    assert results.stats[0].takeoff_airport_id_match_pct == 1.0
    assert results.to_frame().columns != written.columns


def test_export_cached_evaluations_filters_and_selects_columns(
    monkeypatch,
    tmp_path,
):
    test_source = _algorithm_nn_source(tmp_path)
    cache = EvaluationCache(tmp_path / "cache")
    frame = flights_to_flights_df([_flight()])
    _patch_inputs(monkeypatch, test_source, lambda source: frame)

    evaluation.evaluate(
        test_source=[test_source, ALGORITHM_NEAREST_SOURCE],
        gold_source=BTS_SOURCE,
        test_dates=TARGET_DATE,
        adsb_source=[ADSBSource.ADSBLOL, ADSBSource.ADSBX],
        evaluation_cache=cache,
    )
    output_csv = tmp_path / "exports" / "adsbx.csv"

    exported, path = evaluation.export_cached_evaluations(
        output_csv,
        columns=["test_date", "adsb_src", "test_src", "precision"],
        test_source=test_source.name,
        gold_source=BTS_SOURCE.name,
        adsb_source=ADSBSource.ADSBX,
        evaluation_cache=cache,
    )

    assert path == output_csv
    assert exported.columns == [
        "test_date",
        "adsb_src",
        "test_src",
        "precision",
    ]
    assert exported.rows() == [
        (TARGET_DATE.isoformat(), "adsbx", "algorithm-nn", 1.0)
    ]
    assert pl.read_csv(output_csv).equals(exported)


def test_empty_cache_export_supports_metric_columns(tmp_path):
    cache = EvaluationCache(tmp_path / "cache")
    output_csv = tmp_path / "empty.csv"

    exported, path = evaluation.export_cached_evaluations(
        output_csv,
        columns=["test_date", "precision", "recall"],
        test_source="algorithm-nn",
        gold_source="bts",
        evaluation_cache=cache,
    )

    assert path == output_csv
    assert exported.is_empty()
    assert exported.columns == ["test_date", "precision", "recall"]
    assert pl.read_csv(output_csv).columns == exported.columns


def test_parse_args_accepts_output_columns():
    args = evaluation.parse_args(
        [
            "--columns",
            "test_date",
            "precision",
            "recall",
        ]
    )

    assert args.columns == ["test_date", "precision", "recall"]


def test_cli_writes_csv(monkeypatch, tmp_path):
    evaluate_kwargs = {}
    output_csv = tmp_path / "evaluation.csv"

    def evaluate(**kwargs):
        evaluate_kwargs.update(kwargs)
        return SimpleNamespace(
            to_frame=lambda: pl.DataFrame(
                {"precision": [1.0], "recall": [1.0]}
            )
        )

    monkeypatch.setattr(
        evaluation,
        "evaluate",
        evaluate,
    )
    monkeypatch.setattr(evaluation, "default_stats_path", lambda: output_csv)

    evaluation.main(
        [
            "--target-day-adsb",
            "--test-dates",
            TARGET_DATE.isoformat(),
            "--columns",
            "precision",
        ]
    )

    assert "write_csv" not in evaluate_kwargs
    assert evaluate_kwargs["target_day_adsb"] is True
    assert pl.read_csv(output_csv).to_dict(as_series=False) == {
        "precision": [1.0]
    }


@pytest.mark.parametrize(
    "empty_parameter",
    ["test_source", "gold_source", "test_dates", "adsb_source"],
)
def test_evaluate_rejects_empty_input_sequences(empty_parameter):
    values = {
        "test_source": [ALGORITHM_NEAREST_SOURCE],
        "gold_source": [BTS_SOURCE],
        "test_dates": [TARGET_DATE],
        "adsb_source": [ADSBSource.ADSBLOL],
    }
    values[empty_parameter] = []

    with pytest.raises(ValueError, match=empty_parameter):
        evaluation.evaluate(**values)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--adsb-context-days", "0"),
        ("--test_src", "algorithm-nn"),
        ("--test-src", "algorithm-nn"),
        ("--gold_src", "bts"),
        ("--gold-src", "bts"),
        ("--adsb_src", "adsblol"),
        ("--adsb-src", "adsblol"),
        ("--matching-adsb-src", "adsbx"),
        ("--pia-or-american-ladd-only", None),
        ("--filter_rotorcraft", None),
        ("--include-consecutive-airport-metrics", None),
        ("--consecutive-airport-displacement-threshold-km", "10"),
        ("--use_all_icaos", None),
        ("--no-cache", None),
        ("--output-csv", "/tmp/evaluation.csv"),
    ],
)
def test_parse_args_rejects_removed_options(option, value):
    argv = [option] if value is None else [option, value]
    with pytest.raises(SystemExit):
        evaluation.parse_args(argv)
