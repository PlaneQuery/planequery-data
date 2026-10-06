from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from airports.models import AirportCatalogProvenance
from data_engineering.adsb.read_adsb import ADSBSource
from flights.airport_nn.source import AlgorithmNNFlightSource
from flights.evaluation import evaluation
from flights.evaluation.cache import (
    CachedEvaluationStats,
    EvaluationCache,
    RESULT_FILENAME,
    default_evaluation_cache_dir,
)
from flights.flight_type import Flight, flights_to_flights_df
from flights.sources import ALGORITHM_NEAREST_SOURCE, BTS_SOURCE


TARGET_DATE = date(2026, 3, 1)
NOON = datetime(2026, 3, 1, 12)  # noqa: DTZ001 - flight times are naive UTC


def _airport_catalog(*, with_provenance: bool = True):
    catalog = SimpleNamespace(
        airports=(
            SimpleNamespace(id=1, lat=40.0, lon=-75.0),
            SimpleNamespace(id=2, lat=41.0, lon=-74.0),
        )
    )
    if with_provenance:
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


def _patch_inputs(monkeypatch, test_source, frame_for_source, *, catalog=None):
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
        lambda target_date: catalog or _airport_catalog(),
    )
    return reads


def _evaluate(test_source, cache, **kwargs):
    return evaluation.evaluate(
        test_source=test_source,
        gold_source=BTS_SOURCE,
        test_dates=TARGET_DATE,
        adsb_source=ADSBSource.ADSBLOL,
        write_outputs=False,
        evaluation_cache=cache,
        **kwargs,
    )


def test_default_cache_directory_is_under_output_dir(monkeypatch, tmp_path):
    from data_engineering import utils as data_engineering_utils

    monkeypatch.setattr(data_engineering_utils, "OUTPUT_DIR", tmp_path)

    assert default_evaluation_cache_dir() == (
        tmp_path / "data" / "cache" / "evaluation"
    )

    monkeypatch.setattr(evaluation, "OUTPUT_DIR", tmp_path)
    assert evaluation.default_evaluation_cache().root == (
        tmp_path / "data" / "cache" / "evaluation"
    )


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


def test_cache_listing_only_returns_the_active_cache_version(
    monkeypatch,
    tmp_path,
):
    test_source = _algorithm_nn_source(tmp_path)
    cache_root = tmp_path / "cache"
    old_cache = EvaluationCache(cache_root, cache_version=2)
    current_cache = EvaluationCache(cache_root, cache_version=3)
    frame = flights_to_flights_df([_flight()])
    _patch_inputs(monkeypatch, test_source, lambda source: frame)

    _evaluate(test_source, old_cache)

    assert old_cache.list_entries().height == 1
    assert current_cache.list_entries().is_empty()


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
        write_outputs=False,
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


def test_catalog_without_provenance_does_not_create_cache_files(
    monkeypatch,
    tmp_path,
):
    test_source = _algorithm_nn_source(tmp_path)
    cache_root = tmp_path / "cache"
    cache = EvaluationCache(cache_root)
    frame = flights_to_flights_df([_flight()])
    reads = _patch_inputs(
        monkeypatch,
        test_source,
        lambda source: frame,
        catalog=_airport_catalog(with_provenance=False),
    )

    _evaluate(test_source, cache)

    assert len(reads) == 2
    assert not list(cache_root.rglob("*"))
    assert cache.list_entries().is_empty()


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


def test_evaluate_writes_selected_columns_but_returns_full_stats(
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
        output_csv=output_csv,
        output_columns=["gold_src", "precision", "test_date"],
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
        write_outputs=False,
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


def test_parse_args_accepts_csv_output_options(tmp_path):
    output_csv = tmp_path / "evaluation.csv"

    args = evaluation.parse_args(
        [
            "--output-csv",
            str(output_csv),
            "--columns",
            "test_date",
            "precision",
            "recall",
        ]
    )

    assert args.output_csv == Path(output_csv)
    assert args.columns == ["test_date", "precision", "recall"]


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
        "write_outputs": False,
    }
    values[empty_parameter] = []

    with pytest.raises(ValueError, match=empty_parameter):
        evaluation.evaluate(**values)


@pytest.mark.parametrize(
    ("option", "value"),
    [
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
    ],
)
def test_parse_args_rejects_removed_options(option, value):
    argv = [option] if value is None else [option, value]
    with pytest.raises(SystemExit):
        evaluation.parse_args(argv)
