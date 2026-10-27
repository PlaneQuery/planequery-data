from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Callable

import polars as pl

from airports.catalog import AirportCatalogView
from airports.runtime import airport_catalog_for_date
from data_engineering.adsb.read_adsb import ADSBSource
from data_engineering.utils import OUTPUT_DIR
from flights.airport_nn import AlgorithmNNFlightSource, OpenSkyNNFlightSource
from flights.evaluation.airport_exposure import (
    airport_exposure_report,
    load_training_airports,
)
from flights.evaluation.cache import EvaluationCache
from flights.evaluation.models import (
    EVALUATION_METADATA_COLUMNS,
    EvaluationResult,
    EvaluationResults,
    EvaluationStats,
)
from flights.flight_source import ADSBFlightSource, FlightSource, get_flights
from flights.flights_comparison import MAX_FLIGHT_MATCH_TIME_DELTA, df_flights_comparison
from flights.sources import (
    ALGORITHM_NEAREST_SOURCE,
    BTS_SOURCE,
    FLIGHT_SOURCES_BY_NAME,
)
from flights.verified.source import VerifiedFlightSource
from planequery_data.utils import current_commit_hash


DEFAULT_TARGET_DATE = date(2026, 3, 1)
DEFAULT_ADSB_SOURCE = ADSBSource.ADSBLOL

SourceInput = FlightSource | str


def _uses_selected_adsb_source(source: FlightSource) -> bool:
    return isinstance(source, ADSBFlightSource)


def default_stats_path() -> Path:
    run_time = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S-%f")
    commit_hash = current_commit_hash()
    filename = f"{run_time}_{commit_hash}_flights_evaluation.csv"
    return OUTPUT_DIR / "data" / "experiments" / "comparison" / filename


def default_evaluation_cache() -> EvaluationCache:
    return EvaluationCache(OUTPUT_DIR / "data" / "cache" / "evaluation")


def select_output_columns(
    frame: pl.DataFrame,
    columns: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Select CSV columns in the requested order, or keep every column."""
    if columns is None:
        return frame
    requested = list(columns)
    if not requested:
        raise ValueError("At least one output column is required")
    duplicates = sorted(
        column for column in set(requested) if requested.count(column) > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate output columns: {duplicates}")
    missing = [column for column in requested if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Unknown output columns: {missing}. "
            f"Available columns: {frame.columns}"
        )
    return frame.select(requested)


def write_evaluation_csv(
    frame: pl.DataFrame,
    output_path: Path,
    *,
    columns: Sequence[str] | None = None,
) -> Path:
    """Write selected evaluation columns; all columns are the default."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    select_output_columns(frame, columns).write_csv(output_path)
    return output_path


def resolve_source(source: SourceInput) -> FlightSource:
    if isinstance(source, str):
        try:
            return FLIGHT_SOURCES_BY_NAME[source]
        except KeyError:
            raise ValueError(f"Unknown flight source: {source}") from None
    return source


def source_flights_reader(
    source: SourceInput,
    adsb_source: ADSBSource | None = DEFAULT_ADSB_SOURCE,
) -> Callable[[date], pl.DataFrame]:
    source = resolve_source(source)
    if _uses_selected_adsb_source(source):
        if adsb_source is None:
            raise ValueError(f"{source.name} requires an ADS-B source")
        source_adsb = ADSBSource(adsb_source)
    else:
        source_adsb = None

    def read_source(target_date: date) -> pl.DataFrame:
        if (
            isinstance(source, (AlgorithmNNFlightSource, OpenSkyNNFlightSource))
            and not source.model_path.is_file()
        ):
            raise FileNotFoundError(
                f"Airport NN checkpoint not found: {source.model_path}. "
                "Pass --airport-nn-model with an existing checkpoint."
            )
        return get_flights(source, target_date, source_adsb)

    return read_source


def adsb_sources_for_evaluation(
    test_source: SourceInput,
    gold_source: SourceInput,
    adsb_sources: Sequence[ADSBSource],
) -> list[ADSBSource | None]:
    """Return provider variants that can affect this source pair."""
    if _uses_selected_adsb_source(
        resolve_source(test_source)
    ) or _uses_selected_adsb_source(
        resolve_source(gold_source)
    ):
        return list(adsb_sources)
    return [None]


def _as_source_list(value: SourceInput | Sequence[SourceInput]) -> list[FlightSource]:
    if isinstance(value, (FlightSource, str)):
        value = [value]
    return [resolve_source(source) for source in value]


def _as_date_list(value: date | Sequence[date]) -> list[date]:
    if isinstance(value, date):
        return [value]
    return list(value)


def _as_adsb_source_list(
    value: ADSBSource | str | Sequence[ADSBSource | str],
) -> list[ADSBSource]:
    if isinstance(value, (ADSBSource, str)):
        value = [value]
    return [ADSBSource(source) for source in value]


def _with_airport_nn_model(
    sources: Sequence[FlightSource],
    model_path: Path | None,
) -> list[FlightSource]:
    if model_path is None:
        return list(sources)
    return [
        replace(source, model_path=model_path)
        if isinstance(source, (AlgorithmNNFlightSource, OpenSkyNNFlightSource))
        else source
        for source in sources
    ]


def _with_target_day_adsb(
    sources: Sequence[FlightSource],
    enabled: bool,
) -> list[FlightSource]:
    if not enabled:
        return list(sources)
    return [
        VerifiedFlightSource(
            source.source, target_day_adsb=True
        )
        if isinstance(source, VerifiedFlightSource)
        else replace(source, target_day_adsb=True)
        if isinstance(source, ADSBFlightSource)
        else source
        for source in sources
    ]


def _flights_on_date(flights: pl.DataFrame, target_date: date) -> pl.DataFrame:
    return flights.filter(
        (pl.col("takeoff_time").dt.date() == target_date)
    )


def _assert_no_null_icaos(flights: pl.DataFrame, source_role: str) -> None:
    null_count = flights.get_column("icao").null_count()
    assert null_count == 0, (
        f"{source_role} contains {null_count} null icao value(s)"
    )


def _format_minutes(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f} min"


def _gold_icao_cohort(
    test_flights: pl.DataFrame,
    gold_flights: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Keep every gold ICAO and restrict test flights to that gold cohort."""
    gold_icaos = set(
        gold_flights.get_column("icao").unique().to_list()
    )
    return (
        test_flights.filter(pl.col("icao").is_in(sorted(gold_icaos))),
        gold_flights,
    )


def _compare_day(
    test_flights: pl.DataFrame,
    gold_flights: pl.DataFrame,
    start_dt: datetime,
    end_dt: datetime,
    airport_catalog: AirportCatalogView,
) -> pl.DataFrame:
    _assert_no_null_icaos(test_flights, "test_source")
    _assert_no_null_icaos(gold_flights, "gold_source")
    assert test_flights.select(
    (pl.col("takeoff_time").dt.date() == start_dt.date())
    .fill_null(False)
    .all()
    ).item(), f"Not all test flights take off on {start_dt.date()}"


    test_flights, gold_flights = _gold_icao_cohort(
        test_flights,
        gold_flights,
    )
    assert not gold_flights.is_empty()
    assert not test_flights.is_empty()

    assert gold_flights.select(
      (pl.col("takeoff_time").dt.date() == start_dt.date())
      .fill_null(False)
      .all()
    ).item(), f"Expected all flights to take off on {start_dt.date()}"

    comparison = df_flights_comparison(
        test_flights,
        gold_flights,
        start_dt,
        end_dt,
        airport_catalog,
    )
    # Keep full-day gold as matching context, then score by gold departure
    # when present, or test departure for an unmatched test flight.
    return comparison.filter(
        pl.coalesce("takeoff_time_df1", "takeoff_time").is_between(
            start_dt, end_dt, closed="both"
        )
    )

def get_start_dt_end_dt_for_gold(target_date: date) -> tuple[datetime, datetime]:
    start_dt = datetime.combine(target_date, datetime.min.time()) + MAX_FLIGHT_MATCH_TIME_DELTA
    end_dt = datetime.combine(target_date + timedelta(days=1), datetime.min.time()) - MAX_FLIGHT_MATCH_TIME_DELTA
    return start_dt, end_dt

def evaluate(
    test_source: SourceInput | Sequence[SourceInput],
    gold_source: SourceInput | Sequence[SourceInput],
    test_dates: date | Sequence[date],
    adsb_source: (
        ADSBSource | str | Sequence[ADSBSource | str]
    ) = DEFAULT_ADSB_SOURCE,
    *,
    airport_nn_model_path: Path | None = None,
    target_day_adsb: bool = False,
    refresh_cache: bool = False,
    evaluation_cache: EvaluationCache | None = None,
) -> EvaluationResults:
    """Compare flight sources, reusing one persistent cache entry per day.

    Each comparison keeps all gold-source ICAOs and restricts test flights to
    ICAOs present in gold. Detailed comparisons and their daily stats are
    cached together. Set
    ``refresh_cache=True`` to recompute and replace matching entries.

    ``target_day_adsb=True`` limits ADS-B inputs to each target UTC day before
    segmentation and inference. False preserves provider-specific windows.
    Target-day-only inputs use separate flight and evaluation cache entries.
    """
    test_sources = _with_airport_nn_model(
        _as_source_list(test_source),
        airport_nn_model_path,
    )
    gold_sources = _with_airport_nn_model(
        _as_source_list(gold_source),
        airport_nn_model_path,
    )
    test_sources = _with_target_day_adsb(test_sources, target_day_adsb)
    gold_sources = _with_target_day_adsb(gold_sources, target_day_adsb)
    dates = _as_date_list(test_dates)
    adsb_sources = _as_adsb_source_list(adsb_source)
    for parameter, values in (
        ("test_source", test_sources),
        ("gold_source", gold_sources),
        ("test_dates", dates),
        ("adsb_source", adsb_sources),
    ):
        if not values:
            raise ValueError(f"{parameter} must contain at least one value")
    cache = evaluation_cache or default_evaluation_cache()

    daily_results: list[EvaluationResult] = []
    airport_catalogs: dict[date, AirportCatalogView] = {}
    flight_frames: dict[
        tuple[FlightSource, ADSBSource | None, date], pl.DataFrame
    ] = {}

    def read_flights(
        source: FlightSource,
        source_adsb: ADSBSource | None,
        target_date: date,
    ) -> pl.DataFrame:
        frame_key = (source, source_adsb, target_date)
        if frame_key not in flight_frames:
            flight_frames[frame_key] = source_flights_reader(
                source,
                adsb_source=source_adsb,
            )(target_date)
        return flight_frames[frame_key]

    for resolved_test_source in test_sources:
        for resolved_gold_source in gold_sources:
            input_settings = {
                source.target_day_adsb
                for source in (resolved_test_source, resolved_gold_source)
                if isinstance(source, (ADSBFlightSource, VerifiedFlightSource))
            }
            if len(input_settings) > 1:
                raise ValueError(
                    "ADS-B test and gold sources must use the same "
                    "target_day_adsb setting; pass target_day_adsb=True to override both"
                )
            pair_target_day_adsb = next(iter(input_settings), False)
            pair_adsb_sources = adsb_sources_for_evaluation(
                resolved_test_source,
                resolved_gold_source,
                adsb_sources,
            )
            for selected_adsb_source in pair_adsb_sources:
                test_adsb_source = (
                    selected_adsb_source
                    if _uses_selected_adsb_source(resolved_test_source)
                    else None
                )
                gold_adsb_source = (
                    selected_adsb_source
                    if _uses_selected_adsb_source(resolved_gold_source)
                    else None
                )
                for target_date in dates:
                    if target_date not in airport_catalogs:
                        airport_catalogs[target_date] = airport_catalog_for_date(
                            target_date
                        )
                    airport_catalog = airport_catalogs[target_date]
                    cache_key = cache.make_key(
                        test_date=target_date,
                        test_source=resolved_test_source,
                        gold_source=resolved_gold_source,
                        adsb_source=selected_adsb_source,
                        airport_catalog=airport_catalog,
                    )

                    cached = None
                    if not refresh_cache:
                        cached = cache.load(cache_key)
                    if cached is not None:
                        comparison = cached.comparison
                        stats = cached.stats
                        print(
                            "Using cached evaluation for "
                            f"{target_date} {resolved_test_source.name}:"
                            f"{resolved_gold_source.name} "
                            f"({selected_adsb_source or 'no ADS-B source'})"
                        )
                    else:
                        test_flights = read_flights(
                            resolved_test_source, test_adsb_source, target_date
                        )
                        gold_flights = read_flights(
                            resolved_gold_source, gold_adsb_source, target_date
                        )
                        start_dt, end_dt = get_start_dt_end_dt_for_gold(target_date)
                        comparison = _compare_day(
                            test_flights,
                            gold_flights,
                            start_dt,
                            end_dt,
                            airport_catalog,
                        )
                        # Measure aircraft presence in the full-day inputs,
                        # before applying the evaluation window.
                        test_icaos = set(
                            test_flights.get_column("icao").unique().to_list()
                        )
                        gold_icaos = set(
                            gold_flights.get_column("icao").unique().to_list()
                        )
                        stats = EvaluationStats.from_comparison(
                            comparison,
                            test_date=target_date,
                            adsb_source=selected_adsb_source,
                            test_source=resolved_test_source.name,
                            gold_source=resolved_gold_source.name,
                            target_day_adsb=pair_target_day_adsb,
                            gold_icao_count=len(gold_icaos),
                            gold_icao_missing_from_test_count=len(
                                gold_icaos - test_icaos
                            ),
                        )
                        cache_path = cache.store(
                            cache_key,
                            comparison,
                            stats,
                        )
                        print(f"Cached evaluation at {cache_path}")

                    daily_results.append(
                        EvaluationResult(
                            comparison=comparison,
                            stats=stats,
                        )
                    )
                    metrics = {
                        key: value
                        for key, value in stats.to_row().items()
                        if key not in EVALUATION_METADATA_COLUMNS
                    }
                    print(
                        f"Stats for {resolved_test_source.name}:"
                        f"{resolved_gold_source.name} on {target_date}: {metrics}"
                    )
                    print(
                        "Gold ICAOs absent from test: "
                        f"{stats.gold_icao_missing_from_test_count}/"
                        f"{stats.gold_icao_count} "
                        f"({stats.gold_icao_missing_from_test_pct:.2%})"
                    )
                    test_flight_count = stats.true_positive + stats.false_positive
                    print(
                        f"Null airports in {resolved_test_source.name} "
                        f"({test_flight_count} retained test flights): "
                        f"takeoff {stats.takeoff_airport_id_null_count} "
                        f"({stats.takeoff_airport_id_null_pct:.2%}), "
                        f"landing {stats.landing_airport_id_null_count} "
                        f"({stats.landing_airport_id_null_pct:.2%})"
                    )
                    print(
                        "Timing against gold: "
                        "max endpoint error "
                        f"median {_format_minutes(stats.median_max_endpoint_time_error_minutes)}, "
                        f"p90 {_format_minutes(stats.p90_max_endpoint_time_error_minutes)}; "
                        "both endpoints within "
                        f"5m {stats.both_endpoints_within_5m_recall:.2%}, "
                        f"15m {stats.both_endpoints_within_15m_recall:.2%}, "
                        f"30m {stats.both_endpoints_within_30m_recall:.2%}; "
                        "bias "
                        f"takeoff {_format_minutes(stats.takeoff_time_bias_minutes)}, "
                        f"landing {_format_minutes(stats.landing_time_bias_minutes)}"
                    )
                    if resolved_test_source.name == "opensky":
                        print(
                            "Non-null airport accuracy in opensky (matched flights): "
                            f"takeoff {stats.takeoff_airport_id_non_null_match_pct:.2%}, "
                            f"landing {stats.landing_airport_id_non_null_match_pct:.2%}"
                        )

    return EvaluationResults(entries=tuple(daily_results))


def list_cached_evaluations(
    *,
    test_source: str | Sequence[str] | None = None,
    gold_source: str | Sequence[str] | None = None,
    adsb_source: ADSBSource | str | Sequence[ADSBSource | str] | None = None,
    test_dates: date | Sequence[date] | None = None,
    target_day_adsb: bool = False,
    evaluation_cache: EvaluationCache | None = None,
) -> pl.DataFrame:
    """List cached daily stats matching the supplied source/date filters."""
    cache = evaluation_cache or default_evaluation_cache()
    frame = cache.list_entries(
        test_source=test_source,
        gold_source=gold_source,
        adsb_source=adsb_source,
        test_dates=test_dates,
    )
    if target_day_adsb:
        frame = frame.filter(pl.col("target_day_adsb"))
    return frame


def export_cached_evaluations(
    output_csv: Path,
    *,
    columns: Sequence[str] | None = None,
    test_source: str | Sequence[str] | None = None,
    gold_source: str | Sequence[str] | None = None,
    adsb_source: ADSBSource | str | Sequence[ADSBSource | str] | None = None,
    test_dates: date | Sequence[date] | None = None,
    target_day_adsb: bool = False,
    evaluation_cache: EvaluationCache | None = None,
) -> tuple[pl.DataFrame, Path]:
    """Filter cached daily results and export selected columns to CSV."""
    cached = list_cached_evaluations(
        test_source=test_source,
        gold_source=gold_source,
        adsb_source=adsb_source,
        test_dates=test_dates,
        target_day_adsb=target_day_adsb,
        evaluation_cache=evaluation_cache,
    )
    selected = select_output_columns(cached, columns)
    path = write_evaluation_csv(selected, output_csv)
    return selected, path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare daily flight sources. Results are cached under "
            "OUTPUT_DIR/data/cache/evaluation."
        )
    )
    parser.add_argument(
        "--test-source",
        nargs="+",
        choices=FLIGHT_SOURCES_BY_NAME,
        help=f"Source(s) to evaluate (default: {ALGORITHM_NEAREST_SOURCE.name}).",
    )
    parser.add_argument(
        "--gold-source",
        nargs="+",
        choices=FLIGHT_SOURCES_BY_NAME,
        help=f"Reference source(s) (default: {BTS_SOURCE.name}).",
    )
    parser.add_argument(
        "--adsb-source",
        nargs="+",
        type=ADSBSource,
        choices=tuple(ADSBSource),
        help=f"ADS-B provider(s) (default: {DEFAULT_ADSB_SOURCE.value}).",
    )
    parser.add_argument(
        "--test-dates",
        nargs="+",
        type=date.fromisoformat,
        help=f"UTC flight dates (default: {DEFAULT_TARGET_DATE}).",
    )
    parser.add_argument(
        "--airport-nn-model",
        type=Path,
        help="Checkpoint to use for algorithm-nn or opensky-nn sources.",
    )
    parser.add_argument(
        "--airport-nn-training",
        type=Path,
        help=(
            "Optional training.json manifest (relative paths use OUTPUT_DIR). "
            "Write separate airport-exposure and training-airport CSVs "
            "alongside the normal stats CSV."
        ),
    )
    parser.add_argument(
        "--target-day-adsb",
        action="store_true",
        help=(
            "Use only each target UTC day's ADS-B data for every provider "
            "([00:00, next 00:00)), before segmentation and inference. "
            "Default: provider-specific windows. With --list-cache, filter "
            "by this setting."
        ),
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Recalculate requested evaluations and replace matching cache entries.",
    )
    parser.add_argument(
        "--list-cache",
        action="store_true",
        help="List cached daily results matching the supplied source/date filters.",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        help="CSV columns in output order (default: every applicable column).",
    )
    args = parser.parse_args(argv)
    if args.list_cache and args.airport_nn_training is not None:
        parser.error(
            "--airport-nn-training requires detailed comparisons; omit --list-cache "
            "to evaluate the requested dates using cached results when available."
        )
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    training = (
        load_training_airports(args.airport_nn_training)
        if args.airport_nn_training is not None
        else None
    )
    if args.list_cache:
        cached = list_cached_evaluations(
            test_source=args.test_source,
            gold_source=args.gold_source,
            adsb_source=args.adsb_source,
            test_dates=args.test_dates,
            target_day_adsb=args.target_day_adsb,
        )
        cached = select_output_columns(cached, args.columns)
        if cached.is_empty():
            print("No cached evaluations matched.")
        else:
            cached.write_csv(sys.stdout)
        return

    results = evaluate(
        test_source=args.test_source or [ALGORITHM_NEAREST_SOURCE.name],
        gold_source=args.gold_source or [BTS_SOURCE.name],
        test_dates=args.test_dates or [DEFAULT_TARGET_DATE],
        adsb_source=args.adsb_source or [DEFAULT_ADSB_SOURCE],
        airport_nn_model_path=args.airport_nn_model,
        target_day_adsb=args.target_day_adsb,
        refresh_cache=args.refresh_cache,
    )
    stats_output = write_evaluation_csv(
        results.to_frame(),
        default_stats_path(),
        columns=args.columns,
    )
    print(f"Stats CSV saved to {stats_output}")
    if training is not None:
        exposure_output = write_evaluation_csv(
            airport_exposure_report(results, training),
            stats_output.with_name(f"{stats_output.stem}_airport_exposure.csv"),
        )
        training_output = write_evaluation_csv(
            training.airport_counts,
            stats_output.with_name(f"{stats_output.stem}_training_airports.csv"),
        )
        print(f"Airport exposure CSV saved to {exposure_output}")
        print(
            f"Training airport IDs and endpoint counts saved to {training_output} "
            f"({training.airport_counts.height} airports)"
        )


if __name__ == "__main__":
    main()
'''
COLUMNS = ["icao", "takeoff_time", "landing_time", "takeoff_airport_ident", "landing_airport_ident", "takeoff_time_df1", "landing_time_df1", "takeoff_airport_ident_df1", "landing_airport_ident_df1", "match_status"]
'''
