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
from flights.airport_nn.source import AlgorithmNNFlightSource
from flights.evaluation.cache import EvaluationCache
from flights.evaluation.models import (
    EVALUATION_METADATA_COLUMNS,
    EvaluationResult,
    EvaluationResults,
    EvaluationStats,
)
from flights.flight_source import ADSBFlightSource, FlightSource, get_flights
from flights.flights_comparison import df_flights_comparison
from flights.sources import (
    ALGORITHM_NEAREST_SOURCE,
    BTS_SOURCE,
    FLIGHT_SOURCES_BY_NAME,
)
from planequery_data.utils import current_commit_hash


DEFAULT_TARGET_DATE = date(2026, 3, 1)
DEFAULT_ADSB_SOURCE = ADSBSource.ADSBLOL

SourceInput = FlightSource | str


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
    if isinstance(source, ADSBFlightSource):
        if adsb_source is None:
            raise ValueError(f"{source.name} requires an ADS-B source")
        source_adsb = ADSBSource(adsb_source)
    else:
        source_adsb = None

    def read_source(target_date: date) -> pl.DataFrame:
        if (
            isinstance(source, AlgorithmNNFlightSource)
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
    if isinstance(resolve_source(test_source), ADSBFlightSource) or isinstance(
        resolve_source(gold_source), ADSBFlightSource
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
        if isinstance(source, AlgorithmNNFlightSource)
        else source
        for source in sources
    ]


def _flights_on_date(flights: pl.DataFrame, target_date: date) -> pl.DataFrame:
    return flights.filter(
        (pl.col("takeoff_time").dt.date() == target_date)
        & (pl.col("landing_time").dt.date() == target_date)
    )


def _assert_no_null_icaos(flights: pl.DataFrame, source_role: str) -> None:
    null_count = flights.get_column("icao").null_count()
    assert null_count == 0, (
        f"{source_role} contains {null_count} null icao value(s)"
    )


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


def _empty_comparison() -> pl.DataFrame:
    """Return the columns needed to calculate zero-valued comparison stats."""
    return pl.DataFrame(
        schema={
            "match_status": pl.String,
            "same_takeoff_airport_id": pl.Boolean,
            "same_landing_airport_id": pl.Boolean,
            "same_airport_id": pl.Boolean,
        }
    )


def _compare_day(
    test_flights: pl.DataFrame,
    gold_flights: pl.DataFrame,
    target_date: date,
    airport_catalog: AirportCatalogView,
) -> pl.DataFrame:
    _assert_no_null_icaos(test_flights, "test_source")
    _assert_no_null_icaos(gold_flights, "gold_source")
    test_flights = _flights_on_date(test_flights, target_date)
    gold_flights = _flights_on_date(gold_flights, target_date)
    test_flights, gold_flights = _gold_icao_cohort(
        test_flights,
        gold_flights,
    )
    if test_flights.is_empty() and gold_flights.is_empty():
        return _empty_comparison()

    start = datetime(target_date.year, target_date.month, target_date.day)
    return df_flights_comparison(
        test_flights,
        gold_flights,
        start,
        start + timedelta(days=1),
        airport_catalog,
        compare_airports=False,
    )


def evaluate(
    test_source: SourceInput | Sequence[SourceInput],
    gold_source: SourceInput | Sequence[SourceInput],
    test_dates: date | Sequence[date],
    adsb_source: (
        ADSBSource | str | Sequence[ADSBSource | str]
    ) = DEFAULT_ADSB_SOURCE,
    *,
    write_outputs: bool = True,
    output_csv: Path | None = None,
    output_columns: Sequence[str] | None = None,
    airport_nn_model_path: Path | None = None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    evaluation_cache: EvaluationCache | None = None,
) -> EvaluationResults:
    """Compare flight sources, reusing one persistent cache entry per day.

    Each comparison keeps all gold-source ICAOs and restricts test flights to
    ICAOs present in gold. Detailed comparisons and their daily stats are
    cached together. Set
    ``refresh_cache=True`` to recompute and replace matching entries, or
    ``use_cache=False`` to bypass both cache reads and writes. CSV output keeps
    all stats columns unless ``output_columns`` supplies a projection/order.
    """
    if not write_outputs and (
        output_csv is not None or output_columns is not None
    ):
        raise ValueError("output_csv/output_columns require write_outputs=True")
    test_sources = _with_airport_nn_model(
        _as_source_list(test_source),
        airport_nn_model_path,
    )
    gold_sources = _with_airport_nn_model(
        _as_source_list(gold_source),
        airport_nn_model_path,
    )
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
            pair_adsb_sources = adsb_sources_for_evaluation(
                resolved_test_source,
                resolved_gold_source,
                adsb_sources,
            )
            for selected_adsb_source in pair_adsb_sources:
                test_adsb_source = (
                    selected_adsb_source
                    if isinstance(resolved_test_source, ADSBFlightSource)
                    else None
                )
                gold_adsb_source = (
                    selected_adsb_source
                    if isinstance(resolved_gold_source, ADSBFlightSource)
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
                    if use_cache and not refresh_cache and cache_key is not None:
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
                        comparison = _compare_day(
                            read_flights(
                                resolved_test_source,
                                test_adsb_source,
                                target_date,
                            ),
                            read_flights(
                                resolved_gold_source,
                                gold_adsb_source,
                                target_date,
                            ),
                            target_date,
                            airport_catalog,
                        )
                        stats = EvaluationStats.from_comparison(
                            comparison,
                            test_date=target_date,
                            adsb_source=selected_adsb_source,
                            test_source=resolved_test_source.name,
                            gold_source=resolved_gold_source.name,
                        )
                        if use_cache and cache_key is not None:
                            cache_path = cache.store(
                                cache_key,
                                comparison,
                                stats,
                                overwrite=refresh_cache,
                            )
                            if cache_path is not None:
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

    results = EvaluationResults(entries=tuple(daily_results))
    if write_outputs:
        stats_output = write_evaluation_csv(
            results.to_frame(),
            output_csv or default_stats_path(),
            columns=output_columns,
        )
        print(f"Stats CSV saved to {stats_output}")
    return results


def list_cached_evaluations(
    *,
    test_source: str | Sequence[str] | None = None,
    gold_source: str | Sequence[str] | None = None,
    adsb_source: ADSBSource | str | Sequence[ADSBSource | str] | None = None,
    test_dates: date | Sequence[date] | None = None,
    evaluation_cache: EvaluationCache | None = None,
) -> pl.DataFrame:
    """List cached daily stats matching the supplied source/date filters."""
    cache = evaluation_cache or default_evaluation_cache()
    return cache.list_entries(
        test_source=test_source,
        gold_source=gold_source,
        adsb_source=adsb_source,
        test_dates=test_dates,
    )


def export_cached_evaluations(
    output_csv: Path,
    *,
    columns: Sequence[str] | None = None,
    test_source: str | Sequence[str] | None = None,
    gold_source: str | Sequence[str] | None = None,
    adsb_source: ADSBSource | str | Sequence[ADSBSource | str] | None = None,
    test_dates: date | Sequence[date] | None = None,
    evaluation_cache: EvaluationCache | None = None,
) -> tuple[pl.DataFrame, Path]:
    """Filter cached daily results and export selected columns to CSV."""
    cached = list_cached_evaluations(
        test_source=test_source,
        gold_source=gold_source,
        adsb_source=adsb_source,
        test_dates=test_dates,
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
        help="Checkpoint to use for an algorithm-nn source.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Recalculate requested evaluations and replace matching cache entries.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not read or write the persistent evaluation cache.",
    )
    parser.add_argument(
        "--list-cache",
        action="store_true",
        help="List cached daily results matching the supplied source/date filters.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help=(
            "CSV destination. Normal evaluations otherwise use a timestamped "
            "path; cache listings otherwise write CSV to stdout."
        ),
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        help="CSV columns in output order (default: every applicable column).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.list_cache:
        cached = list_cached_evaluations(
            test_source=args.test_source,
            gold_source=args.gold_source,
            adsb_source=args.adsb_source,
            test_dates=args.test_dates,
        )
        cached = select_output_columns(cached, args.columns)
        if args.output_csv is not None:
            output_path = write_evaluation_csv(cached, args.output_csv)
            print(f"Cached evaluation CSV saved to {output_path}")
            return
        if cached.is_empty():
            print("No cached evaluations matched.")
        else:
            cached.write_csv(sys.stdout)
        return

    evaluate(
        test_source=args.test_source or [ALGORITHM_NEAREST_SOURCE.name],
        gold_source=args.gold_source or [BTS_SOURCE.name],
        test_dates=args.test_dates or [DEFAULT_TARGET_DATE],
        adsb_source=args.adsb_source or [DEFAULT_ADSB_SOURCE],
        airport_nn_model_path=args.airport_nn_model,
        output_csv=args.output_csv,
        output_columns=args.columns,
        use_cache=not args.no_cache,
        refresh_cache=args.refresh_cache,
    )


if __name__ == "__main__":
    main()
