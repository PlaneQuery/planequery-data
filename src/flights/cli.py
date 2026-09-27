import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Sequence

from data_engineering.adsb.read_adsb import ADSBSource
from flights.algorithm import AlgorithmNearestFlightSource
from flights.airport_nn import AlgorithmNNFlightSource
from flights.flight_source import (
    ADSBFlightSource,
    FlightSource,
    flight_path,
    write_flights,
)
from flights.providers.mike import MikeFlightSource
from flights.sources import (
    ALGORITHM_NN_SOURCE,
    FLIGHT_SOURCES,
    FLIGHT_SOURCES_BY_NAME,
)
from flights.workers import default_worker_count
from planequery_data.utils import date_range


PARALLEL_ALGORITHM_SOURCES = (
    AlgorithmNearestFlightSource,
    AlgorithmNNFlightSource,
)
_RANGE_WORKER_SOURCE: FlightSource | None = None
_RANGE_WORKER_ADSB_SOURCE: ADSBSource | None = None


def default_workers_for_source(source: FlightSource) -> int:
    if isinstance(source, PARALLEL_ALGORITHM_SOURCES):
        return default_worker_count()
    return 1


def _write_flights_day(
    source: FlightSource,
    target_date: date,
    adsb_source: ADSBSource | None,
    batch_workers: int = 1,
) -> tuple[date, int, Path]:
    if batch_workers == 1:
        flights = write_flights(source, target_date, adsb_source)
    else:
        flights = write_flights(
            source,
            target_date,
            adsb_source,
            batch_workers=batch_workers,
        )
    return (
        target_date,
        len(flights),
        flight_path(source, target_date, adsb_source),
    )


def _initialize_range_worker(
    source: FlightSource,
    adsb_source: ADSBSource | None,
) -> None:
    global _RANGE_WORKER_SOURCE, _RANGE_WORKER_ADSB_SOURCE
    _RANGE_WORKER_SOURCE = source
    _RANGE_WORKER_ADSB_SOURCE = adsb_source


def _write_flights_day_in_worker(target_date: date) -> tuple[date, int, Path]:
    assert _RANGE_WORKER_SOURCE is not None
    return _write_flights_day(
        _RANGE_WORKER_SOURCE,
        target_date,
        _RANGE_WORKER_ADSB_SOURCE,
    )


def _print_write_result(result: tuple[date, int, Path], source: FlightSource) -> None:
    target_date, flight_count, path = result
    print(
        f"Wrote {flight_count} {source.name} flights for {target_date} to {path}"
    )


def write_flights_range(
    start_date: date,
    end_date: date,
    source: FlightSource,
    adsb_source: ADSBSource = ADSBSource.ADSBLOL,
    workers: int | None = None,
) -> None:
    source_adsb = adsb_source if isinstance(source, ADSBFlightSource) else None
    if workers is None:
        workers = default_workers_for_source(source)
    if workers < 1:
        raise ValueError("workers must be positive")

    target_dates = list(date_range(start_date, end_date))
    within_day_parallel = (
        len(target_dates) == 1
        and isinstance(source, PARALLEL_ALGORITHM_SOURCES)
    )
    if within_day_parallel:
        print(
            f"Using {workers} within-day worker(s) for {source.name}",
            flush=True,
        )
        _print_write_result(
            _write_flights_day(
                source,
                target_dates[0],
                source_adsb,
                batch_workers=workers,
            ),
            source,
        )
        return

    worker_count = min(workers, len(target_dates))
    print(f"Using {worker_count} day worker(s) for {source.name}", flush=True)

    if worker_count == 1:
        for target_date in target_dates:
            _print_write_result(
                _write_flights_day(source, target_date, source_adsb),
                source,
            )
        return

    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_range_worker,
        initargs=(source, source_adsb),
    ) as executor:
        results = executor.map(_write_flights_day_in_worker, target_dates)
        for result in results:
            _print_write_result(result, source)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create daily flight parquet files for a [start, end) date range."
    )
    parser.add_argument("start_date", type=date.fromisoformat)
    parser.add_argument("end_date", type=date.fromisoformat)
    parser.add_argument("source", choices=FLIGHT_SOURCES_BY_NAME)
    parser.add_argument(
        "--adsb-source",
        type=ADSBSource,
        choices=tuple(ADSBSource),
        default=ADSBSource.ADSBLOL,
        help="ADS-B provider used by ADS-B flight sources (default: adsblol).",
    )
    parser.add_argument(
        "--airport-nn-model",
        type=Path,
        help=f"Checkpoint used by {ALGORITHM_NN_SOURCE.name}.",
    )
    parser.add_argument(
        "--mike-input",
        type=Path,
        help="Input CSV used by the mike source.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help=(
            "Parallel workers. A one-day algorithm run parallelizes ADS-B "
            "aircraft batches; a multi-day run parallelizes days instead. "
            "Defaults are sized from CPU count and RAM."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.workers is not None and args.workers < 1:
        raise SystemExit("--workers must be positive")
    source = FLIGHT_SOURCES_BY_NAME[args.source]
    if args.mike_input is not None:
        assert isinstance(source, MikeFlightSource)
        source = replace(source, input_csv=args.mike_input)
    if args.airport_nn_model is not None:
        assert isinstance(source, AlgorithmNNFlightSource)
        source = replace(source, model_path=args.airport_nn_model)
    if isinstance(source, ADSBFlightSource) and args.adsb_source is ADSBSource.ADSBLOL:
        from data_engineering.adsblol.check_for_new_release import (
            prepare_releases_for_run,
        )

        prepare_releases_for_run(
            {day.year for day in date_range(args.start_date, args.end_date)}
        )
    kwargs = {
        "adsb_source": args.adsb_source,
    }
    if args.workers is not None:
        kwargs["workers"] = args.workers
    write_flights_range(
        args.start_date,
        args.end_date,
        source,
        **kwargs,
    )


if __name__ == "__main__":
    main()
