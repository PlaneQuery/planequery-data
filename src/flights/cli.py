import argparse
import multiprocessing
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Sequence

from data_engineering.adsb.read_adsb import ADSBSource
from data_engineering.adsb.known_bad_days import ADSBLOL_SKIP_DAYS
from flights.algorithm import AlgorithmNearestFlightSource
from flights.airport_nn import AlgorithmNNFlightSource, OpenSkyNNFlightSource
from flights.flight_source import (
    ADSBFlightSource,
    FlightSource,
    flight_path,
    write_flights,
)
from flights.junzis import JunzisFlightSource
from flights.opensky_reverse_eng import OpenSkyReverseEngFlightSource
from flights.providers.mike import MikeFlightSource
from flights.providers.eurocontrol import EurocontrolFlightSource
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
    JunzisFlightSource,
    OpenSkyReverseEngFlightSource,
)
_RANGE_WORKER_SOURCE: FlightSource | None = None
_RANGE_WORKER_ADSB_SOURCE: ADSBSource | None = None


def default_workers_for_source(source: FlightSource) -> int:
    if isinstance(source, PARALLEL_ALGORITHM_SOURCES):
        return default_worker_count()
    return 1


def adsb_dates_for_run(
    source: ADSBFlightSource,
    target_dates: Sequence[date],
    adsb_source: ADSBSource = ADSBSource.ADSBLOL,
) -> list[date]:
    """Return every ADS-B partition needed by the source read windows."""
    input_dates = set()
    for target_date in target_dates:
        start, end = source.adsb_input_interval(
            target_date,
            adsb_source,
        )
        if not isinstance(start, datetime):
            start = datetime.combine(start, datetime.min.time())
        end = end if end is not None else start + timedelta(days=1)
        if end <= start:
            continue
        final_date = (end - timedelta(microseconds=1)).date()
        input_dates.update(
            start.date() + timedelta(days=offset)
            for offset in range((final_date - start.date()).days + 1)
        )
    return sorted(input_dates)


def _downloaded_adsb_paths(
    source: ADSBSource,
    target_dates: Sequence[date],
) -> tuple[Path, ...]:
    """Return per-day download/cache paths that a flight run may create."""
    from data_engineering.adsb.canonical import adsb_day_dir

    paths = [adsb_day_dir(source, target_date) for target_date in target_dates]
    if source is ADSBSource.ADSBLOL:
        from data_engineering.adsb.download_adsb_to_parquet import (
            EXTRACT_DIR,
            RELEASES_DIR,
        )

        for root in (EXTRACT_DIR, RELEASES_DIR):
            paths.extend(
                root
                / f"year={target_date.year}"
                / f"month={target_date.month:02d}"
                / f"day={target_date.day:02d}"
                for target_date in target_dates
            )
    return tuple(dict.fromkeys(paths))


def _remove_new_downloaded_adsb_paths(
    paths: Sequence[Path],
    preexisting_paths: set[Path],
) -> None:
    """Remove run-created ADS-B artifacts without deleting an existing cache."""
    for path in paths:
        if path in preexisting_paths or not path.exists():
            continue
        print(f"Removing downloaded ADS-B artifact: {path}")
        shutil.rmtree(path)


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
    if isinstance(source, ADSBFlightSource) and source_adsb is ADSBSource.ADSBLOL:
        valid_dates = []
        for target_date in target_dates:
            bad_days = ADSBLOL_SKIP_DAYS.intersection(
                adsb_dates_for_run(source, [target_date], source_adsb)
            )
            if bad_days:
                print(
                    f"[SKIP] {source.name} flights for {target_date}: "
                    f"requires known bad adsblol day(s): "
                    f"{', '.join(map(str, sorted(bad_days)))}",
                    flush=True,
                )
            else:
                valid_dates.append(target_date)
        target_dates = valid_dates
    if not target_dates:
        return

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
        help=(
            "ADS-B provider for flight sources, verified sources, or "
            "EUROCONTROL callsigns (default: adsblol)."
        ),
    )
    parser.add_argument(
        "--target-day-adsb",
        action="store_true",
        help=(
            "Use only each target UTC day's ADS-B data ([00:00, next 00:00)), "
            "before segmentation and inference. Requires an ADS-B flight "
            "source. Default: provider-specific windows."
        ),
    )
    parser.add_argument(
        "--airport-nn-model",
        type=Path,
        help="Checkpoint used by algorithm-nn or opensky-nn.",
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
    parser.add_argument(
        "--no-save-adsb-downloads",
        action="store_true",
        help=(
            "Process at most --workers days per batch and remove ADS-B release "
            "archives, extracted traces, and canonical ADS-B Parquet created "
            "by this invocation after their last required batch. Existing "
            "cached data and the generated flight Parquet are preserved."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.workers is not None and args.workers < 1:
        raise SystemExit("--workers must be positive")
    source = FLIGHT_SOURCES_BY_NAME[args.source]
    if isinstance(source, EurocontrolFlightSource):
        source = replace(source, callsign_source=args.adsb_source)
    if args.mike_input is not None:
        assert isinstance(source, MikeFlightSource)
        source = replace(source, input_csv=args.mike_input)
    if args.airport_nn_model is not None:
        assert isinstance(source, (AlgorithmNNFlightSource, OpenSkyNNFlightSource))
        source = replace(source, model_path=args.airport_nn_model)
    if args.target_day_adsb:
        if not isinstance(source, ADSBFlightSource):
            raise SystemExit("--target-day-adsb requires an ADS-B flight source")
        source = replace(source, target_day_adsb=True)
    if args.no_save_adsb_downloads and not isinstance(source, ADSBFlightSource):
        raise SystemExit("--no-save-adsb-downloads requires an ADS-B flight source")

    target_dates = list(date_range(args.start_date, args.end_date))
    input_dates = (
        adsb_dates_for_run(source, target_dates, args.adsb_source)
        if isinstance(source, ADSBFlightSource)
        else target_dates
    )
    downloaded_paths: tuple[Path, ...] = ()
    preexisting_paths: set[Path] = set()
    if args.no_save_adsb_downloads:
        downloaded_paths = _downloaded_adsb_paths(args.adsb_source, input_dates)
        preexisting_paths = {path for path in downloaded_paths if path.exists()}

    try:
        if (
            isinstance(source, ADSBFlightSource)
            and args.adsb_source is ADSBSource.ADSBLOL
        ):
            from data_engineering.adsblol.check_for_new_release import (
                prepare_releases_for_run,
            )

            prepare_releases_for_run({day.year for day in input_dates})
        kwargs = {
            "adsb_source": args.adsb_source,
        }
        if args.workers is not None:
            kwargs["workers"] = args.workers
        if args.no_save_adsb_downloads:
            # Do not enqueue the whole range: fast workers could otherwise
            # download months of data before the parent gets to cleanup.
            batch_size = args.workers or default_workers_for_source(source)
            last_use = {}
            for target_date in target_dates:
                for input_date in adsb_dates_for_run(
                    source, [target_date], args.adsb_source
                ):
                    last_use[input_date] = target_date
            for offset in range(0, len(target_dates), batch_size):
                batch = target_dates[offset : offset + batch_size]
                write_flights_range(
                    batch[0], batch[-1] + timedelta(days=1), source, **kwargs
                )
                # Adjacent flight days share input partitions. Wait until all
                # readers have finished, including readers in later batches.
                expired = [day for day, last in last_use.items() if last <= batch[-1]]
                _remove_new_downloaded_adsb_paths(
                    _downloaded_adsb_paths(args.adsb_source, expired),
                    preexisting_paths,
                )
                for day in expired:
                    del last_use[day]
        else:
            write_flights_range(
                args.start_date,
                args.end_date,
                source,
                **kwargs,
            )
    finally:
        if args.no_save_adsb_downloads:
            _remove_new_downloaded_adsb_paths(
                downloaded_paths,
                preexisting_paths,
            )


if __name__ == "__main__":
    main()
