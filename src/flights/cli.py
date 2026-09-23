import argparse
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Sequence

from data_engineering.adsb.read_adsb import ADSBSource
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
from planequery_data.utils import date_range


def write_flights_range(
    start_date: date,
    end_date: date,
    source: FlightSource,
    adsb_source: ADSBSource = ADSBSource.ADSBLOL,
) -> None:
    source_adsb = adsb_source if isinstance(source, ADSBFlightSource) else None

    for target_date in date_range(start_date, end_date):
        flights = write_flights(source, target_date, source_adsb)
        print(
            f"Wrote {len(flights)} {source.name} flights for {target_date} to "
            f"{flight_path(source, target_date, source_adsb)}"
        )


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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    source = FLIGHT_SOURCES_BY_NAME[args.source]
    if args.mike_input is not None:
        assert isinstance(source, MikeFlightSource)
        source = replace(source, input_csv=args.mike_input)
    if args.airport_nn_model is not None:
        assert isinstance(source, AlgorithmNNFlightSource)
        source = replace(source, model_path=args.airport_nn_model)
    write_flights_range(
        args.start_date,
        args.end_date,
        source,
        adsb_source=args.adsb_source,
    )


if __name__ == "__main__":
    main()
