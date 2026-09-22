import argparse
from datetime import date
from pathlib import Path

from flights.flight_source import write_flights
from flights.providers.mike import DEFAULT_INPUT, MikeFlightSource


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Mike's flights CSV into one daily flight partition."
    )
    parser.add_argument("date", type=date.fromisoformat)
    parser.add_argument(
        "input_csv",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input flights CSV (default: {DEFAULT_INPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_flights(
        MikeFlightSource(input_csv=args.input_csv),
        args.date,
    )


if __name__ == "__main__":
    main()
