"""Download UTC days of OpenSky flights_data4 from Trino to parquet."""

import argparse
from collections.abc import Sequence
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pyarrow.parquet as pq
from trino.dbapi import Connection

from data_engineering.opensky.paths import (
    OPENSKY_RAW_FLIGHTS_ROOT,
    raw_flights_day_path,
)
from data_engineering.opensky.trino_client import (
    BATCH_SIZE,
    DEFAULT_TRINO_USER,
    create_connection,
    rows_to_arrow,
)


DEFAULT_OUTPUT_DIR = OPENSKY_RAW_FLIGHTS_ROOT
QUERY_WINDOW_MINUTES = 60

# These are all columns exposed by the current flights_data4 table except for
# ``track``. That nested trajectory dominates the download size and the flight
# evaluation pipeline does not consume it.
FLIGHTS_DATA4_SELECT = """
    icao24,
    firstseen,
    estdepartureairport,
    lastseen,
    estarrivalairport,
    callsign,
    estdepartureairporthorizdistance,
    estdepartureairportvertdistance,
    estarrivalairporthorizdistance,
    estarrivalairportvertdistance,
    departureairportcandidatescount,
    arrivalairportcandidatescount,
    otherdepartureairportcandidates,
    otherarrivalairportcandidates,
    day
""".strip()


def flights_output_path(target_date: date, output_dir: Path) -> Path:
    """Return the raw parquet path for one UTC day."""
    return raw_flights_day_path(target_date, root=output_dir)


def download_day(
    target_date: date,
    *,
    user: str = DEFAULT_TRINO_USER,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    batch_size: int = BATCH_SIZE,
    query_window_minutes: int = QUERY_WINDOW_MINUTES,
    overwrite: bool = False,
    connection: Connection | None = None,
) -> Path:
    """Download one ``flights_data4`` UTC-day partition.

    Every query includes the exact ``day`` partition predicate required by
    OpenSky. The partition is split into ``lastseen`` windows to remain below
    Trino's query time limit. Existing output is left untouched. Partial output
    is removed if the query or parquet write fails.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if query_window_minutes <= 0:
        raise ValueError("query_window_minutes must be positive")

    output_path = flights_output_path(target_date, output_dir)
    if output_path.exists() and not overwrite:
        print(f"Skipping {target_date}: already downloaded")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_path.unlink(missing_ok=True)

    owns_connection = connection is None
    if connection is None:
        connection = create_connection(user)

    start = datetime.combine(target_date, time.min, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    window_size = timedelta(minutes=query_window_minutes)
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    columns: list[str] | None = None

    print(f"Downloading flights_data4 for {target_date}")
    try:
        window_start = start
        while window_start < end:
            window_end = min(window_start + window_size, end)
            cursor = connection.cursor()
            try:
                query = f"""
                    SELECT {FLIGHTS_DATA4_SELECT}
                    FROM flights_data4
                    WHERE day = ?
                      AND lastseen >= ?
                      AND lastseen < ?
                    """
                cursor.execute(
                    query,
                    [
                        int(start.timestamp()),
                        int(window_start.timestamp()),
                        int(window_end.timestamp()),
                    ],
                )
                window_columns = [column[0] for column in cursor.description]
                if columns is None:
                    columns = window_columns
                elif window_columns != columns:
                    raise RuntimeError("Trino returned inconsistent columns")

                while rows := cursor.fetchmany(batch_size):
                    table = rows_to_arrow(rows, columns)
                    if writer is None:
                        writer = pq.ParquetWriter(
                            temp_path,
                            table.schema,
                            compression="zstd",
                            use_dictionary=True,
                        )
                    writer.write_table(table)
                    total_rows += len(rows)
                    print(f"\r  {total_rows:,} rows", end="", flush=True)
            finally:
                cursor.close()

            window_start = window_end

        if writer is None:
            if columns is None:
                raise RuntimeError("Trino did not return column metadata")
            pq.write_table(rows_to_arrow([], columns), temp_path)
        else:
            writer.close()
            writer = None

        temp_path.replace(output_path)
        print(f"\nSaved {total_rows:,} rows -> {output_path}")
        return output_path
    except BaseException:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
            writer = None
        temp_path.unlink(missing_ok=True)
        raise
    finally:
        if owns_connection:
            connection.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download OpenSky flights_data4 from Trino for one UTC day or a "
            "range of UTC days."
        ),
    )
    parser.add_argument(
        "start_date",
        type=date.fromisoformat,
        help="First UTC date to download (YYYY-MM-DD)",
    )
    parser.add_argument(
        "end_date",
        type=date.fromisoformat,
        nargs="?",
        help=(
            "Exclusive end UTC date (YYYY-MM-DD); omit to download only "
            "START_DATE"
        ),
    )
    parser.add_argument(
        "--user",
        default=DEFAULT_TRINO_USER,
        help=(
            "OpenSky Trino user "
            "(default: OPENSKY_TRINO_USER or goodejonah)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Base output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Rows fetched per batch (default: {BATCH_SIZE:,})",
    )
    parser.add_argument(
        "--query-window-minutes",
        type=int,
        default=QUERY_WINDOW_MINUTES,
        help=(
            "Minutes covered by each Trino query "
            f"(default: {QUERY_WINDOW_MINUTES})"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing completed day after a successful download",
    )
    args = parser.parse_args(argv)
    if args.end_date is not None and args.end_date <= args.start_date:
        parser.error("END_DATE must be after START_DATE")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    end_date = args.end_date or args.start_date + timedelta(days=1)
    target_date = args.start_date
    while target_date < end_date:
        download_day(
            target_date,
            user=args.user,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            query_window_minutes=args.query_window_minutes,
            overwrite=args.overwrite,
        )
        target_date += timedelta(days=1)


if __name__ == "__main__":
    main()
