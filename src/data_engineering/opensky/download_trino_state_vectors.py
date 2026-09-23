"""Download one UTC day of OpenSky state vectors from Trino to parquet."""

import argparse
import os
from collections.abc import Sequence
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from trino.auth import OAuth2Authentication
from trino.dbapi import Connection, connect

from data_engineering.opensky.paths import (
    OPENSKY_RAW_STATE_VECTORS_ROOT,
    raw_state_vectors_hour_path,
)


BATCH_SIZE = 100_000
DEFAULT_OUTPUT_DIR = OPENSKY_RAW_STATE_VECTORS_ROOT
DEFAULT_TRINO_USER = os.environ.get("OPENSKY_TRINO_USER", "goodejonah")


def rows_to_arrow(
    rows: list[tuple[Any, ...]],
    columns: list[str],
) -> pa.Table:
    """Convert one batch of Trino DB-API rows into an Arrow table."""
    if not rows:
        return pa.table({column: [] for column in columns})

    return pa.Table.from_arrays(
        [pa.array(values) for values in zip(*rows)],
        names=columns,
    )


def create_connection(user: str) -> Connection:
    """Create an OAuth-authenticated connection to OpenSky's Trino service."""
    return connect(
        host="trino.opensky-network.org",
        port=443,
        user=user,
        catalog="minio",
        schema="osky",
        http_scheme="https",
        auth=OAuth2Authentication(),
        timezone="UTC",
        encoding=["json+zstd", "json"],
    )


def state_vectors_output_path(start: datetime, output_dir: Path) -> Path:
    """Return the parquet path for a UTC hour."""
    return raw_state_vectors_hour_path(start, root=output_dir)


def download_hour(
    connection: Connection,
    start: datetime,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    batch_size: int = BATCH_SIZE,
) -> Path:
    """Download one UTC hour, returning its final parquet path.

    Existing output is left untouched. A partially written file is kept under
    a ``.tmp`` suffix and removed if querying or writing fails.
    """
    if start.tzinfo is None or start.utcoffset() != timedelta(0):
        raise ValueError("start must be timezone-aware and in UTC")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    output_path = state_vectors_output_path(start, output_dir)
    if output_path.exists():
        print(f"Skipping {start}: already downloaded")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_path.unlink(missing_ok=True)

    print(f"Downloading {start}")
    cursor = connection.cursor()
    writer: pq.ParquetWriter | None = None
    total_rows = 0

    try:
        cursor.execute(
            """
            SELECT *
            FROM state_vectors_data4
            WHERE hour = ?
            """,
            [int(start.timestamp())],
        )
        columns = [column[0] for column in cursor.description]

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

        if writer is None:
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
        if writer is not None:
            writer.close()
        cursor.close()


def download_day(
    target_date: date,
    *,
    user: str = DEFAULT_TRINO_USER,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    batch_size: int = BATCH_SIZE,
    connection: Connection | None = None,
) -> list[Path]:
    """Download all 24 UTC hours for ``target_date``.

    When ``connection`` is omitted, this function creates and closes its own
    OAuth-authenticated OpenSky Trino connection.
    """
    owns_connection = connection is None
    if connection is None:
        connection = create_connection(user)

    start_of_day = datetime.combine(target_date, time.min, tzinfo=timezone.utc)
    try:
        return [
            download_hour(
                connection,
                start_of_day + timedelta(hours=hour),
                output_dir=output_dir,
                batch_size=batch_size,
            )
            for hour in range(24)
        ]
    finally:
        if owns_connection:
            connection.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download one UTC day of OpenSky state vectors from Trino.",
    )
    parser.add_argument("date", type=date.fromisoformat, help="UTC date (YYYY-MM-DD)")
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    download_day(
        args.date,
        user=args.user,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
