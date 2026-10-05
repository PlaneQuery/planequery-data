import argparse
import hashlib
import logging
import os
import time as time_module
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import polars as pl
import requests

logger = logging.getLogger(__name__)

FR24_FLIGHT_SUMMARY_LIGHT_URL = (
    "https://fr24api.flightradar24.com/api/flight-summary/light"
)
TARGET_DATE = date(2026, 6, 1)
FR24_FLIGHTS_VERSION = "v3"
MAX_RETURNED_FLIGHTS = 20
INITIAL_REGISTRATION_BATCH_SIZE = 10
QUERIES_PER_MINUTE = 10
MIN_WINDOW = timedelta(minutes=1)


def _format_fr24_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _day_bounds(target_date: date) -> tuple[datetime, datetime]:
    start = datetime(target_date.year, target_date.month, target_date.day, tzinfo=UTC)
    return start, start + timedelta(days=1)


def _chunked(values: Sequence[str], size: int) -> list[list[str]]:
    return [list(values[i:i + size]) for i in range(0, len(values), size)]


def _batch_hash(registrations: Sequence[str]) -> str:
    return hashlib.sha1(",".join(registrations).encode()).hexdigest()[:10]


class RateLimiter:
    def __init__(self, queries_per_minute: int) -> None:
        self.interval_seconds = 60 / queries_per_minute
        self.last_request_at: float | None = None

    def wait(self) -> None:
        if self.last_request_at is None:
            self.last_request_at = time_module.monotonic()
            return

        elapsed = time_module.monotonic() - self.last_request_at
        sleep_seconds = self.interval_seconds - elapsed
        if sleep_seconds > 0:
            time_module.sleep(sleep_seconds)

        self.last_request_at = time_module.monotonic()


def get_fr24_flight_summary_light_by_registration(
    *,
    flight_datetime_from: datetime,
    flight_datetime_to: datetime,
    registrations: str | list[str],
    limit: int = MAX_RETURNED_FLIGHTS,
    rate_limiter: RateLimiter | None = None,
) -> pd.DataFrame:
    if isinstance(registrations, list):
        registrations_param = ",".join(registrations)
    else:
        registrations_param = registrations

    params = {
        "flight_datetime_from": _format_fr24_datetime(flight_datetime_from),
        "flight_datetime_to": _format_fr24_datetime(flight_datetime_to),
        "registrations": registrations_param,
        "sort": "asc",
        "limit": limit,
    }

    headers = {
        "Accept": "application/json",
        "Accept-Version": "v1",
        "Authorization": f"Bearer {os.environ['FR24_API_KEY']}",
    }

    if rate_limiter is not None:
        rate_limiter.wait()

    response = requests.get(
        FR24_FLIGHT_SUMMARY_LIGHT_URL,
        params=params,
        headers=headers,
        timeout=30,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise requests.HTTPError(
            f"{exc}. Response body: {response.text}",
            response=response,
        ) from exc

    payload = response.json()
    if isinstance(payload, dict) and "data" in payload:
        return pd.DataFrame(payload["data"])
    if isinstance(payload, list):
        return pd.DataFrame(payload)

    raise ValueError(f"Unexpected response shape: {payload}")


def _fetch_rows(
    *,
    registrations: list[str],
    flight_datetime_from: datetime,
    flight_datetime_to: datetime,
    max_returned_flights: int,
    rate_limiter: RateLimiter,
) -> list[dict]:
    df = get_fr24_flight_summary_light_by_registration(
        flight_datetime_from=flight_datetime_from,
        flight_datetime_to=flight_datetime_to,
        registrations=registrations,
        limit=max_returned_flights,
        rate_limiter=rate_limiter,
    )
    return df.to_dict("records")


def _fetch_complete_rows(
    *,
    registrations: list[str],
    flight_datetime_from: datetime,
    flight_datetime_to: datetime,
    max_returned_flights: int,
    rate_limiter: RateLimiter,
) -> list[dict]:
    try:
        rows = _fetch_rows(
            registrations=registrations,
            flight_datetime_from=flight_datetime_from,
            flight_datetime_to=flight_datetime_to,
            max_returned_flights=max_returned_flights,
            rate_limiter=rate_limiter,
        )
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        if status_code != 400:
            raise
        if len(registrations) == 1:
            logger.warning(
                "Skipping invalid FR24 registration %r: %s",
                registrations[0],
                exc,
            )
            return []

        midpoint = len(registrations) // 2
        return (
            _fetch_complete_rows(
                registrations=registrations[:midpoint],
                flight_datetime_from=flight_datetime_from,
                flight_datetime_to=flight_datetime_to,
                max_returned_flights=max_returned_flights,
                rate_limiter=rate_limiter,
            )
            + _fetch_complete_rows(
                registrations=registrations[midpoint:],
                flight_datetime_from=flight_datetime_from,
                flight_datetime_to=flight_datetime_to,
                max_returned_flights=max_returned_flights,
                rate_limiter=rate_limiter,
            )
        )

    if len(rows) < max_returned_flights:
        return rows

    if len(registrations) > 1:
        midpoint = len(registrations) // 2
        return (
            _fetch_complete_rows(
                registrations=registrations[:midpoint],
                flight_datetime_from=flight_datetime_from,
                flight_datetime_to=flight_datetime_to,
                max_returned_flights=max_returned_flights,
                rate_limiter=rate_limiter,
            )
            + _fetch_complete_rows(
                registrations=registrations[midpoint:],
                flight_datetime_from=flight_datetime_from,
                flight_datetime_to=flight_datetime_to,
                max_returned_flights=max_returned_flights,
                rate_limiter=rate_limiter,
            )
        )

    window = flight_datetime_to - flight_datetime_from
    if window <= MIN_WINDOW:
        raise RuntimeError(
            f"{registrations[0]} still returned {len(rows)} rows between "
            f"{_format_fr24_datetime(flight_datetime_from)} and "
            f"{_format_fr24_datetime(flight_datetime_to)}. "
            "Increase MAX_RETURNED_FLIGHTS or use a smaller MIN_WINDOW."
        )

    midpoint = flight_datetime_from + (window / 2)
    return (
        _fetch_complete_rows(
            registrations=registrations,
            flight_datetime_from=flight_datetime_from,
            flight_datetime_to=midpoint,
            max_returned_flights=max_returned_flights,
            rate_limiter=rate_limiter,
        )
        + _fetch_complete_rows(
            registrations=registrations,
            flight_datetime_from=midpoint,
            flight_datetime_to=flight_datetime_to,
            max_returned_flights=max_returned_flights,
            rate_limiter=rate_limiter,
        )
    )


def default_raw_batch_dir(target_date: date) -> Path:
    from data_engineering.utils import OUTPUT_DIR

    return (
        OUTPUT_DIR
        / "data"
        / "raw"
        / "fr24"
        / f"year={target_date.year}"
        / f"month={target_date.month:02d}"
        / f"day={target_date.day:02d}"
    )


def _batch_output_path(
    raw_output_dir: Path,
    batch_number: int,
    registrations: Sequence[str],
) -> Path:
    return (
        raw_output_dir
        / f"batch_{batch_number:05d}_{_batch_hash(registrations)}.parquet"
    )


def _write_raw_batch(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)


def _read_raw_batches(paths: Sequence[Path]) -> pd.DataFrame:
    existing_paths = [path for path in paths if path.exists()]
    if not existing_paths:
        return pd.DataFrame()

    df = pd.concat(
        [pd.read_parquet(path) for path in existing_paths],
        ignore_index=True,
    )
    if df.empty:
        return df

    if "fr24_id" in df.columns:
        return df.drop_duplicates(subset=["fr24_id"])
    return df.drop_duplicates()


def get_test_data(
    registrations: Sequence[str] | None = None,
    target_date: date = TARGET_DATE,
    initial_registration_batch_size: int = INITIAL_REGISTRATION_BATCH_SIZE,
    max_returned_flights: int = MAX_RETURNED_FLIGHTS,
    queries_per_minute: int = QUERIES_PER_MINUTE,
    raw_output_dir: Path | None = None,
) -> pd.DataFrame:
    if initial_registration_batch_size < 1:
        raise ValueError("initial_registration_batch_size must be at least 1")
    if max_returned_flights < 1:
        raise ValueError("max_returned_flights must be at least 1")
    if queries_per_minute < 1:
        raise ValueError("queries_per_minute must be at least 1")

    if registrations is None:
        from data_engineering.fr24.get_registrations_to_use import (
            get_registrations_to_use,
        )

        registrations = get_registrations_to_use(target_date=target_date)

    registrations = list(registrations)
    raw_output_dir = raw_output_dir or default_raw_batch_dir(target_date)
    flight_datetime_from, flight_datetime_to = _day_bounds(target_date)
    rate_limiter = RateLimiter(queries_per_minute=queries_per_minute)

    batches = _chunked(registrations, initial_registration_batch_size)
    batch_paths: list[Path] = []
    for batch_number, batch in enumerate(batches, start=1):
        batch_path = _batch_output_path(raw_output_dir, batch_number, batch)
        batch_paths.append(batch_path)
        if batch_path.exists():
            print(f"batch {batch_number}/{len(batches)}: using {batch_path}")
            continue

        rows = _fetch_complete_rows(
            registrations=batch,
            flight_datetime_from=flight_datetime_from,
            flight_datetime_to=flight_datetime_to,
            max_returned_flights=max_returned_flights,
            rate_limiter=rate_limiter,
        )
        _write_raw_batch(rows, batch_path)
        print(
            f"batch {batch_number}/{len(batches)}: "
            f"{len(batch)} registrations, {len(rows)} rows, wrote {batch_path}"
        )

    return _read_raw_batches(batch_paths)


def default_output_path(target_date: date) -> Path:
    from data_engineering.utils import OUTPUT_DIR

    return (
        OUTPUT_DIR
        / "data"
        / "flights"
        / "fr24"
        / FR24_FLIGHTS_VERSION
        / f"year={target_date.year}"
        / f"month={target_date.month:02d}"
        / f"day={target_date.day:02d}"
        / "part-0.parquet"
    )


def _add_airport_reference(
    df: pl.DataFrame,
    *,
    code_column: str,
    prefix: str,
    catalogs: dict,
) -> pl.DataFrame:
    from airports.runtime import airport_catalog_for_date

    id_column = f"{prefix}_airport_id"
    ident_column = f"{prefix}_airport_ident"
    flight_date_column = "_airport_flight_date"
    normalized_code_column = "_airport_code"
    df = df.with_columns(
        pl.col("takeoff_time").dt.date().alias(flight_date_column),
        pl.col(code_column)
        .str.strip_chars()
        .str.to_uppercase()
        .alias(normalized_code_column),
    )
    keys = (
        df.select(flight_date_column, normalized_code_column)
        .filter(
            pl.col(normalized_code_column).is_not_null()
            & (pl.col(normalized_code_column) != "")
        )
        .unique()
        .sort(flight_date_column, normalized_code_column)
    )
    mappings = []
    for flight_date, code in keys.iter_rows():
        if flight_date not in catalogs:
            catalogs[flight_date] = airport_catalog_for_date(flight_date)
        airport = catalogs[flight_date].resolve_code(
            code,
            code_priority=(
                "icao_code",
                "gps_code",
                "ident",
                "iata_code",
                "local_code",
            ),
        )
        mappings.append(
            {
                flight_date_column: flight_date,
                normalized_code_column: code,
                id_column: airport.id,
                ident_column: airport.ident,
            }
        )

    mapping = pl.DataFrame(
        mappings,
        schema={
            flight_date_column: pl.Date,
            normalized_code_column: pl.String,
            id_column: pl.Int64,
            ident_column: pl.String,
        },
    )
    return (
        df.join(
            mapping,
            on=[flight_date_column, normalized_code_column],
            how="left",
        )
        .drop(flight_date_column, normalized_code_column)
    )


def _natural_empty_expr(col_name: str, dtype: pl.DataType) -> pl.Expr:
    if dtype == pl.String:
        return pl.lit("").alias(col_name)
    if dtype == pl.Boolean:
        return pl.lit(False).alias(col_name)
    return pl.lit(None).cast(dtype).alias(col_name)


def fr24_raw_to_flights(df: pd.DataFrame | pl.DataFrame) -> pl.DataFrame:
    from flights.flight_type import FLIGHT_POLARS_SCHEMA

    if isinstance(df, pd.DataFrame):
        if df.empty:
            return pl.DataFrame(schema=FLIGHT_POLARS_SCHEMA)
        df = pl.from_pandas(df)
    elif df.is_empty():
        return pl.DataFrame(schema=FLIGHT_POLARS_SCHEMA)

    df = df.select(
        "hex",
        "callsign",
        "datetime_takeoff",
        "datetime_landed",
        "orig_icao",
        "dest_icao_actual",
        "reg",
    )
    df = df.with_columns(
        pl.col("datetime_takeoff", "datetime_landed")
        .str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%SZ", strict=False)
    )
    df = df.rename({
        "hex": "icao",
        "datetime_takeoff": "takeoff_time",
        "datetime_landed": "landing_time",
        "reg": "registration",
    })
    catalogs = {}
    df = _add_airport_reference(
        df,
        code_column="orig_icao",
        prefix="takeoff",
        catalogs=catalogs,
    )
    df = _add_airport_reference(
        df,
        code_column="dest_icao_actual",
        prefix="landing",
        catalogs=catalogs,
    )
    df = (
        df
        .drop("dest_icao_actual", "orig_icao")
        .with_columns(pl.col("icao").str.to_lowercase().alias("icao"))
    )

    return (
        df.with_columns([
            _natural_empty_expr(col_name, dtype)
            for col_name, dtype in FLIGHT_POLARS_SCHEMA.items()
            if col_name not in df.columns
        ])
        .select(list(FLIGHT_POLARS_SCHEMA.keys()))
        .cast(FLIGHT_POLARS_SCHEMA)
    )


def write_fr24_flights(
    df: pd.DataFrame | pl.DataFrame,
    output_path: Path,
    target_date: date,
) -> pl.DataFrame:
    from airports.runtime import airport_catalog_for_date
    from flights.flight_manifest import (
        FlightManifest,
        flight_manifest_path,
    )

    df_flights = fr24_raw_to_flights(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_flights.write_parquet(output_path)
    FlightManifest.from_provenance(
        airport_catalog_for_date(target_date).provenance
    ).write(flight_manifest_path(output_path))
    return df_flights


def main() -> None:
    from data_engineering.fr24.get_registrations_to_use import (
        get_registrations_to_use,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=TARGET_DATE.isoformat())
    parser.add_argument("--n-planes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=INITIAL_REGISTRATION_BATCH_SIZE)
    parser.add_argument("--max-returned-flights", type=int, default=MAX_RETURNED_FLIGHTS)
    parser.add_argument("--queries-per-minute", type=int, default=QUERIES_PER_MINUTE)
    parser.add_argument("--raw-output-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    target_date = date.fromisoformat(args.date)
    registrations = get_registrations_to_use(target_date=target_date)
    if args.n_planes is not None:
        registrations = registrations[:args.n_planes]

    df = get_test_data(
        registrations=registrations,
        target_date=target_date,
        initial_registration_batch_size=args.batch_size,
        max_returned_flights=args.max_returned_flights,
        queries_per_minute=args.queries_per_minute,
        raw_output_dir=args.raw_output_dir,
    )

    output_path = args.output or default_output_path(target_date)
    df_flights = write_fr24_flights(df, output_path, target_date)
    print(f"Wrote {len(df_flights)} flights to {output_path}")


if __name__ == "__main__":
    main()
