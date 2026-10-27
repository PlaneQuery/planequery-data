import calendar
import gzip
import os
import shutil
import tempfile
from contextlib import contextmanager
from datetime import date
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterator
from zipfile import BadZipFile, ZipFile, ZipInfo

import polars as pl

from data_engineering.utils import OUTPUT_DIR


def eurocontrol_month_bounds(target_date: date) -> tuple[date, date]:
    last_day = calendar.monthrange(target_date.year, target_date.month)[1]
    return (
        target_date.replace(day=1),
        target_date.replace(day=last_day),
    )


def eurocontrol_archive_path(target_date: date) -> Path:
    return (
        OUTPUT_DIR
        / "data"
        / "raw"
        / "eurocontrol"
        / f"{target_date:%Y%m}.zip"
    )


def eurocontrol_intermediate_path(target_date: date) -> Path:
    return (
        OUTPUT_DIR
        / "data"
        / "intermediate"
        / "eurocontrol"
        / "version=2"
        / f"{target_date:%Y%m}.parquet"
    )


def eurocontrol_flights_filename(target_date: date) -> str:
    month_start, month_end = eurocontrol_month_bounds(target_date)
    return f"Flights_{month_start:%Y%m%d}_{month_end:%Y%m%d}.csv"


def eurocontrol_firs_filename(target_date: date) -> str:
    month_start, month_end = eurocontrol_month_bounds(target_date)
    return f"Flight_FIRs_Actual_{month_start:%Y%m%d}_{month_end:%Y%m%d}.csv"


def _archive_member(archive: ZipFile, filename: str) -> ZipInfo:
    matches = [
        info
        for info in archive.infolist()
        if not info.is_dir() and PurePosixPath(info.filename).name == filename
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {filename!r} member in {archive.filename}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _copy_to_path(source: BinaryIO, destination: Path) -> None:
    with destination.open("wb") as output:
        shutil.copyfileobj(source, output)


def _extract_archive_csv(
    csv_filename: str,
    archive_path: Path,
    destination: Path,
) -> None:
    gzip_filename = csv_filename + ".gz"
    try:
        with ZipFile(archive_path) as archive:
            try:
                member = _archive_member(archive, gzip_filename)
            except FileNotFoundError as gzip_error:
                try:
                    member = _archive_member(archive, csv_filename)
                except FileNotFoundError:
                    raise gzip_error

            with archive.open(member) as archive_source:
                if PurePosixPath(member.filename).name.endswith(".gz"):
                    with gzip.GzipFile(fileobj=archive_source) as csv_source:
                        _copy_to_path(csv_source, destination)
                else:
                    _copy_to_path(archive_source, destination)
    except BadZipFile as error:
        raise ValueError(f"Invalid Eurocontrol ZIP archive: {archive_path}") from error
    except gzip.BadGzipFile as error:
        raise ValueError(
            f"Invalid Eurocontrol gzip member in archive: {archive_path}"
        ) from error


@contextmanager
def _month_csvs(target_date: date) -> Iterator[tuple[Path, Path]]:
    archive_path = eurocontrol_archive_path(target_date)
    flights_filename = eurocontrol_flights_filename(target_date)
    firs_filename = eurocontrol_firs_filename(target_date)
    if not archive_path.is_file():
        extracted_dir = archive_path.with_suffix("")
        paths = []
        for filename in (flights_filename, firs_filename):
            candidates = (extracted_dir / (filename + ".gz"), extracted_dir / filename)
            path = next((path for path in candidates if path.is_file()), None)
            if path is None:
                raise FileNotFoundError(
                    "Eurocontrol monthly input not found. Expected the raw archive "
                    f"{archive_path} or extracted file {candidates[0]} or {candidates[1]}"
                )
            paths.append(path)
        yield paths[0], paths[1]
        return

    with tempfile.TemporaryDirectory(prefix="planequery-eurocontrol-") as temp_dir:
        flights_path = Path(temp_dir) / flights_filename
        firs_path = Path(temp_dir) / firs_filename
        _extract_archive_csv(flights_filename, archive_path, flights_path)
        _extract_archive_csv(firs_filename, archive_path, firs_path)
        yield flights_path, firs_path


def _datetime_expr(column_name: str, alias: str) -> pl.Expr:
    return (
        pl.col(column_name)
        .str.strptime(
            pl.Datetime("ms"),
            format="%d-%m-%Y %H:%M:%S",
            strict=False,
        )
        .alias(alias)
    )


def _normalized_month_scan(csv_path: Path, firs_path: Path) -> pl.LazyFrame:
    taxi_out = (
        pl.scan_csv(firs_path, schema_overrides={"ECTRL ID": pl.String})
        .filter(pl.col("FIR ID") == "TAXI_OUT")
        .select("ECTRL ID", _datetime_expr("Exit Time", "takeoff_time"))
    )
    flights = pl.scan_csv(csv_path, schema_overrides={"ECTRL ID": pl.String})
    # Missing taxi-out times stay null rather than falling back to off-block time.
    return flights.join(
        taxi_out, on="ECTRL ID", how="left", validate="m:1"
    ).select(
        pl.col("AC Registration")
        .str.strip_chars()
        .str.to_uppercase()
        .str.replace_all("-", "")
        .alias("registration"),
        pl.col("ADEP")
        .str.strip_chars()
        .str.to_uppercase()
        .alias("takeoff_airport_code"),
        pl.col("ADES")
        .str.strip_chars()
        .str.to_uppercase()
        .alias("landing_airport_code"),
        pl.col("AC Type").alias("aircraft_type"),
        pl.col("AC Operator").alias("owner"),
        pl.col("takeoff_time"),
        _datetime_expr("ACTUAL ARRIVAL TIME", "landing_time"),
    )


def prepare_eurocontrol_month_parquet(target_date: date) -> Path:
    """Cache a normalized month from a raw ZIP or an extracted monthly folder.

    ZIP members are extracted through a temporary directory and removed when
    preprocessing finishes. Existing extracted CSV/gzip files are read directly
    and retained unchanged.
    """
    destination = eurocontrol_intermediate_path(target_date)
    if destination.is_file():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        with _month_csvs(target_date) as (csv_path, firs_path):
            _normalized_month_scan(csv_path, firs_path).sink_parquet(temporary_path)
        os.replace(temporary_path, destination)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return destination
