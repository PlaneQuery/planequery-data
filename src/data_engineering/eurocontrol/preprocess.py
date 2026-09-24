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
        / f"{target_date:%Y%m}.parquet"
    )


def eurocontrol_flights_filename(target_date: date) -> str:
    month_start, month_end = eurocontrol_month_bounds(target_date)
    return f"Flights_{month_start:%Y%m%d}_{month_end:%Y%m%d}.csv"


def legacy_eurocontrol_month_dir(target_date: date) -> Path:
    return (
        OUTPUT_DIR
        / "data"
        / "raw"
        / "eurocontrol"
        / f"{target_date:%Y%m}"
    )


def legacy_eurocontrol_flights_gzip_path(target_date: date) -> Path:
    return legacy_eurocontrol_month_dir(target_date) / (
        eurocontrol_flights_filename(target_date) + ".gz"
    )


def legacy_eurocontrol_flights_csv_path(target_date: date) -> Path:
    return (
        legacy_eurocontrol_month_dir(target_date)
        / "extract"
        / eurocontrol_flights_filename(target_date)
    )


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
    target_date: date,
    archive_path: Path,
    destination: Path,
) -> None:
    csv_filename = eurocontrol_flights_filename(target_date)
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
def _temporary_month_csv(target_date: date) -> Iterator[Path]:
    archive_path = eurocontrol_archive_path(target_date)
    legacy_csv_path = legacy_eurocontrol_flights_csv_path(target_date)
    if not archive_path.is_file() and legacy_csv_path.is_file():
        yield legacy_csv_path
        return

    with tempfile.TemporaryDirectory(prefix="planequery-eurocontrol-") as temp_dir:
        temporary_csv_path = Path(temp_dir) / eurocontrol_flights_filename(target_date)
        if archive_path.is_file():
            _extract_archive_csv(target_date, archive_path, temporary_csv_path)
            yield temporary_csv_path
            return

        legacy_gzip_path = legacy_eurocontrol_flights_gzip_path(target_date)
        if legacy_gzip_path.is_file():
            try:
                with gzip.open(legacy_gzip_path, "rb") as source:
                    _copy_to_path(source, temporary_csv_path)
            except gzip.BadGzipFile as error:
                raise ValueError(
                    f"Invalid Eurocontrol gzip file: {legacy_gzip_path}"
                ) from error
            yield temporary_csv_path
            return

        raise FileNotFoundError(
            "Eurocontrol monthly input not found. Expected the raw archive "
            f"{archive_path} or legacy input {legacy_csv_path} or "
            f"{legacy_gzip_path}"
        )


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


def _normalized_month_scan(csv_path: Path) -> pl.LazyFrame:
    return pl.scan_csv(csv_path).select(
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
        _datetime_expr("ACTUAL OFF BLOCK TIME", "takeoff_time"),
        _datetime_expr("ACTUAL ARRIVAL TIME", "landing_time"),
    )


def prepare_eurocontrol_month_parquet(target_date: date) -> Path:
    """Create and cache one normalized monthly Parquet from a raw ZIP archive.

    Only the original archive and normalized Parquet are retained. Nested gzip
    and CSV data are streamed through a temporary directory and removed when
    preprocessing finishes. Existing legacy CSV/gzip inputs remain readable.
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

        with _temporary_month_csv(target_date) as csv_path:
            _normalized_month_scan(csv_path).sink_parquet(temporary_path)
        os.replace(temporary_path, destination)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return destination
