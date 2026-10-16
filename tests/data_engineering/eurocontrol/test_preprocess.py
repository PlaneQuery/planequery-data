import gzip
from datetime import date, datetime
from zipfile import ZipFile

import polars as pl
import pytest

from data_engineering.eurocontrol import preprocess


TARGET_DATE = date(2025, 8, 1)
CSV_CONTENT = (
    '"AC Registration","ADEP","ADES","AC Type","AC Operator",'
    '"ACTUAL OFF BLOCK TIME","ACTUAL ARRIVAL TIME"\n'
    '"N-123AB","kjfk","egll","GLEX","TST",'
    '"01-08-2025 00:20:00","01-08-2025 07:15:00"\n'
)


def write_monthly_zip(path, member_name, content=CSV_CONTENT):
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w") as archive:
        archive.writestr(member_name, gzip.compress(content.encode()))


@pytest.mark.parametrize(
    ("target_date", "expected_filename"),
    [
        (date(2024, 2, 10), "Flights_20240201_20240229.csv"),
        (date(2024, 6, 10), "Flights_20240601_20240630.csv"),
        (date(2025, 8, 10), "Flights_20250801_20250831.csv"),
    ],
)
def test_eurocontrol_flights_filename_uses_calendar_month(
    target_date,
    expected_filename,
):
    assert preprocess.eurocontrol_flights_filename(target_date) == expected_filename


def test_prepare_eurocontrol_month_parquet_from_monthly_zip(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(preprocess, "OUTPUT_DIR", tmp_path)
    archive_path = preprocess.eurocontrol_archive_path(TARGET_DATE)
    member_name = (
        f"202508/{preprocess.eurocontrol_flights_filename(TARGET_DATE)}.gz"
    )
    write_monthly_zip(archive_path, member_name)

    result = preprocess.prepare_eurocontrol_month_parquet(TARGET_DATE)

    assert result == (
        tmp_path / "data/intermediate/eurocontrol/202508.parquet"
    )
    assert pl.read_parquet(result).to_dict(as_series=False) == {
        "registration": ["N123AB"],
        "takeoff_airport_code": ["KJFK"],
        "landing_airport_code": ["EGLL"],
        "aircraft_type": ["GLEX"],
        "owner": ["TST"],
        "takeoff_time": [datetime(2025, 8, 1, 0, 20)],
        "landing_time": [datetime(2025, 8, 1, 7, 15)],
    }
    assert list(archive_path.parent.iterdir()) == [archive_path]


def test_prepare_eurocontrol_month_parquet_reuses_existing_intermediate(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(preprocess, "OUTPUT_DIR", tmp_path)
    intermediate_path = preprocess.eurocontrol_intermediate_path(TARGET_DATE)
    intermediate_path.parent.mkdir(parents=True)
    intermediate_path.write_bytes(b"already prepared")

    assert (
        preprocess.prepare_eurocontrol_month_parquet(TARGET_DATE)
        == intermediate_path
    )


def test_prepare_eurocontrol_month_parquet_requires_monthly_input(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(preprocess, "OUTPUT_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="202508.zip"):
        preprocess.prepare_eurocontrol_month_parquet(TARGET_DATE)


def test_prepare_eurocontrol_month_parquet_rejects_archive_without_flights(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(preprocess, "OUTPUT_DIR", tmp_path)
    archive_path = preprocess.eurocontrol_archive_path(TARGET_DATE)
    archive_path.parent.mkdir(parents=True)
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("202508/Route_2508.csv.gz", b"not needed")

    with pytest.raises(FileNotFoundError, match="Flights_20250801_20250831"):
        preprocess.prepare_eurocontrol_month_parquet(TARGET_DATE)
