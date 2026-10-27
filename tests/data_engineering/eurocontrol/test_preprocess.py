import gzip
from datetime import date, datetime
from zipfile import ZipFile

import polars as pl
import pytest

from data_engineering.eurocontrol import preprocess


TARGET_DATE = date(2025, 8, 1)
CSV_CONTENT = (
    '"ECTRL ID","AC Registration","ADEP","ADES","AC Type","AC Operator",'
    '"ACTUAL OFF BLOCK TIME","ACTUAL ARRIVAL TIME"\n'
    '"123","N-123AB","kjfk","egll","GLEX","TST",'
    '"01-08-2025 00:20:00","01-08-2025 07:15:00"\n'
)
FIR_CONTENT = (
    '"ECTRL ID","Sequence Number","FIR ID","Entry Time","Exit Time"\n'
    '"123","0","TAXI_OUT","01-08-2025 00:20:00","01-08-2025 00:45:00"\n'
    '"123","1","KZNYFIR","01-08-2025 00:45:00","01-08-2025 01:00:00"\n'
    '"999","0","TAXI_OUT","01-08-2025 00:20:00","01-08-2025 00:55:00"\n'
)


def write_monthly_zip(path, member_name, content=CSV_CONTENT, fir_content=FIR_CONTENT):
    def encode(text):
        raw = text.encode()
        return gzip.compress(raw) if member_name.endswith(".gz") else raw

    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w") as archive:
        archive.writestr(member_name, encode(content))
        if fir_content is not None:
            archive.writestr(
                member_name.replace("Flights_", "Flight_FIRs_Actual_"),
                encode(fir_content),
            )


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


@pytest.mark.parametrize("suffix", ["", ".gz"])
def test_prepare_eurocontrol_month_parquet_from_monthly_zip(
    monkeypatch,
    tmp_path,
    suffix,
):
    monkeypatch.setattr(preprocess, "OUTPUT_DIR", tmp_path)
    archive_path = preprocess.eurocontrol_archive_path(TARGET_DATE)
    member_name = (
        f"202508/{preprocess.eurocontrol_flights_filename(TARGET_DATE)}{suffix}"
    )
    write_monthly_zip(archive_path, member_name)
    # The old off-block-based cache must not be reused.
    old_cache = tmp_path / "data/intermediate/eurocontrol/202508.parquet"
    old_cache.parent.mkdir(parents=True)
    old_cache.write_bytes(b"old off-block data")

    result = preprocess.prepare_eurocontrol_month_parquet(TARGET_DATE)

    assert result == (
        tmp_path / "data/intermediate/eurocontrol/version=2/202508.parquet"
    )
    assert pl.read_parquet(result).to_dict(as_series=False) == {
        "registration": ["N123AB"],
        "takeoff_airport_code": ["KJFK"],
        "landing_airport_code": ["EGLL"],
        "aircraft_type": ["GLEX"],
        "owner": ["TST"],
        "takeoff_time": [datetime(2025, 8, 1, 0, 45)],
        "landing_time": [datetime(2025, 8, 1, 7, 15)],
    }
    assert list(archive_path.parent.iterdir()) == [archive_path]


@pytest.mark.parametrize("exit_time", [None, "", "invalid"])
def test_missing_taxi_exit_does_not_fall_back_to_off_block(
    monkeypatch, tmp_path, exit_time
):
    monkeypatch.setattr(preprocess, "OUTPUT_DIR", tmp_path)
    fir_content = FIR_CONTENT.replace('"123","0","TAXI_OUT"', '"456","0","TAXI_OUT"')
    if exit_time is not None:
        fir_content += f'"123","0","TAXI_OUT","01-08-2025 00:20:00","{exit_time}"\n'
    write_monthly_zip(
        preprocess.eurocontrol_archive_path(TARGET_DATE),
        preprocess.eurocontrol_flights_filename(TARGET_DATE) + ".gz",
        fir_content=fir_content,
    )

    result = pl.read_parquet(preprocess.prepare_eurocontrol_month_parquet(TARGET_DATE))

    assert result.height == 1
    assert result["takeoff_time"].to_list() == [None]
    assert result["landing_time"].to_list() == [datetime(2025, 8, 1, 7, 15)]


def test_prepare_eurocontrol_month_parquet_requires_actual_firs(monkeypatch, tmp_path):
    monkeypatch.setattr(preprocess, "OUTPUT_DIR", tmp_path)
    write_monthly_zip(
        preprocess.eurocontrol_archive_path(TARGET_DATE),
        preprocess.eurocontrol_flights_filename(TARGET_DATE) + ".gz",
        fir_content=None,
    )

    with pytest.raises(FileNotFoundError, match="Flight_FIRs_Actual"):
        preprocess.prepare_eurocontrol_month_parquet(TARGET_DATE)
    assert not preprocess.eurocontrol_intermediate_path(TARGET_DATE).exists()


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


@pytest.mark.parametrize("flights_suffix", ["", ".gz"])
@pytest.mark.parametrize("firs_suffix", ["", ".gz"])
def test_prepare_eurocontrol_month_parquet_from_extracted_folder(
    monkeypatch, tmp_path, flights_suffix, firs_suffix
):
    monkeypatch.setattr(preprocess, "OUTPUT_DIR", tmp_path)
    extracted_dir = tmp_path / "data/raw/eurocontrol/202508"
    extracted_dir.mkdir(parents=True)
    inputs = {}
    for filename, suffix, content in (
        (preprocess.eurocontrol_flights_filename(TARGET_DATE), flights_suffix, CSV_CONTENT),
        (preprocess.eurocontrol_firs_filename(TARGET_DATE), firs_suffix, FIR_CONTENT),
    ):
        path = extracted_dir / (filename + suffix)
        raw = content.encode()
        inputs[path] = gzip.compress(raw) if suffix else raw
        path.write_bytes(inputs[path])

    result = pl.read_parquet(preprocess.prepare_eurocontrol_month_parquet(TARGET_DATE))

    assert result.height == 1
    assert result["takeoff_time"].to_list() == [datetime(2025, 8, 1, 0, 45)]
    assert result["landing_time"].to_list() == [datetime(2025, 8, 1, 7, 15)]
    assert {path: path.read_bytes() for path in extracted_dir.iterdir()} == inputs
    assert not preprocess.eurocontrol_archive_path(TARGET_DATE).exists()


def test_extracted_folder_requires_actual_firs(monkeypatch, tmp_path):
    monkeypatch.setattr(preprocess, "OUTPUT_DIR", tmp_path)
    extracted_dir = tmp_path / "data/raw/eurocontrol/202508"
    extracted_dir.mkdir(parents=True)
    (extracted_dir / preprocess.eurocontrol_flights_filename(TARGET_DATE)).write_text(
        CSV_CONTENT
    )

    with pytest.raises(FileNotFoundError, match="Flight_FIRs_Actual"):
        preprocess.prepare_eurocontrol_month_parquet(TARGET_DATE)
    assert not preprocess.eurocontrol_intermediate_path(TARGET_DATE).exists()


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
