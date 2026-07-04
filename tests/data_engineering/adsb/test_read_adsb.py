from datetime import date, datetime
from unittest.mock import Mock

import polars as pl

from data_engineering.adsb import read_adsb


def test_create_adsb_parquet_for_day_passes_pia_or_american_ladd_only(monkeypatch):
    create_parquet_for_day = Mock()
    monkeypatch.setattr(read_adsb, "_parquet_partition_has_data", lambda _: False)
    monkeypatch.setattr(read_adsb, "_legacy_parquet_path", lambda _: Mock(exists=lambda: False))
    monkeypatch.setattr(
        "data_engineering.adsb.download_adsb_to_parquet.create_parquet_for_day",
        create_parquet_for_day,
    )

    target_date = date(2026, 3, 1)

    read_adsb.create_adsb_parquet_for_day(
        target_date,
        pia_or_american_ladd_only=True,
    )

    create_parquet_for_day.assert_called_once_with(
        target_date,
        pia_or_american_ladd_only=True,
    )


def test_create_adsb_parquet_for_day_skips_creation_when_legacy_v5_exists(monkeypatch):
    create_parquet_for_day = Mock()
    monkeypatch.setattr(read_adsb, "_parquet_partition_has_data", lambda _: False)
    monkeypatch.setattr(read_adsb, "_legacy_parquet_path", lambda _: Mock(exists=lambda: True))
    monkeypatch.setattr(
        "data_engineering.adsb.download_adsb_to_parquet.create_parquet_for_day",
        create_parquet_for_day,
    )

    read_adsb.create_adsb_parquet_for_day(date(2026, 3, 1))

    create_parquet_for_day.assert_not_called()


def test_parquet_read_paths_falls_back_to_legacy_v5_when_v6_is_empty(
    tmp_path,
    monkeypatch,
):
    target_date = date(2026, 3, 1)
    v6_day_dir = tmp_path / "v6" / "year=2026" / "month=03" / "day=01"
    (v6_day_dir / "icao_bucket=0").mkdir(parents=True)
    v5_day_dir = tmp_path / "v5" / "year=2026" / "month=03" / "day=01"
    v5_day_dir.mkdir(parents=True)
    legacy_path = v5_day_dir / "part-00000.parquet"
    legacy_path.touch()
    monkeypatch.setattr(read_adsb, "ADSBLOL_PARQUET_ROOT", tmp_path / "v6")
    monkeypatch.setattr(read_adsb, "ADSBLOL_LEGACY_PARQUET_ROOT", tmp_path / "v5")

    assert read_adsb._parquet_read_paths(target_date) == str(legacy_path)
    assert read_adsb._parquet_read_paths(target_date, [0]) == str(legacy_path)


def test_read_adsbx_reads_whole_day_then_filters_icaos(
    tmp_path,
    monkeypatch,
):
    target_date = date(2026, 3, 1)
    bucket_0_dir = tmp_path / "year=2026" / "month=03" / "day=01" / "icao_bucket=0"
    bucket_1_dir = tmp_path / "year=2026" / "month=03" / "day=01" / "icao_bucket=1"
    bucket_0_dir.mkdir(parents=True)
    bucket_1_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "time": [
                datetime(2026, 3, 1, 12, 0),
            ],
            "icao": ["010001"],
            "callsign": ["OTHER"],
        }
    ).write_parquet(bucket_0_dir / "data.parquet")
    pl.DataFrame(
        {
            "time": [
                datetime(2026, 3, 1, 12, 0),
                datetime(2026, 3, 2, 12, 0),
            ],
            "icao": ["200001", "200001"],
            "callsign": [" TEST 1 ", "TEST2"],
        }
    ).write_parquet(bucket_1_dir / "data.parquet")
    monkeypatch.setattr(read_adsb, "ADSBX_PARQUET_ROOT", tmp_path)

    df = read_adsb.read_adsbx(
        target_date,
        columns=["time", "icao", "callsign"],
        icaos=["200001"],
    )

    assert df["icao"].to_list() == ["200001"]
    assert df["callsign"].to_list() == ["TEST1"]


def test_scan_adsb_pia_or_american_ladd_only_keeps_all_rows_for_matching_icaos(
    tmp_path,
    monkeypatch,
):
    target_date = date(2026, 3, 1)
    day_dir = tmp_path / "year=2026" / "month=03" / "day=01" / "icao_bucket=0"
    day_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "time": [
                datetime(2026, 3, 1, 12, 0),
                datetime(2026, 3, 1, 12, 1),
                datetime(2026, 3, 1, 12, 2),
                datetime(2026, 3, 1, 12, 3),
            ],
            "icao": ["010001", "010001", "a00002", "010003"],
            "pia": [False, True, False, False],
            "ladd": [False, False, True, True],
        }
    ).write_parquet(day_dir / "data.parquet")
    monkeypatch.setattr(read_adsb, "ADSBLOL_PARQUET_ROOT", tmp_path)
    monkeypatch.setattr(read_adsb, "create_adsb_parquet_for_day", lambda *_args, **_kwargs: None)

    df = read_adsb.scan_adsb(
        target_date,
        columns=["icao"],
        pia_or_american_ladd_only=True,
    ).collect()

    assert df["icao"].to_list() == ["010001", "010001", "a00002"]
