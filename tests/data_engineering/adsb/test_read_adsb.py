from datetime import date, datetime
from unittest.mock import Mock

import polars as pl

from data_engineering.adsb import read_adsb


def test_create_adsb_parquet_for_day_passes_pia_or_american_ladd_only(monkeypatch):
    create_parquet_for_day = Mock()
    monkeypatch.setattr(read_adsb, "_parquet_partition_has_data", lambda _: False)
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
