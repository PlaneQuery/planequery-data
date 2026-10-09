from datetime import date, datetime

import polars as pl

from data_engineering.adsb import canonical
from data_engineering.adsb.canonical import ADSBSource
from data_engineering.adsb.partitions import ICAO_BUCKET_COUNT, icao_bucket


def test_all_sources_share_canonical_layout(monkeypatch, tmp_path):
    monkeypatch.setattr(canonical, "ADSB_PARQUET_ROOT", tmp_path)

    expected_versions = {
        ADSBSource.ADSBLOL: 6,
        ADSBSource.ADSBX: 3,
        ADSBSource.OPENSKY: 4,
    }
    for source, version in expected_versions.items():
        assert canonical.adsb_day_dir(source, date(2024, 6, 1)) == (
            tmp_path
            / f"source={source.value}"
            / f"version={version}"
            / "year=2024"
            / "month=06"
            / "day=01"
        )


def test_canonical_scanner_prunes_icao_buckets(monkeypatch, tmp_path):
    monkeypatch.setattr(canonical, "ADSB_PARQUET_ROOT", tmp_path)
    day = date(2024, 6, 1)
    source = ADSBSource.OPENSKY
    rows = [
        ("010000", datetime(2024, 6, 1, 1)),
        ("f10000", datetime(2024, 6, 1, 2)),
    ]
    for icao, timestamp in rows:
        output = (
            canonical.adsb_day_dir(source, day)
            / f"icao_bucket={icao_bucket(icao)}"
            / "data.parquet"
        )
        pl.DataFrame({"time": [timestamp], "icao": [icao]}).write_parquet(
            output,
            mkdir=True,
        )
    canonical.mark_adsb_day_complete(canonical.adsb_day_dir(source, day))

    result = canonical.scan_adsb_parquet(
        source,
        [day],
        ["time", "icao"],
        icaos=["010000"],
    ).collect()

    assert result.get_column("icao").to_list() == ["010000"]
    assert ICAO_BUCKET_COUNT == 8
