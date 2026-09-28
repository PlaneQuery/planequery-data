from datetime import date, datetime

import polars as pl
import pytest

from data_engineering.adsb import read_adsb as read_adsb_module
from data_engineering.adsb.partitions import query_icao_buckets
from data_engineering.adsb.read_adsb import ADSBSource


def _scan_fixture(source, days, columns, **kwargs) -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "time": [
                datetime(2026, 1, 2, 12),
                datetime(2026, 1, 3, 12),
            ],
            "icao": ["abc123", "def456"],
        }
    ).select(columns).lazy()


def _stub_canonical_data(monkeypatch):
    monkeypatch.setattr(
        read_adsb_module,
        "ensure_adsb_day",
        lambda source, day: False,
    )
    monkeypatch.setattr(
        read_adsb_module,
        "scan_adsb_parquet",
        _scan_fixture,
    )


def test_read_adsb_is_eager_by_default(monkeypatch):
    _stub_canonical_data(monkeypatch)

    result = read_adsb_module.read_adsb(
        date(2026, 1, 2),
        columns=["time", "icao"],
    )

    assert isinstance(result, pl.DataFrame)
    assert result.get_column("icao").to_list() == ["abc123"]


def test_read_adsb_can_return_lazy_frame(monkeypatch):
    _stub_canonical_data(monkeypatch)

    result = read_adsb_module.read_adsb(
        date(2026, 1, 2),
        columns=["time", "icao"],
        lazy=True,
    )

    assert isinstance(result, pl.LazyFrame)
    assert result.collect().get_column("icao").to_list() == ["abc123"]


@pytest.mark.parametrize("source", list(ADSBSource))
def test_read_adsb_uses_canonical_scanner_for_every_source(monkeypatch, source):
    _stub_canonical_data(monkeypatch)

    result = read_adsb_module.read_adsb(
        date(2026, 1, 2),
        columns=["time", "icao"],
        source=source,
    )

    assert result.get_column("icao").to_list() == ["abc123"]


def test_read_adsb_requires_adsb_source_enum():
    with pytest.raises(TypeError, match="source must be an ADSBSource"):
        read_adsb_module.read_adsb(date(2026, 1, 2), source="unknown")


def test_malformed_icao_falls_back_to_scanning_all_buckets():
    assert query_icao_buckets(icaos=["abc123", "not-an-icao"]) is None
