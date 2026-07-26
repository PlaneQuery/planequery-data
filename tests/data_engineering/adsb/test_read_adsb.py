from datetime import date, datetime

import polars as pl

from data_engineering.adsb import read_adsb as read_adsb_module


def _scan_fixture(*args, **kwargs) -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "time": [
                datetime(2026, 1, 2, 12),
                datetime(2026, 1, 3, 12),
            ],
            "icao": ["abc123", "def456"],
        }
    ).lazy()


def test_read_adsb_is_eager_by_default(monkeypatch):
    monkeypatch.setattr(read_adsb_module, "scan_adsb", _scan_fixture)

    result = read_adsb_module.read_adsb(
        date(2026, 1, 2),
        columns=["time", "icao"],
    )

    assert isinstance(result, pl.DataFrame)
    assert result.get_column("icao").to_list() == ["abc123"]


def test_read_adsb_can_return_lazy_frame(monkeypatch):
    monkeypatch.setattr(read_adsb_module, "scan_adsb", _scan_fixture)

    result = read_adsb_module.read_adsb(
        date(2026, 1, 2),
        columns=["time", "icao"],
        lazy=True,
    )

    assert isinstance(result, pl.LazyFrame)
    assert result.collect().get_column("icao").to_list() == ["abc123"]
