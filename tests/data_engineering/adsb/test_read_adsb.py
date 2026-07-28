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


def test_read_adsb_applies_opensky_preprocessing_after_icao_filter(monkeypatch):
    from data_engineering.opensky import read_opensky_trino_states

    def scan_opensky_fixture(*args, columns=None, **kwargs) -> pl.LazyFrame:
        df = pl.DataFrame(
            {
                "time": [
                    datetime(2026, 1, 2, 0, 0, 0),
                    datetime(2026, 1, 2, 0, 0, 10),
                    datetime(2026, 1, 2, 0, 0, 20),
                ],
                "icao": ["abc123"] * 3,
                "lat": [40.0, 45.0, 40.01],
                "lon": [-75.0, -100.0, -74.99],
                "ground_speed_kt": pl.Series(
                    [250.0] * 3,
                    dtype=pl.Float32,
                ),
                "on_ground": [False] * 3,
                "baro_altitude_ft": pl.Series(
                    [10_000] * 3,
                    dtype=pl.Int32,
                ),
                "callsign": ["TEST1"] * 3,
            }
        )
        return df.select(columns).lazy() if columns else df.lazy()

    monkeypatch.setattr(
        read_opensky_trino_states,
        "scan_cached_opensky_state_vectors_for_day",
        scan_opensky_fixture,
    )

    result = read_adsb_module.read_adsb(
        date(2026, 1, 2),
        columns=["time", "icao", "lat", "lon"],
        icaos=["abc123"],
        source="opensky",
    )

    assert result.get_column("time").to_list() == [
        datetime(2026, 1, 2, 0, 0, 0),
        datetime(2026, 1, 2, 0, 0, 20),
    ]
