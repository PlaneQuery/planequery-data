from datetime import datetime, timedelta
import importlib

import polars as pl

from flights_algorithm.adsb_filter import filter_adsb_messages


def _trajectory_rows(icao: str, latitude: float) -> list[dict]:
    start = datetime(2026, 1, 1)
    return [
        {
            "time": start + timedelta(seconds=index),
            "icao": icao,
            "lat": latitude if index != 9 else latitude + 20,
            "lon": -75.0,
            "ground_speed_kt": 100.0 if index != 9 else 1_000.0,
            "baro_altitude_ft": 1_000 if index != 9 else 50_000,
            "callsign": f"{icao}-callsign",
        }
        for index in range(19)
    ]


def test_filter_adsb_messages_replaces_outliers_per_icao_and_preserves_schema():
    df_adsb = pl.DataFrame(
        _trajectory_rows("aaa111", 40.0)
        + _trajectory_rows("bbb222", 50.0)
    )

    filtered = filter_adsb_messages(df_adsb)

    outlier_time = datetime(2026, 1, 1) + timedelta(seconds=9)
    corrected = filtered.filter(pl.col("time") == outlier_time).sort("icao")
    assert corrected.get_column("lat").to_list() == [40.0, 50.0]
    assert corrected.get_column("ground_speed_kt").to_list() == [100.0, 100.0]
    assert corrected.get_column("baro_altitude_ft").to_list() == [1_000, 1_000]
    assert corrected.get_column("callsign").to_list() == [
        "aaa111-callsign",
        "bbb222-callsign",
    ]
    assert filtered.schema == df_adsb.schema


def test_filter_adsb_messages_backward_then_forward_fills_missing_values():
    start = datetime(2026, 1, 1)
    df_adsb = pl.DataFrame(
        {
            "time": [
                start,
                start + timedelta(seconds=1),
                start,
                start + timedelta(seconds=1),
            ],
            "icao": ["aaa111", "aaa111", "bbb222", "bbb222"],
            "ground_speed_kt": [1.0, None, None, 3.0],
        }
    )

    filtered = filter_adsb_messages(
        df_adsb,
        column_kernels={"ground_speed_kt": 5},
    )

    assert filtered.get_column("ground_speed_kt").to_list() == [1.0, 1.0, 3.0, 3.0]


def test_segment_adsb_df_filters_before_identifying_segments(monkeypatch):
    flights_main = importlib.import_module("flights_algorithm.main")
    calls = []

    def record_filter(df_adsb):
        calls.append("filter")
        return df_adsb

    def record_segmentation(messages):
        calls.append("segment")
        return []

    monkeypatch.setattr(flights_main, "filter_adsb_messages", record_filter)
    monkeypatch.setattr(flights_main, "identify_flight_segments", record_segmentation)

    start = datetime(2026, 1, 1)
    df_adsb = pl.DataFrame(
        {
            "time": [start, start + timedelta(seconds=1)],
            "icao": ["aaa111", "aaa111"],
            "lat": [40.0, 40.1],
            "lon": [-75.0, -74.9],
        }
    )

    assert flights_main._segment_adsb_df(df_adsb) == []
    assert calls == ["filter", "segment"]
