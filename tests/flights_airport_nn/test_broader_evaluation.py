from datetime import datetime, timedelta

import polars as pl

from flights_airport_nn.broader_evaluation import (
    mask_reception,
    match_by_time,
    stress_selected,
)


def flights(rows):
    start = datetime(  # noqa: DTZ001 - source parquet timestamps are UTC-naive
        2026, 4, 1, 12
    )  # UTC-naive timestamps match repository parquet schema.
    return pl.DataFrame(
        [
            {
                "icao": icao,
                "takeoff_time": start + timedelta(seconds=offset),
                "landing_time": start + timedelta(seconds=offset + 3600),
                "takeoff_airport_ident": airport,
                "landing_airport_ident": airport,
            }
            for icao, offset, airport in rows
        ]
    )


def test_match_uses_time_not_airport():
    segments = flights([("abc123", 0, "GUESS")])
    labels = flights([("abc123", 60, "TRUTH")])
    matched, stats = match_by_time(segments, labels)
    assert stats["matched_flights"] == 1
    assert matched["raw_takeoff_label"][0] == "TRUTH"
    assert matched["takeoff_airport_ident"][0] == "GUESS"
    assert matched["start_delta_s"][0] == 60


def test_ambiguous_and_distant_matches_excluded():
    segments = flights([("abc123", 0, "A"), ("def123", 0, "B")])
    labels = flights([("abc123", 30, "A"), ("abc123", 60, "B"), ("def123", 1000, "B")])
    matched, stats = match_by_time(segments, labels)
    assert matched.is_empty()
    assert stats["temporal_candidate_pairs"] == 2


def test_conflicting_labels_not_deduplicated():
    segments = flights([("abc123", 0, "A")])
    labels = flights([("abc123", 0, "A"), ("abc123", 0, "B")])
    matched, _ = match_by_time(segments, labels)
    assert matched.is_empty()


def test_mask_preserves_segment_times_and_does_not_use_airports():
    f = flights([("abc123", 0, "A")])
    start = f["takeoff_time"][0]
    messages = pl.DataFrame(
        {
            "icao": ["abc123"] * 5,
            "time": [start + timedelta(seconds=s) for s in [0, 119, 120, 3480, 3481]],
        }
    )
    masked = mask_reception(f, messages, 120)
    assert masked["time"].to_list() == [
        start + timedelta(seconds=120),
        start + timedelta(seconds=3480),
    ]
    assert f["takeoff_time"][0] == start
    assert stress_selected("abc123_2026-04-01_12-00") == stress_selected(
        "abc123_2026-04-01_12-00"
    )


def test_endpoint_windows_merge_overlaps_without_reading_cruise():
    from flights_airport_nn.broader_evaluation import endpoint_windows

    f = flights([("abc123", 0, "A")])
    windows = endpoint_windows(f)
    assert windows.height == 2
    assert (
        windows["_window_end"][0] - windows["_window_start"][0]
    ).total_seconds() == 600
    short = f.with_columns(
        (pl.col("takeoff_time") + pl.duration(minutes=15)).alias("landing_time")
    )
    merged = endpoint_windows(short)
    assert merged.height == 1
    assert (
        merged["_window_end"][0] - merged["_window_start"][0]
    ).total_seconds() == 900
