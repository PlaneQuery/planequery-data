from datetime import UTC, datetime, timedelta

import polars as pl

from flights_airport_nn.evaluate_native_sources import temporal_matches


def test_off_block_matching_uses_signed_bounds_and_landing_time():
    start = datetime(2024, 6, 1, 10, tzinfo=UTC)
    offsets = [-301, -300, 1800, 3600, 3601, 1200]
    refs = pl.DataFrame(
        {
            "reference_id": [str(i) for i in range(6)],
            "reference_source": ["EUROCONTROL"] * 6,
            "icao": [str(i) for i in range(6)],
            "takeoff_time": [start] * 6,
            "landing_time": [start + timedelta(hours=2)] * 6,
            "start_min_s": [-300] * 6,
            "start_max_s": [3600] * 6,
            "end_tolerance_s": [900] * 6,
            "takeoff_truth": ["EGLL"] * 6,
        }
    )
    segments = pl.DataFrame(
        {
            "flight_id": [f"native{i}" for i in range(6)],
            "icao": [str(i) for i in range(6)],
            "takeoff_time": [start + timedelta(seconds=o) for o in offsets],
            "landing_time": [
                start + timedelta(hours=2, seconds=901 if i == 5 else 0)
                for i in range(6)
            ],
        }
    )
    matched = temporal_matches(segments, refs)
    assert set(matched["reference_id"]) == {"1", "2", "3"}
    assert matched["unique_match"].all()
    assert temporal_matches(
        segments, refs.with_columns(pl.lit("WRONG").alias("takeoff_truth"))
    ).equals(matched)
