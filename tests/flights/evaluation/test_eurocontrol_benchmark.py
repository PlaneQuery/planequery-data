from datetime import datetime, timedelta

import polars as pl

from flights.evaluation.eurocontrol_benchmark import match_eurocontrol


def test_offblock_matching_accepts_taxi_time_but_rejects_ambiguity_and_conflicts():
    start = datetime(2024, 6, 12, 12)  # noqa: DTZ001 - source timestamps are naive UTC
    refs = pl.DataFrame(
        {
            "icao": ["abc123"],
            "reference_id": ["one"],
            "time_conflict": [False],
            "takeoff_time": [start],
            "landing_time": [start + timedelta(hours=2)],
        }
    )
    segments = refs.select("icao", "takeoff_time", "landing_time").with_columns(
        (pl.col("takeoff_time") + pl.duration(minutes=40)).alias("takeoff_time"),
        pl.lit(0).alias("segment"),
    )
    assert match_eurocontrol(segments, refs).height == 1
    too_late = segments.with_columns(pl.col("takeoff_time") + pl.duration(minutes=21))
    assert match_eurocontrol(too_late, refs).is_empty()
    ambiguous = pl.concat([segments, segments.with_columns(pl.lit(1).alias("segment"))])
    assert match_eurocontrol(ambiguous, refs).is_empty()
    assert match_eurocontrol(
        segments, refs.with_columns(pl.lit(True).alias("time_conflict"))
    ).is_empty()
