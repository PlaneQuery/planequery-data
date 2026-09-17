from datetime import datetime, timedelta

import polars as pl

from flights_airport_nn.error_audit import airport_evidence, classify


def test_corroboration_rejects_fast_overflight_and_brief_single_point():
    airport = {"latitude_deg": "40", "longitude_deg": "-75", "elevation_ft": "100"}
    points = pl.DataFrame(
        {
            "time": [
                datetime(2026, 4, 1) + timedelta(seconds=i)  # noqa: DTZ001
                for i in (0, 5, 10)
            ],
            "lat": [40.0] * 3,
            "lon": [-75.0] * 3,
            "baro_altitude_ft": [200.0] * 3,
            "ground_speed_kt": [80.0] * 3,
            "on_ground": [False] * 3,
        }
    )
    assert airport_evidence(points, airport)["strong"]
    assert not airport_evidence(points.head(1), airport)["strong"]
    assert not airport_evidence(
        points.with_columns(pl.lit(200.0).alias("ground_speed_kt")), airport
    )["strong"]
    assert not airport_evidence(
        points.with_columns(pl.lit(5000.0).alias("baro_altitude_ft")), airport
    )["strong"]
    grounded = points.with_columns(
        pl.lit(None).alias("baro_altitude_ft"),
        pl.lit(None).alias("ground_speed_kt"),
        pl.lit(True).alias("on_ground"),
    )
    assert airport_evidence(grounded, airport)["strong"]


def test_audit_is_symmetric_and_leaves_nearby_or_unsupported_cases_unresolved():
    yes, no = {"strong": True}, {"strong": False}
    assert classify(yes, no, no, 20, None) == "likely_model_error"
    assert classify(no, yes, no, 20, None) == "likely_label_or_segment_error"
    assert classify(no, no, yes, 20, 20) == "likely_label_or_segment_error"
    assert classify(yes, yes, no, 20, None) == "conflicting_track_evidence"
    assert classify(yes, no, no, 5, None) == "unresolved"
    assert classify(no, no, no, 20, None) == "unresolved"
