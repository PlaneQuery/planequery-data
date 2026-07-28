from datetime import datetime, timedelta

from flights_algorithm.main import AdsbPositionMessage, identify_flight_segments


ICAO = "abc123"
START = datetime(2026, 3, 1)


def message(
    minutes: int,
    lat: float,
    lon: float,
    *,
    speed: float,
    altitude: float | None,
    track: float | None,
    on_ground: bool = False,
) -> AdsbPositionMessage:
    return AdsbPositionMessage(
        time=START + timedelta(minutes=minutes),
        icao=ICAO,
        lat=lat,
        lon=lon,
        ground_speed_kt=speed,
        track_deg=track,
        on_ground=on_ground,
        baro_altitude_ft=altitude,
    )


def messages_with_gap(
    *,
    gap_end_lon: float,
    before_gap_track: float,
    after_gap_track: float,
) -> list[AdsbPositionMessage]:
    return [
        message(0, 0.0, 0.0, speed=0.0, altitude=0.0, track=90.0, on_ground=True),
        message(2, 0.0, 0.1, speed=180.0, altitude=2_000.0, track=90.0),
        message(30, 0.0, 2.5, speed=300.0, altitude=25_000.0, track=90.0),
        message(60, 0.0, 5.0, speed=300.0, altitude=30_000.0, track=before_gap_track),
        message(180, 0.0, gap_end_lon, speed=300.0, altitude=30_000.0, track=after_gap_track),
        message(200, 0.0, gap_end_lon + 1.5, speed=300.0, altitude=25_000.0, track=90.0),
        message(220, 0.0, gap_end_lon + 3.5, speed=300.0, altitude=15_000.0, track=90.0),
        message(240, 0.0, gap_end_lon + 5.0, speed=45.0, altitude=500.0, track=90.0),
        message(
            243,
            0.0,
            gap_end_lon + 5.01,
            speed=10.0,
            altitude=0.0,
            track=90.0,
            on_ground=True,
        ),
    ]


def test_continues_a_fast_directionally_consistent_airborne_gap():
    segments = identify_flight_segments(
        messages_with_gap(
            gap_end_lon=15.0,
            before_gap_track=90.0,
            after_gap_track=90.0,
        )
    )

    assert len(segments) == 1


def test_splits_gap_with_too_little_displacement_for_sustained_flight():
    segments = identify_flight_segments(
        messages_with_gap(
            gap_end_lon=6.0,
            before_gap_track=90.0,
            after_gap_track=90.0,
        )
    )

    assert len(segments) == 2
    assert segments[0].landing_time == START + timedelta(minutes=60)
    assert segments[1].takeoff_time >= START + timedelta(minutes=150)


def test_splits_gap_when_endpoint_track_reverses_the_gap_direction():
    segments = identify_flight_segments(
        messages_with_gap(
            gap_end_lon=15.0,
            before_gap_track=270.0,
            after_gap_track=90.0,
        )
    )

    assert len(segments) == 2
    assert segments[0].landing_time == START + timedelta(minutes=60)
