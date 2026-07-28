from datetime import datetime, timedelta

from flights_algorithm.main import AdsbPositionMessage, identify_flight_segments
from flights_algorithm.readsb_legs import mark_leg_indexes


ICAO = "abc123"
START = datetime(2026, 3, 1)


def message(
    minutes: int,
    altitude: float | None,
    *,
    on_ground: bool | None = False,
    geom_altitude: int | None = None,
    lon: float | None = None,
    speed: float | None = None,
) -> AdsbPositionMessage:
    return AdsbPositionMessage(
        time=START + timedelta(minutes=minutes),
        icao=ICAO,
        lat=0.0,
        lon=minutes * 0.1 if lon is None else lon,
        ground_speed_kt=(0.0 if on_ground else 200.0) if speed is None else speed,
        on_ground=on_ground,
        baro_altitude_ft=altitude,
        geom_altitude_ft=geom_altitude,
    )


def two_flight_trace() -> list[AdsbPositionMessage]:
    altitudes = [
        0,
        0,
        1_000,
        5_000,
        10_000,
        20_000,
        30_000,
        30_000,
        30_000,
        25_000,
        18_000,
        10_000,
        5_000,
        1_000,
        0,
        0,
        0,
        0,
        1_000,
        5_000,
        10_000,
        20_000,
        30_000,
        30_000,
    ]
    return [
        message(index, altitude, on_ground=altitude == 0)
        for index, altitude in enumerate(altitudes)
    ]


def test_marks_new_leg_during_ground_interval_between_descent_and_climb():
    messages = two_flight_trace()

    assert mark_leg_indexes(messages) == [16]


def test_converts_readsb_leg_markers_to_existing_flight_segments():
    segments = identify_flight_segments(two_flight_trace())

    assert len(segments) == 2
    assert segments[0].takeoff_time == START + timedelta(minutes=2)
    assert segments[0].landing_time == START + timedelta(minutes=14)
    assert segments[1].takeoff_time == START + timedelta(minutes=18)
    assert segments[1].landing_time == START + timedelta(minutes=23)


def test_uses_ground_air_transitions_instead_of_day_and_leg_boundaries():
    messages = two_flight_trace()
    messages[0] = message(0, 0, on_ground=True)
    messages[1] = message(1, 0, on_ground=True)

    segments = identify_flight_segments(messages)

    assert segments[0].takeoff_message.on_ground is False
    assert segments[0].landing_message.on_ground is True
    assert segments[1].takeoff_message.on_ground is False


def test_falls_back_to_trace_boundaries_when_ground_state_is_missing():
    messages = [
        message(index, altitude, on_ground=None)
        for index, altitude in enumerate(
            [1_000, 5_000, 10_000, 20_000, 30_000, 30_000]
        )
    ]

    segments = identify_flight_segments(messages)

    assert len(segments) == 1
    assert segments[0].takeoff_time == START
    assert segments[0].landing_time == START + timedelta(minutes=5)


def test_ignores_isolated_airborne_status_blips_while_parked():
    messages = [
        message(0, None, on_ground=True, lon=0.0, speed=0.1),
        message(1, None, on_ground=False, lon=0.0, speed=0.1),
        message(2, None, on_ground=True, lon=0.0, speed=0.1),
        message(3, None, on_ground=True, lon=0.0, speed=0.1),
        message(4, 0, on_ground=False, lon=0.0, speed=87),
        message(5, 1_000, on_ground=False, lon=0.1, speed=150),
        message(6, 5_000, on_ground=False, lon=0.2, speed=200),
        message(10, None, on_ground=True, lon=0.3, speed=80),
        message(11, None, on_ground=False, lon=0.3, speed=0.1),
        message(12, None, on_ground=True, lon=0.3, speed=0.1),
    ]

    segments = identify_flight_segments(messages)

    assert len(segments) == 1
    assert segments[0].takeoff_time == START + timedelta(minutes=4)
    assert segments[0].landing_time == START + timedelta(minutes=10)


def test_rejects_all_ground_slice_despite_duration_and_accumulated_movement():
    messages = [
        message(
            index * 2,
            None,
            on_ground=True,
            lon=index * 0.012,
            speed=15,
        )
        for index in range(6)
    ]

    assert identify_flight_segments(messages) == []


def test_readsb_does_not_mark_traces_shorter_than_twenty_points():
    messages = two_flight_trace()[:19]

    assert mark_leg_indexes(messages) == []
    assert len(identify_flight_segments(messages)) == 1


def test_uses_geometric_altitude_when_baro_altitude_is_missing():
    messages = [
        message(
            index,
            None,
            on_ground=source.baro_altitude_ft == 0,
            geom_altitude=int(source.baro_altitude_ft or 0),
        )
        for index, source in enumerate(two_flight_trace())
    ]

    assert mark_leg_indexes(messages) == [16]
