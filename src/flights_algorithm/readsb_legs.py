"""Python port of readsb's ``mark_legs`` trace segmentation algorithm."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Protocol, Sequence, TypeVar


class TraceMessage(Protocol):
    time: datetime


T = TypeVar("T", bound=TraceMessage)

_STATE_CHUNK_SIZE = 4
_MIN_TRACE_POINTS = 20
_MIN_ACCEPTED_POINT_GAP = timedelta(seconds=5)
_GAP_MARKER_INTERVAL = timedelta(minutes=5)
_LONG_GROUND_RECEPTION_GAP = timedelta(minutes=25)
_MAX_GROUND_DWELL_WITHOUT_AIRBORNE = timedelta(minutes=45)
_MAX_GAP_MARKER_ALTITUDE_FT = 20_000


def _optional_number(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _baro_altitude_ft(message: TraceMessage) -> float | None:
    return _optional_number(
        getattr(
            message,
            "baro_altitude_ft",
            getattr(message, "alt_baro", None),
        )
    )


def _geom_altitude_ft(message: TraceMessage) -> float | None:
    return _optional_number(
        getattr(
            message,
            "geom_altitude_ft",
            getattr(message, "alt_geom", None),
        )
    )


def _on_ground(message: TraceMessage) -> bool:
    return bool(getattr(message, "on_ground", False))


def _c_div(numerator: int, denominator: int) -> int:
    """Match C integer division, including for altitudes below sea level."""
    quotient = abs(numerator) // denominator
    return -quotient if numerator < 0 else quotient


def _raw_altitude_ft(message: TraceMessage) -> int | None:
    altitude = _baro_altitude_ft(message)
    if altitude is None:
        altitude = _geom_altitude_ft(message)
    return None if altitude is None else int(altitude)


class _AltitudeSmoother:
    """Carry the last airborne altitude across ground/invalid trace points."""

    def __init__(self, initial_altitude_ft: int) -> None:
        self._last_five = [initial_altitude_ft] * 5
        self._five_position = 0
        self._ground_altitude: int | None = None

    def altitude_ft(self, message: TraceMessage) -> int:
        altitude = _raw_altitude_ft(message)
        if _on_ground(message) or altitude is None:
            if self._ground_altitude is None:
                self._ground_altitude = _c_div(sum(self._last_five), 5)
            return self._ground_altitude

        self._ground_altitude = None
        self._last_five[self._five_position] = altitude
        self._five_position = (self._five_position + 1) % len(self._last_five)
        return altitude


def _altitude_threshold_ft(messages: Sequence[T], start: int) -> int:
    initial_altitude = int(_baro_altitude_ft(messages[start]) or 0)
    smoother = _AltitudeSmoother(initial_altitude)
    increment = _STATE_CHUNK_SIZE
    if len(messages) > 256 * _STATE_CHUNK_SIZE:
        increment = 4 * _STATE_CHUNK_SIZE

    sample_start = start - (start % _STATE_CHUNK_SIZE)
    altitudes = [
        smoother.altitude_ft(messages[index])
        for index in range(sample_start, len(messages), increment)
    ]
    threshold = int(sum(altitudes) / (len(altitudes) * 3))
    return min(2_500, max(200, threshold))


def mark_leg_indexes(messages: Sequence[T], start: int = 0) -> list[int]:
    """Return indexes carrying readsb ``leg_marker`` flags.

    This follows ``mark_legs`` in ``globe_index.c``. The input must be ordered
    by timestamp, as a readsb trace buffer is.
    """
    if len(messages) < _MIN_TRACE_POINTS or start >= len(messages):
        return []

    start = max(0, start)
    threshold = _altitude_threshold_ft(messages, start)
    threshold_third = _c_div(threshold, 3)
    threshold_nine_tenths = _c_div(threshold * 9, 10)

    initial_altitude = int(_baro_altitude_ft(messages[start]) or 0)
    smoother = _AltitudeSmoother(initial_altitude)

    high = 0
    low = 100_000
    major_climb_time: datetime | None = None
    major_descent_time: datetime | None = None
    major_climb_index = 0
    major_descent_index = 0
    last_high_time: datetime | None = None
    last_low_time: datetime | None = None
    last_low_index = 0

    last_airborne_time: datetime | None = None
    last_ground_time: datetime | None = None
    last_ground_index = 0
    first_ground_time: datetime | None = None
    first_ground_index = 0

    last_five_minute_gap_index = -1
    last_five_minute_gap_message: T | None = None
    was_ground = False
    marker_indexes: list[int] = []
    new_leg_index: int | None = None

    start = max(1, start)
    accepted_index = start - 1
    accepted_message = messages[accepted_index]

    for index in range(start, len(messages)):
        previous_index = accepted_index
        previous = accepted_message
        message = messages[index]
        elapsed = message.time - previous.time

        if elapsed < _MIN_ACCEPTED_POINT_GAP:
            continue

        accepted_index = index
        accepted_message = message

        if elapsed > _GAP_MARKER_INTERVAL:
            last_five_minute_gap_index = accepted_index
            last_five_minute_gap_message = message

        on_ground = _on_ground(message)
        altitude = smoother.altitude_ft(message)

        if on_ground or was_ground:
            if (
                last_ground_time is None
                or message.time > last_ground_time + _GAP_MARKER_INTERVAL
            ):
                first_ground_time = message.time
                first_ground_index = index
            last_ground_time = message.time
            last_ground_index = index
        else:
            last_airborne_time = message.time

        if was_ground:
            low = altitude
            high = altitude

        if altitude >= high:
            high = altitude

        if (
            not on_ground
            and major_descent_time is not None
            and last_ground_time is not None
            and last_ground_time >= major_descent_time
            and first_ground_time is not None
            and last_ground_time > first_ground_time + timedelta(minutes=1)
            and message.time > last_ground_time + timedelta(seconds=15)
            and high - low > 200
        ):
            high = low + threshold + 1
            last_high_time = message.time
            last_low_time = last_ground_time
            last_low_index = last_ground_index

        if altitude <= low:
            low = altitude

        if abs(low - altitude) < threshold_third:
            last_low_time = message.time
            last_low_index = index
        if abs(high - altitude) < threshold_third:
            last_high_time = message.time

        if high - low > threshold:
            if (
                last_high_time is not None
                and (
                    last_low_time is None
                    or last_high_time > last_low_time
                )
            ):
                if (
                    major_climb_time is None
                    or (
                        major_descent_time is not None
                        and major_climb_time <= major_descent_time
                    )
                ):
                    major_climb_index = min(len(messages) - 1, last_low_index + 3)
                    major_climb_time = messages[major_climb_index].time
                low = high - threshold_nine_tenths
            elif (
                last_low_time is not None
                and (
                    last_high_time is None
                    or last_low_time > last_high_time
                )
            ):
                descent_index = max(0, last_low_index - 3)
                while descent_index > 0:
                    candidate = messages[descent_index]
                    if (
                        _baro_altitude_ft(candidate) is not None
                        and not _on_ground(candidate)
                    ):
                        break
                    descent_index -= 1
                major_descent_index = descent_index
                major_descent_time = messages[descent_index].time
                high = low + threshold_nine_tenths

        leg_now = (
            major_descent_time is not None
            and (on_ground or was_ground)
            and elapsed > _LONG_GROUND_RECEPTION_GAP
        ) or (
            major_descent_time is not None
            and on_ground
            and (
                last_airborne_time is None
                or message.time
                > last_airborne_time + _MAX_GROUND_DWELL_WITHOUT_AIRBORNE
            )
        )

        leg_float = False
        if (
            major_climb_time is not None
            and major_descent_time is not None
            and major_climb_time > major_descent_time + timedelta(minutes=12)
            and last_five_minute_gap_index >= major_descent_index
            and last_five_minute_gap_message is not None
        ):
            gap_baro_altitude = _baro_altitude_ft(last_five_minute_gap_message)
            if (
                _on_ground(last_five_minute_gap_message)
                or gap_baro_altitude is None
                or gap_baro_altitude < _MAX_GAP_MARKER_ALTITUDE_FT
            ):
                leg_float = True

        if (
            major_climb_time is not None
            and major_descent_time is not None
            and major_climb_time > major_descent_time + timedelta(minutes=1)
            and last_ground_time is not None
            and last_ground_time >= major_descent_time
            and first_ground_time is not None
            and last_ground_time > first_ground_time + timedelta(minutes=1)
        ):
            leg_float = True

        if leg_float or leg_now:
            if leg_now:
                new_leg_index = index
                for candidate_index in range(previous_index + 1, index):
                    if (
                        messages[candidate_index].time
                        > messages[candidate_index - 1].time
                        + _GAP_MARKER_INTERVAL
                    ):
                        new_leg_index = candidate_index
                        break
            elif major_descent_index + 1 == major_climb_index:
                new_leg_index = major_climb_index
            else:
                for candidate_index in range(
                    major_climb_index,
                    major_descent_index,
                    -1,
                ):
                    if (
                        messages[candidate_index].time
                        > messages[candidate_index - 1].time
                        + _GAP_MARKER_INTERVAL
                    ):
                        new_leg_index = candidate_index
                        break

                if (
                    last_ground_time is not None
                    and last_ground_time > major_descent_time
                    and first_ground_time is not None
                ):
                    midpoint = first_ground_time + (
                        last_ground_time - first_ground_time
                    ) / 2
                    for candidate_index in range(
                        first_ground_index + 1,
                        last_ground_index + 1,
                    ):
                        if messages[candidate_index].time > midpoint:
                            new_leg_index = candidate_index
                            break
                else:
                    midpoint = major_descent_time + (
                        major_climb_time - major_descent_time
                    ) / 2
                    for candidate_index in range(
                        major_descent_index + 1,
                        major_climb_index,
                    ):
                        if messages[candidate_index].time > midpoint:
                            new_leg_index = candidate_index
                            break

            if new_leg_index is not None and new_leg_index not in marker_indexes:
                marker_indexes.append(new_leg_index)

            major_climb_time = None
            major_climb_index = 0
            major_descent_time = None
            major_descent_index = 0
            low += threshold
            high -= threshold

            if (
                new_leg_index is not None
                and _on_ground(messages[new_leg_index])
            ):
                high = 0
                low = 100_000

        was_ground = on_ground

    return marker_indexes
