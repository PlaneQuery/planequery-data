import argparse
import math
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Sequence
from concurrent.futures import ProcessPoolExecutor

import polars as pl

from adsb.adsb_messages_types import AdsbMessageRow
from data_engineering.adsb.read_adsb import get_icaos_in_adsb, read_adsb_lz
from data_engineering.utils import OUTPUT_DIR


AIRPORT_CANDIDATE_RADIUS_KM = 40.0
AIRPORT_SIZE_BONUS_KM = 5.0
MAX_AIRPORT_IDENT_DISTANCE_KM = 8.0
DB_FLAG_MILITARY = 1
DB_FLAG_INTERESTING = 2
DB_FLAG_PIA = 4
DB_FLAG_LADD = 8
AIRPORT_SIZE_RANK = {
    "small_airport": 0,
    "medium_airport": 1,
    "large_airport": 2,
}
TAKEOFF_SPEED_KT = 35.0
LANDING_SPEED_KT = 55.0
TAKEOFF_ACCELERATION_KT = 20.0
APPROACH_SPEED_KT = 130.0
APPROACH_DECELERATION_KT = -20.0
MAX_LANDING_ALTITUDE_FT = 20_000
HIGH_ALTITUDE_APPROACH_SPEED_DROP_KT = 80.0
AIRBORNE_GAP_MIN_SPEED_KT = 90.0
AIRBORNE_GAP_MIN_ALTITUDE_FT = 1_000
MIN_AIRBORNE_GAP_AVERAGE_SPEED_KT = 30.0
MAX_AIRBORNE_GAP_AVERAGE_SPEED_KT = 700.0
HIGH_ALTITUDE_GAP_MIN_ALTITUDE_FT = 30_000
MIN_STOP_LIKE_GAP = timedelta(minutes=7)
MAX_STOP_LIKE_GAP_DISTANCE_KM = 20.0
MAX_STOP_LIKE_GAP_AVERAGE_SPEED_KT = 25.0
MAX_STOP_LIKE_GAP_ALTITUDE_FT = 7_750
MAX_SHORT_SAME_AIRPORT_DURATION = timedelta(minutes=25)
TAKEOFF_BACKDATE_MIN_ALTITUDE_FT = 3_000
TAKEOFF_BACKDATE_CLIMB_RATE_FT_PER_MIN = 800.0
TAKEOFF_BACKDATE_MAX_SPEED_KT = 520.0
MAX_MESSAGE_GAP = timedelta(minutes=30)
MAX_AIRBORNE_GAP = timedelta(hours=8)
MAX_TAKEOFF_BACKDATE = timedelta(minutes=30)
LANDING_CONFIRMATION_TIME = timedelta(minutes=2)
MIN_FLIGHT_DURATION = timedelta(minutes=5)
MIN_FLIGHT_DISTANCE_KM = 5.0
MAX_SEGMENT_STITCH_GAP = timedelta(minutes=45)
RUN_MAIN_CONTEXT = timedelta(hours=12)
ADSB_ALGORITHM_COLUMNS = [
    "time",
    "icao",
    "lat",
    "lon",
    "ground_speed_kt",
    "track_deg",
    "on_ground",
    "baro_altitude_ft",
    "geom_altitude_ft",
    "pia",
    "ladd",
    "military",
    "interesting",
    "aircraft_type",
    "owner",
    "aircraft_description",
    "category",
]
AIRCRAFT_METADATA_COLUMNS = (
    "aircraft_type",
    "owner",
    "aircraft_description",
    "category",
)
AIRCRAFT_METADATA_ALIASES = {
    "aircraft_type": ("aircraft_type", "t"),
    "owner": ("owner", "ownOp"),
    "aircraft_description": ("aircraft_description", "desc"),
    "category": ("category",),
}


class FlightState(Enum):
    GROUND = "ground"
    IN_FLIGHT = "in_flight"


@dataclass(frozen=True)
class AdsbPositionMessage:
    time: datetime
    icao: str
    lat: float
    lon: float
    ground_speed_kt: float | None = None
    on_ground: bool | None = None
    baro_altitude_ft: float | None = None
    geom_altitude_ft: int | None = None
    dbFlags: int | None = None
    pia: bool | None = None
    ladd: bool | None = None
    military: bool | None = None
    interesting: bool | None = None
    aircraft_type: str | None = None
    owner: str | None = None
    aircraft_description: str | None = None
    category: str | None = None


@dataclass(frozen=True)
class MessageMotion:
    prev: AdsbMessageRow | AdsbPositionMessage
    curr: AdsbMessageRow | AdsbPositionMessage
    time_gap: timedelta
    calculated_ground_speed_kt: float | None
    reported_ground_speed_kt: float | None
    effective_ground_speed_kt: float | None
    previous_ground_speed_kt: float | None
    speed_delta_kt: float | None
    distance_km: float


@dataclass(frozen=True)
class FlightSegment:
    icao: str
    takeoff_time: datetime
    landing_time: datetime
    takeoff_message: AdsbMessageRow | AdsbPositionMessage
    landing_message: AdsbMessageRow | AdsbPositionMessage
    distance_travelled_km: float = 0.0
    db_flags_counts: tuple[tuple[int, int], ...] = ()
    aircraft_metadata_counts: tuple[
        tuple[str, tuple[tuple[str, int], ...]],
        ...,
    ] = ()


@dataclass(frozen=True)
class AirportIdentGuess:
    ident: str | None
    distance_km: float | None
    lat: float | None
    lon: float | None
    elevation_ft: int | None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def calculated_ground_speed_kt(
    prev: AdsbMessageRow | AdsbPositionMessage,
    curr: AdsbMessageRow | AdsbPositionMessage,
) -> float | None:
    dt_s = (curr.time - prev.time).total_seconds()
    if dt_s <= 0:
        return None

    distance_km = haversine_km(prev.lat, prev.lon, curr.lat, curr.lon)
    return distance_km * 1943.844492 / dt_s


def _optional_float(value: float | None) -> float | None:
    if value is None:
        return None
    if math.isnan(value):
        return None
    return float(value)


def _optional_int(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value) -> bool | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return bool(value)


def _db_flags_from_message(message: AdsbMessageRow | AdsbPositionMessage) -> int | None:
    db_flags = _optional_int(getattr(message, "dbFlags", getattr(message, "db_flags", None)))
    if db_flags is not None:
        return db_flags

    split_flags = [
        ("military", DB_FLAG_MILITARY),
        ("interesting", DB_FLAG_INTERESTING),
        ("pia", DB_FLAG_PIA),
        ("ladd", DB_FLAG_LADD),
    ]
    db_flags = 0
    found_split_flag = False
    for attr, bit in split_flags:
        value = _optional_bool(getattr(message, attr, None))
        if value is None:
            continue
        found_split_flag = True
        if value:
            db_flags |= bit

    return db_flags if found_split_flag else None


def _db_flags_counts_for_messages(
    messages: Sequence[AdsbMessageRow | AdsbPositionMessage],
) -> tuple[tuple[int, int], ...]:
    db_flags = [
        value
        for message in messages
        if (value := _db_flags_from_message(message)) is not None
    ]
    return tuple(Counter(db_flags).items())


def _merge_db_flags_counts(
    *counts_by_segment: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    counter = Counter()
    for counts in counts_by_segment:
        for db_flags, count in counts:
            counter[db_flags] += count
    return tuple(counter.items())


def _db_flags_mode_from_counts(db_flags_counts: tuple[tuple[int, int], ...]) -> int | None:
    if not db_flags_counts:
        return None
    return max(db_flags_counts, key=lambda item: item[1])[0]


def _db_flags_to_flight_flags(db_flags: int | None) -> dict[str, bool]:
    if db_flags is None:
        db_flags = 0
    return {
        "pia": bool(db_flags & DB_FLAG_PIA),
        "ladd": bool(db_flags & DB_FLAG_LADD),
        "military": bool(db_flags & DB_FLAG_MILITARY),
        "interesting": bool(db_flags & DB_FLAG_INTERESTING),
    }


def _non_empty_string(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    value = str(value).strip()
    return value if value else None


def _message_aircraft_metadata(
    message: AdsbMessageRow | AdsbPositionMessage,
    column: str,
) -> str | None:
    for attr in AIRCRAFT_METADATA_ALIASES[column]:
        value = _non_empty_string(getattr(message, attr, None))
        if value is not None:
            return value
    return None


def _aircraft_metadata_counts_for_messages(
    messages: Sequence[AdsbMessageRow | AdsbPositionMessage],
) -> tuple[tuple[str, tuple[tuple[str, int], ...]], ...]:
    return tuple(
        (
            column,
            tuple(
                Counter(
                    value
                    for message in messages
                    if (value := _message_aircraft_metadata(message, column)) is not None
                ).items()
            ),
        )
        for column in AIRCRAFT_METADATA_COLUMNS
    )


def _merge_aircraft_metadata_counts(
    *counts_by_segment: tuple[tuple[str, tuple[tuple[str, int], ...]], ...],
) -> tuple[tuple[str, tuple[tuple[str, int], ...]], ...]:
    counters = {column: Counter() for column in AIRCRAFT_METADATA_COLUMNS}
    for counts in counts_by_segment:
        for column, value_counts in counts:
            counters[column].update(dict(value_counts))
    return tuple(
        (column, tuple(counters[column].items()))
        for column in AIRCRAFT_METADATA_COLUMNS
    )


def _aircraft_metadata_mode_from_counts(
    aircraft_metadata_counts: tuple[tuple[str, tuple[tuple[str, int], ...]], ...],
) -> dict[str, str]:
    modes = {column: "" for column in AIRCRAFT_METADATA_COLUMNS}
    for column, value_counts in aircraft_metadata_counts:
        if value_counts:
            modes[column] = max(value_counts, key=lambda item: item[1])[0]
    return modes


def _reported_ground_speed_kt(message: AdsbMessageRow | AdsbPositionMessage) -> float | None:
    return _optional_float(
        getattr(message, "ground_speed_kt", getattr(message, "gs", None))
    )


def _on_ground(message: AdsbMessageRow | AdsbPositionMessage) -> bool | None:
    value = getattr(message, "on_ground", None)
    if value is None:
        return None
    return bool(value)


def _baro_altitude_ft(message: AdsbMessageRow | AdsbPositionMessage) -> float | None:
    return _optional_float(
        getattr(message, "baro_altitude_ft", getattr(message, "alt_baro", None))
    )


def _baro_altitude_int_ft(message: AdsbMessageRow | AdsbPositionMessage) -> int | None:
    return _optional_int(
        getattr(message, "baro_altitude_ft", getattr(message, "alt_baro", None))
    )


def _geom_altitude_ft(message: AdsbMessageRow | AdsbPositionMessage) -> int | None:
    return _optional_int(
        getattr(message, "geom_altitude_ft", getattr(message, "alt_geom", None))
    )


def _motion_edges(messages: Sequence[AdsbMessageRow | AdsbPositionMessage]) -> list[MessageMotion]:
    sorted_messages = sorted(messages, key=lambda row: row.time)
    edges = []
    for prev, curr in zip(sorted_messages, sorted_messages[1:]):
        distance_km = haversine_km(prev.lat, prev.lon, curr.lat, curr.lon)
        calculated_speed_kt = calculated_ground_speed_kt(prev, curr)
        reported_speed_kt = _reported_ground_speed_kt(curr)
        effective_speed_kt = reported_speed_kt if reported_speed_kt is not None else calculated_speed_kt
        previous_speed_kt = _reported_ground_speed_kt(prev)
        speed_delta_kt = (
            effective_speed_kt - previous_speed_kt
            if effective_speed_kt is not None and previous_speed_kt is not None
            else None
        )
        edges.append(
            MessageMotion(
                prev=prev,
                curr=curr,
                time_gap=curr.time - prev.time,
                calculated_ground_speed_kt=calculated_speed_kt,
                reported_ground_speed_kt=reported_speed_kt,
                effective_ground_speed_kt=effective_speed_kt,
                previous_ground_speed_kt=previous_speed_kt,
                speed_delta_kt=speed_delta_kt,
                distance_km=distance_km,
            )
        )
    return edges


def _is_takeoff_motion(motion: MessageMotion) -> bool:
    curr_on_ground = _on_ground(motion.curr)
    if curr_on_ground is True:
        return False
    if curr_on_ground is False and _on_ground(motion.prev) is True:
        if motion.effective_ground_speed_kt is None:
            return True
        return (
            motion.effective_ground_speed_kt >= TAKEOFF_SPEED_KT
            or (
                motion.speed_delta_kt is not None
                and motion.speed_delta_kt >= TAKEOFF_ACCELERATION_KT
            )
        )
    speed = motion.effective_ground_speed_kt
    return speed is not None and speed >= TAKEOFF_SPEED_KT


def _is_landing_motion(motion: MessageMotion) -> bool:
    if _on_ground(motion.curr) is True:
        return True
    speed = motion.effective_ground_speed_kt
    if speed is None:
        return False
    if speed <= LANDING_SPEED_KT:
        return True
    return (
        speed <= APPROACH_SPEED_KT
        and motion.speed_delta_kt is not None
        and motion.speed_delta_kt <= APPROACH_DECELERATION_KT
    )


def _landing_endpoint_makes_sense(
    landing_message: AdsbMessageRow | AdsbPositionMessage,
    max_speed_kt: float | None,
) -> bool:
    if _on_ground(landing_message) is True:
        return True

    landing_altitude_ft = _baro_altitude_ft(landing_message)
    if landing_altitude_ft is None:
        return True
    if landing_altitude_ft <= MAX_LANDING_ALTITUDE_FT:
        return True

    landing_speed_kt = _reported_ground_speed_kt(landing_message)
    return (
        max_speed_kt is not None
        and landing_speed_kt is not None
        and landing_speed_kt <= APPROACH_SPEED_KT
        and max_speed_kt - landing_speed_kt >= HIGH_ALTITUDE_APPROACH_SPEED_DROP_KT
    )


def _gap_average_speed_kt(motion: MessageMotion) -> float | None:
    dt_s = motion.time_gap.total_seconds()
    if dt_s <= 0:
        return None
    return motion.distance_km * 1943.844492 / dt_s


def _message_is_low_for_stop_gap(message: AdsbMessageRow | AdsbPositionMessage) -> bool:
    if _on_ground(message) is True:
        return True

    altitude_ft = _baro_altitude_ft(message)
    if altitude_ft is None:
        return True
    return altitude_ft <= MAX_STOP_LIKE_GAP_ALTITUDE_FT


def _is_stop_like_gap(motion: MessageMotion) -> bool:
    if motion.time_gap < MIN_STOP_LIKE_GAP:
        return False

    average_speed_kt = _gap_average_speed_kt(motion)
    if average_speed_kt is None:
        return False
    if average_speed_kt > MAX_STOP_LIKE_GAP_AVERAGE_SPEED_KT:
        return False
    if motion.distance_km > MAX_STOP_LIKE_GAP_DISTANCE_KM:
        return False
    return _message_is_low_for_stop_gap(motion.prev) and _message_is_low_for_stop_gap(motion.curr)


def _message_looks_airborne_for_gap(message: AdsbMessageRow | AdsbPositionMessage) -> bool:
    if _on_ground(message) is True:
        return False

    speed_kt = _reported_ground_speed_kt(message)
    altitude_ft = _baro_altitude_ft(message)
    return (
        speed_kt is not None
        and speed_kt >= AIRBORNE_GAP_MIN_SPEED_KT
    ) or (
        altitude_ft is not None
        and altitude_ft >= AIRBORNE_GAP_MIN_ALTITUDE_FT
    )


def _should_continue_across_gap(motion: MessageMotion) -> bool:
    if motion.time_gap > MAX_AIRBORNE_GAP:
        return False
    if _is_stop_like_gap(motion):
        return False
    if not _message_looks_airborne_for_gap(motion.prev):
        return False
    if not _message_looks_airborne_for_gap(motion.curr):
        return False

    average_speed_kt = _gap_average_speed_kt(motion)
    if average_speed_kt is None or average_speed_kt > MAX_AIRBORNE_GAP_AVERAGE_SPEED_KT:
        return False
    if average_speed_kt >= MIN_AIRBORNE_GAP_AVERAGE_SPEED_KT:
        return True

    prev_altitude_ft = _baro_altitude_ft(motion.prev)
    curr_altitude_ft = _baro_altitude_ft(motion.curr)
    return (
        prev_altitude_ft is not None
        and curr_altitude_ft is not None
        and prev_altitude_ft >= HIGH_ALTITUDE_GAP_MIN_ALTITUDE_FT
        and curr_altitude_ft >= HIGH_ALTITUDE_GAP_MIN_ALTITUDE_FT
    )


def _estimated_takeoff_time(message: AdsbMessageRow | AdsbPositionMessage) -> datetime:
    if _on_ground(message) is True:
        return message.time

    altitude_ft = _baro_altitude_ft(message)
    if altitude_ft is None or altitude_ft < TAKEOFF_BACKDATE_MIN_ALTITUDE_FT:
        return message.time

    speed_kt = _reported_ground_speed_kt(message)
    if speed_kt is not None and speed_kt > TAKEOFF_BACKDATE_MAX_SPEED_KT:
        return message.time

    backdate = min(
        MAX_TAKEOFF_BACKDATE,
        timedelta(minutes=altitude_ft / TAKEOFF_BACKDATE_CLIMB_RATE_FT_PER_MIN),
    )
    return message.time - backdate


def _maybe_add_segment(
    segments: list[FlightSegment],
    takeoff_message: AdsbMessageRow | AdsbPositionMessage | None,
    landing_message: AdsbMessageRow | AdsbPositionMessage,
    distance_travelled_km: float,
    max_speed_kt: float | None,
    segment_messages: Sequence[AdsbMessageRow | AdsbPositionMessage],
    *,
    takeoff_time_message: AdsbMessageRow | AdsbPositionMessage | None = None,
    landing_time_message: AdsbMessageRow | AdsbPositionMessage | None = None,
) -> None:
    if takeoff_message is None:
        return
    if takeoff_time_message is None:
        takeoff_time_message = takeoff_message
    if landing_time_message is None:
        landing_time_message = landing_message

    takeoff_time = _estimated_takeoff_time(takeoff_time_message)
    landing_time = landing_time_message.time
    if landing_time - takeoff_time < MIN_FLIGHT_DURATION:
        return
    if distance_travelled_km < MIN_FLIGHT_DISTANCE_KM:
        return
    if not _landing_endpoint_makes_sense(landing_message, max_speed_kt):
        return
    segments.append(
        FlightSegment(
            icao=takeoff_message.icao,
            takeoff_time=takeoff_time,
            landing_time=landing_time,
            takeoff_message=takeoff_message,
            landing_message=landing_message,
            distance_travelled_km=distance_travelled_km,
            db_flags_counts=_db_flags_counts_for_messages(segment_messages),
            aircraft_metadata_counts=_aircraft_metadata_counts_for_messages(
                segment_messages
            ),
        )
    )


def identify_flight_segments(messages: Sequence[AdsbMessageRow | AdsbPositionMessage]) -> list[FlightSegment]:
    state = FlightState.GROUND
    takeoff_message = None
    takeoff_time_message = None
    landing_candidate = None
    distance_travelled_km = 0.0
    max_speed_kt = None
    last_message = None
    segment_messages = []
    ground_messages = []
    ground_distance_travelled_km = 0.0
    segments = []

    for motion in _motion_edges(messages):
        last_message = motion.curr
        speed = motion.effective_ground_speed_kt
        has_ground_state = _on_ground(motion.prev) is not None or _on_ground(motion.curr) is not None
        if speed is None and not has_ground_state:
            continue

        gap_was_continued = False
        if state == FlightState.IN_FLIGHT and _is_stop_like_gap(motion):
            _maybe_add_segment(
                segments,
                takeoff_message,
                motion.prev,
                distance_travelled_km,
                max_speed_kt,
                segment_messages,
                takeoff_time_message=takeoff_time_message,
                landing_time_message=motion.prev,
            )
            state = FlightState.GROUND
            takeoff_message = None
            takeoff_time_message = None
            landing_candidate = None
            distance_travelled_km = 0.0
            max_speed_kt = None
            segment_messages = []
            ground_messages = []
            ground_distance_travelled_km = 0.0
            if motion.time_gap > MAX_MESSAGE_GAP:
                ground_messages = [motion.curr]
            elif _is_takeoff_motion(motion):
                state = FlightState.IN_FLIGHT
                takeoff_message = motion.curr
                takeoff_time_message = motion.curr
                landing_candidate = None
                distance_travelled_km = 0.0
                max_speed_kt = speed
                segment_messages = [motion.curr]
            else:
                ground_messages = [motion.curr]
            continue

        if motion.time_gap > MAX_MESSAGE_GAP:
            if state == FlightState.IN_FLIGHT and _should_continue_across_gap(motion):
                landing_candidate = None
                distance_travelled_km += motion.distance_km
                gap_was_continued = True
            elif state == FlightState.IN_FLIGHT:
                _maybe_add_segment(
                    segments,
                    takeoff_message,
                    motion.prev,
                    distance_travelled_km,
                    max_speed_kt,
                    segment_messages,
                    takeoff_time_message=takeoff_time_message,
                    landing_time_message=motion.prev,
                )
                state = FlightState.GROUND
                takeoff_message = None
                takeoff_time_message = None
                landing_candidate = None
                distance_travelled_km = 0.0
                max_speed_kt = None
                segment_messages = []
                ground_messages = []
                ground_distance_travelled_km = 0.0
                continue
            else:
                state = FlightState.GROUND
                takeoff_message = None
                takeoff_time_message = None
                landing_candidate = None
                distance_travelled_km = 0.0
                max_speed_kt = None
                segment_messages = []
                ground_messages = [motion.curr]
                ground_distance_travelled_km = 0.0
                continue

        if state == FlightState.GROUND:
            if not ground_messages:
                ground_messages = [motion.prev]
            elif ground_messages[-1].time != motion.prev.time:
                ground_messages.append(motion.prev)

            if _is_takeoff_motion(motion):
                if ground_messages[-1].time != motion.curr.time:
                    ground_messages.append(motion.curr)
                state = FlightState.IN_FLIGHT
                takeoff_message = ground_messages[0]
                takeoff_time_message = motion.curr
                landing_candidate = None
                distance_travelled_km = ground_distance_travelled_km + motion.distance_km
                max_speed_kt = speed
                segment_messages = ground_messages
                ground_messages = []
                ground_distance_travelled_km = 0.0
            else:
                if ground_messages[-1].time != motion.curr.time:
                    ground_messages.append(motion.curr)
                ground_distance_travelled_km += motion.distance_km
            continue

        if speed is not None:
            max_speed_kt = speed if max_speed_kt is None else max(max_speed_kt, speed)
        if not gap_was_continued:
            distance_travelled_km += motion.distance_km
        segment_messages.append(motion.curr)
        if _is_landing_motion(motion):
            if landing_candidate is None:
                landing_candidate = motion.curr
            if motion.curr.time - landing_candidate.time >= LANDING_CONFIRMATION_TIME:
                _maybe_add_segment(
                    segments,
                    takeoff_message,
                    motion.curr,
                    distance_travelled_km,
                    max_speed_kt,
                    segment_messages,
                    takeoff_time_message=takeoff_time_message,
                    landing_time_message=landing_candidate,
                )
                state = FlightState.GROUND
                takeoff_message = None
                takeoff_time_message = None
                landing_candidate = None
                distance_travelled_km = 0.0
                max_speed_kt = None
                segment_messages = []
                ground_messages = [motion.curr]
                ground_distance_travelled_km = 0.0
        else:
            landing_candidate = None

    if state == FlightState.IN_FLIGHT and landing_candidate is not None and last_message is not None:
        _maybe_add_segment(
            segments,
            takeoff_message,
            last_message,
            distance_travelled_km,
            max_speed_kt,
            segment_messages,
            takeoff_time_message=takeoff_time_message,
            landing_time_message=landing_candidate,
        )
    elif state == FlightState.IN_FLIGHT and last_message is not None:
        _maybe_add_segment(
            segments,
            takeoff_message,
            last_message,
            distance_travelled_km,
            max_speed_kt,
            segment_messages,
            takeoff_time_message=takeoff_time_message,
            landing_time_message=last_message,
        )

    return segments


def stitch_flight_segments(segments: Sequence[FlightSegment]) -> list[FlightSegment]:
    stitched = []
    current = None

    for segment in sorted(segments, key=lambda row: (row.icao, row.takeoff_time)):
        if current is None:
            current = segment
            continue

        gap = segment.takeoff_time - current.landing_time
        if segment.icao == current.icao and gap <= MAX_SEGMENT_STITCH_GAP:
            current = FlightSegment(
                icao=current.icao,
                takeoff_time=current.takeoff_time,
                landing_time=max(current.landing_time, segment.landing_time),
                takeoff_message=current.takeoff_message,
                landing_message=segment.landing_message,
                distance_travelled_km=current.distance_travelled_km + segment.distance_travelled_km,
                db_flags_counts=_merge_db_flags_counts(
                    current.db_flags_counts,
                    segment.db_flags_counts,
                ),
                aircraft_metadata_counts=_merge_aircraft_metadata_counts(
                    current.aircraft_metadata_counts,
                    segment.aircraft_metadata_counts,
                ),
            )
        else:
            stitched.append(current)
            current = segment

    if current is not None:
        stitched.append(current)

    return stitched


def main(adsb_messages: Sequence[AdsbMessageRow | AdsbPositionMessage]) -> list[FlightSegment]:
    return identify_flight_segments(adsb_messages)


def _rows_to_position_messages(df: pl.DataFrame) -> list[AdsbPositionMessage]:
    return [
        AdsbPositionMessage(
            time=row["time"],
            icao=row["icao"],
            lat=row["lat"],
            lon=row["lon"],
            ground_speed_kt=row.get("ground_speed_kt"),
            on_ground=row.get("on_ground"),
            baro_altitude_ft=row.get("baro_altitude_ft"),
            geom_altitude_ft=row.get("geom_altitude_ft"),
            dbFlags=row.get("dbFlags"),
            pia=row.get("pia"),
            ladd=row.get("ladd"),
            military=row.get("military"),
            interesting=row.get("interesting"),
            aircraft_type=row.get("aircraft_type"),
            owner=row.get("owner"),
            aircraft_description=row.get("aircraft_description"),
            category=row.get("category"),
        )
        for row in df.iter_rows(named=True)
    ]


def _get_airport_lookup():
    from airports.airportLookup import AirportLookup
    try:
        airport_lookup = AirportLookup()
    except (FileNotFoundError, KeyError):
        AirportLookup._instance = None
        return None
    if not getattr(airport_lookup, "_ball_tree", None):
        AirportLookup._instance = None
        return None
    return airport_lookup



def _airport_rank(airport) -> int:
    return AIRPORT_SIZE_RANK.get(str(airport.type), 0)


def _airport_distance_to_message(
    airport,
    message: AdsbMessageRow | AdsbPositionMessage,
) -> float:
    return haversine_km(message.lat, message.lon, airport.lat, airport.lon)


def _airport_guess_for_message(
    airport_lookup,
    message: AdsbMessageRow | AdsbPositionMessage,
) -> AirportIdentGuess:
    if airport_lookup is None:
        return AirportIdentGuess(
            ident=None,
            distance_km=None,
            lat=None,
            lon=None,
            elevation_ft=None,
        )

    airports = airport_lookup.getAirportsWithinRadius(
        message.lat,
        message.lon,
        AIRPORT_CANDIDATE_RADIUS_KM,
    )
    if not airports:
        airports = [airport_lookup.getClosestAirport(message.lat, message.lon)]

    best_airport = min(
        airports,
        key=lambda airport: (
            _airport_distance_to_message(airport, message)
            - AIRPORT_SIZE_BONUS_KM * _airport_rank(airport)
        ),
    )
    return AirportIdentGuess(
        ident=best_airport.ident,
        distance_km=_airport_distance_to_message(best_airport, message),
        lat=best_airport.lat,
        lon=best_airport.lon,
        elevation_ft=best_airport.elevation_ft,
    )


def _airport_guess_for_ident(
    airport_lookup,
    ident: str | None,
    message: AdsbMessageRow | AdsbPositionMessage,
) -> AirportIdentGuess:
    if airport_lookup is None or ident is None:
        return AirportIdentGuess(
            ident=None,
            distance_km=None,
            lat=None,
            lon=None,
            elevation_ft=None,
        )

    airport = airport_lookup.get_Airport_from_airport_ident(ident)
    if airport is None:
        return AirportIdentGuess(
            ident=None,
            distance_km=None,
            lat=None,
            lon=None,
            elevation_ft=None,
        )
    return AirportIdentGuess(
        ident=ident,
        distance_km=_airport_distance_to_message(airport, message),
        lat=airport.lat,
        lon=airport.lon,
        elevation_ft=airport.elevation_ft,
    )


def _airport_ident_is_confident(guess: AirportIdentGuess) -> bool:
    if guess.ident is None or guess.distance_km is None:
        return False
    return guess.distance_km <= MAX_AIRPORT_IDENT_DISTANCE_KM


def _airport_ident_guesses_for_segments(
    segments: Sequence[FlightSegment],
) -> tuple[list[AirportIdentGuess], list[AirportIdentGuess]]:
    airport_lookup = _get_airport_lookup()
    takeoff_guesses = [
        _airport_guess_for_message(airport_lookup, segment.takeoff_message)
        for segment in segments
    ]
    landing_guesses = [
        _airport_guess_for_message(airport_lookup, segment.landing_message)
        for segment in segments
    ]

    sorted_indices = sorted(
        range(len(segments)),
        key=lambda index: (segments[index].icao, segments[index].takeoff_time),
    )
    for previous_index, current_index in zip(sorted_indices, sorted_indices[1:]):
        previous_segment = segments[previous_index]
        current_segment = segments[current_index]
        if previous_segment.icao != current_segment.icao:
            continue

        shared_guess = _airport_guess_for_ident(
            airport_lookup,
            takeoff_guesses[current_index].ident,
            previous_segment.landing_message,
        )
        if _airport_ident_is_confident(shared_guess):
            landing_guesses[previous_index] = shared_guess

    return takeoff_guesses, landing_guesses


def _segments_to_flights_df(segments: list[FlightSegment]) -> pl.DataFrame:
    from data_engineering.flights.flight_type import FLIGHT_POLARS_SCHEMA

    if not segments:
        return pl.DataFrame(schema=FLIGHT_POLARS_SCHEMA)

    takeoff_guesses, landing_guesses = _airport_ident_guesses_for_segments(segments)
    rows = []
    for segment, takeoff_guess, landing_guess in zip(segments, takeoff_guesses, landing_guesses):
        flight_flags = _db_flags_to_flight_flags(
            _db_flags_mode_from_counts(segment.db_flags_counts)
        )
        aircraft_metadata = _aircraft_metadata_mode_from_counts(
            segment.aircraft_metadata_counts
        )
        rows.append(
            {
                "icao": segment.icao,
                "callsign": "",
                "registration": "",
                "takeoff_time": segment.takeoff_time,
                "takeoff_airport_ident": takeoff_guess.ident,
                "landing_time": segment.landing_time,
                "landing_airport_ident": landing_guess.ident,
                "first_message_time": segment.takeoff_message.time,
                "first_lat": segment.takeoff_message.lat,
                "first_lon": segment.takeoff_message.lon,
                "first_baro_altitude_ft": _baro_altitude_int_ft(segment.takeoff_message),
                "first_geom_altitude_ft": _geom_altitude_ft(segment.takeoff_message),
                "last_message_time": segment.landing_message.time,
                "last_lat": segment.landing_message.lat,
                "last_lon": segment.landing_message.lon,
                "last_baro_altitude_ft": _baro_altitude_int_ft(segment.landing_message),
                "last_geom_altitude_ft": _geom_altitude_ft(segment.landing_message),
                **flight_flags,
                **aircraft_metadata,
            }
        )
    return (
        pl.DataFrame(rows)
        .select(list(FLIGHT_POLARS_SCHEMA.keys()))
        .cast(FLIGHT_POLARS_SCHEMA)
    )


def _filter_short_same_airport_flights(flights_df: pl.DataFrame) -> pl.DataFrame:
    same_named_airport = (
        pl.col("takeoff_airport_ident").is_not_null()
        & pl.col("landing_airport_ident").is_not_null()
        & (pl.col("takeoff_airport_ident") != "")
        & (pl.col("takeoff_airport_ident") == pl.col("landing_airport_ident"))
    )
    short_flight = (
        pl.col("landing_time") - pl.col("takeoff_time")
    ) <= MAX_SHORT_SAME_AIRPORT_DURATION
    return flights_df.filter(~(same_named_airport & short_flight))


def _day_bounds(target_date: date) -> tuple[datetime, datetime]:
    start = datetime.combine(target_date, datetime.min.time())
    return start, start + timedelta(days=1)


def _segment_takes_off_in_window(segment: FlightSegment, start: datetime, end: datetime) -> bool:
    return start <= segment.takeoff_time < end


def _is_american_icao_expr() -> pl.Expr:
    icao = pl.col("icao").str.to_lowercase()
    return (icao >= "a00000") & (icao <= "afffff")


def _pia_or_american_ladd_filter() -> pl.Expr:
    return (
        pl.col("pia").fill_null(False)
        | (_is_american_icao_expr() & pl.col("ladd").fill_null(False))
    )


def _pia_or_american_ladd_icao_filter() -> pl.Expr:
    return _pia_or_american_ladd_filter().any().over("icao")


def _segment_adsb_df(
    df_adsb: pl.DataFrame,
) -> list[FlightSegment]:

    df_adsb = (
        df_adsb
        .filter(
            pl.col("time").is_not_null()
            & pl.col("icao").is_not_null()
            & pl.col("lat").is_not_null()
            & pl.col("lon").is_not_null()
        )
        .sort(["icao", "time"])
    )

    icao_dfs = [
        df_icao
        for _, df_icao in df_adsb.group_by("icao", maintain_order=True)
    ]

    processed_icaos = 0
    segments = []
    for df_icao in icao_dfs:
        segments.extend(identify_flight_segments(_rows_to_position_messages(df_icao)))
        processed_icaos += 1

    return segments


def _sfdps_or_bts_icaos(target_date: date) -> list[str]:
    from data_engineering.bts.read_bts_ontime_to_flights import get_bts_flights_for_day
    from data_engineering.flights.sfdps_to_flights import get_sfdps_flights_day

    source_icaos = set()
    for flights_reader in (get_sfdps_flights_day, get_bts_flights_for_day):
        df_flights = flights_reader(target_date)
        source_icaos.update(
            df_flights.get_column("icao").drop_nulls().cast(pl.String).unique().to_list()
        )
    return sorted(source_icaos)


def _limit_icaos_to_sfdps_or_bts(
    target_date: date,
    icaos: Sequence[str] | None,
) -> list[str]:
    source_icaos = set(_sfdps_or_bts_icaos(target_date))
    if icaos is None:
        return sorted(source_icaos)
    return [icao for icao in icaos if icao in source_icaos]


def _run_airport_model_inference(
    flights_df: pl.DataFrame,
    df_adsb_full: pl.DataFrame,
) -> pl.DataFrame:
    from flights_ai_model_airport.flights_ai_model_utils import get_latest_airport_model_path
    from flights_ai_model_airport.inference import inference

    return inference(
        flights_df,
        get_latest_airport_model_path(),
        df_adsb_full,
    )

def _flights_algorithm_output_path(target_date: date, source: str) -> Path:
    return (
        OUTPUT_DIR
        / "data"
        / "flights"
        / "algorithm"
        / "v1"
        / source
        / f"year={target_date.year}"
        / f"month={target_date.month:02d}"
        / f"day={target_date.day:02d}"
        / "flights.parquet"
    )

def process_icaos(start_dt: datetime, end_dt: datetime, icaos: list[str], no_airports_model: bool = False) -> pl.DataFrame:
    df_adsb = read_adsb_lz(start_dt, end_dt, icaos).collect()
    segments = _segment_adsb_df(df_adsb)
    segments = [
        segment
        for segment in segments
        if _segment_takes_off_in_window(segment, start_dt, end_dt)
    ]
    flights_df = _segments_to_flights_df(segments)
    if not no_airports_model:
        flights_df = _run_airport_model_inference(flights_df, df_adsb)
    return flights_df

def _chunks(items: list[str], num_chunks: int) -> list[list[str]]:
    chunk_size = max(1, math.ceil(len(items) / num_chunks))
    return [
        items[start:start + chunk_size]
        for start in range(0, len(items), chunk_size)
    ]


def run_main(
    target_date: date = date(2026, 3, 1),
    icaos: Sequence[str] | None = None,
    pia_or_american_ladd_only: bool = False,
    sfdps_or_bts_only: bool = False,
    use_adsbx: bool = False,
    max_workers: int | None = None,
    no_airports_model: bool = False,
) -> pl.DataFrame:
    print(f"number of workers: {max_workers or 1}")
    if no_airports_model and use_adsbx:
        raise ValueError("no_airports_model is only supported for ADSB.lol input")
    # figure out how big dataframe is so I can always fit them in ram. 
    day_start, day_end = _day_bounds(target_date)
    start_dt = day_start - RUN_MAIN_CONTEXT
    end_dt = day_end + RUN_MAIN_CONTEXT
    if sfdps_or_bts_only:
        icaos = _limit_icaos_to_sfdps_or_bts(target_date, icaos)
    else:
        icaos = sorted(get_icaos_in_adsb(target_date))
    icao_lists = _chunks(icaos, max_workers or 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_icaos, start_dt, end_dt, icao_list, no_airports_model)
            for icao_list in icao_lists
        ]
        flights_dfs = [future.result() for future in futures]
    flights_df = pl.concat(flights_dfs)

    output_path = _flights_algorithm_output_path(
        target_date,
        "adsbx" if use_adsbx else "adsblol",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flights_df.write_parquet(output_path)
    return flights_df


def _date_arg(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected date in YYYY-MM-DD format") from exc


def _positive_int_arg(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected positive integer")
    return parsed


def _icao_arg(value: str) -> str:
    icao = value.strip().lower()
    if not icao:
        raise argparse.ArgumentTypeError("expected non-empty ICAO")
    return icao


def _date_range(start_date: date, end_date: date):
    current_date = start_date
    while current_date < end_date:
        yield current_date
        current_date += timedelta(days=1)


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("start_date", type=_date_arg)
    parser.add_argument("end_date", type=_date_arg)
    parser.add_argument(
        "--pia-or-american-ladd-only",
        action="store_true",
        help="only segment and output PIA ICAOs or American LADD aircraft",
    )
    parser.add_argument(
        "--sfdps-or-bts-only",
        action="store_true",
        help="only segment ICAOs present in SFDPS or BTS flights for the target date",
    )
    parser.add_argument(
        "--use-adsbx",
        action="store_true",
        help="read ADSBExchange trace parquet instead of ADSB.lol parquet",
    )
    parser.add_argument(
        "--max-workers",
        type=_positive_int_arg,
    )
    parser.add_argument(
        "--icaos",
        nargs="+",
        type=_icao_arg,
        help="only run the algorithm for these ICAOs",
    )
    parser.add_argument(
        "--no-airports-model",
        action="store_true",
        help="use closest first/last ADS-B message airports instead of LightGBM airport inference",
    )
    args = parser.parse_args()

    if args.end_date <= args.start_date:
        parser.error("end_date must be after start_date")
    if args.no_airports_model and args.use_adsbx:
        parser.error("--no-airports-model is only supported for ADSB.lol input")

    for target_date in _date_range(args.start_date, args.end_date):
        print(f"Running flights algorithm for {target_date}")
        flights_df = run_main(
            target_date=target_date,
            pia_or_american_ladd_only=args.pia_or_american_ladd_only,
            sfdps_or_bts_only=args.sfdps_or_bts_only,
            use_adsbx=args.use_adsbx,
            max_workers=args.max_workers,
            icaos=args.icaos,
            no_airports_model=args.no_airports_model,
        )
        print(flights_df)


if __name__ == "__main__":
    cli()
