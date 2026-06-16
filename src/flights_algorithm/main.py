import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Sequence

import polars as pl

from adsb.adsb_messages_types import AdsbMessageRow


TAKEOFF_SPEED_KT = 35.0
LANDING_SPEED_KT = 55.0
TAKEOFF_ACCELERATION_KT = 20.0
APPROACH_SPEED_KT = 130.0
APPROACH_DECELERATION_KT = -20.0
MAX_LANDING_ALTITUDE_FT = 20_000
HIGH_ALTITUDE_APPROACH_SPEED_DROP_KT = 80.0
AIRBORNE_GAP_MIN_SPEED_KT = 90.0
AIRBORNE_GAP_MIN_ALTITUDE_FT = 1_000
MIN_AIRBORNE_GAP_AVERAGE_SPEED_KT = 70.0
MAX_AIRBORNE_GAP_AVERAGE_SPEED_KT = 700.0
HIGH_ALTITUDE_GAP_MIN_ALTITUDE_FT = 30_000
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
) -> None:
    if takeoff_message is None:
        return
    if landing_message.time - takeoff_message.time < MIN_FLIGHT_DURATION:
        return
    if distance_travelled_km < MIN_FLIGHT_DISTANCE_KM:
        return
    if not _landing_endpoint_makes_sense(landing_message, max_speed_kt):
        return
    takeoff_time = _estimated_takeoff_time(takeoff_message)
    segments.append(
        FlightSegment(
            icao=takeoff_message.icao,
            takeoff_time=takeoff_time,
            landing_time=landing_message.time,
            takeoff_message=takeoff_message,
            landing_message=landing_message,
            distance_travelled_km=distance_travelled_km,
        )
    )


def identify_flight_segments(messages: Sequence[AdsbMessageRow | AdsbPositionMessage]) -> list[FlightSegment]:
    state = FlightState.GROUND
    takeoff_message = None
    landing_candidate = None
    distance_travelled_km = 0.0
    max_speed_kt = None
    last_message = None
    segments = []

    for motion in _motion_edges(messages):
        last_message = motion.curr
        speed = motion.effective_ground_speed_kt
        has_ground_state = _on_ground(motion.prev) is not None or _on_ground(motion.curr) is not None
        if speed is None and not has_ground_state:
            continue

        gap_was_continued = False
        if motion.time_gap > MAX_MESSAGE_GAP:
            if state == FlightState.IN_FLIGHT and _should_continue_across_gap(motion):
                landing_candidate = None
                distance_travelled_km += motion.distance_km
                gap_was_continued = True
            elif state == FlightState.IN_FLIGHT:
                _maybe_add_segment(segments, takeoff_message, motion.prev, distance_travelled_km, max_speed_kt)
                state = FlightState.GROUND
                takeoff_message = None
                landing_candidate = None
                distance_travelled_km = 0.0
                max_speed_kt = None
                continue
            else:
                state = FlightState.GROUND
                takeoff_message = None
                landing_candidate = None
                distance_travelled_km = 0.0
                max_speed_kt = None
                continue

        if state == FlightState.GROUND:
            if _is_takeoff_motion(motion):
                state = FlightState.IN_FLIGHT
                takeoff_message = motion.curr
                landing_candidate = None
                distance_travelled_km = 0.0
                max_speed_kt = speed
            continue

        if speed is not None:
            max_speed_kt = speed if max_speed_kt is None else max(max_speed_kt, speed)
        if not gap_was_continued:
            distance_travelled_km += motion.distance_km
        if _is_landing_motion(motion):
            if landing_candidate is None:
                landing_candidate = motion.curr
            if motion.curr.time - landing_candidate.time >= LANDING_CONFIRMATION_TIME:
                _maybe_add_segment(segments, takeoff_message, landing_candidate, distance_travelled_km, max_speed_kt)
                state = FlightState.GROUND
                takeoff_message = None
                landing_candidate = None
                distance_travelled_km = 0.0
                max_speed_kt = None
        else:
            landing_candidate = None

    if state == FlightState.IN_FLIGHT and landing_candidate is not None:
        _maybe_add_segment(segments, takeoff_message, landing_candidate, distance_travelled_km, max_speed_kt)
    elif state == FlightState.IN_FLIGHT and last_message is not None:
        _maybe_add_segment(segments, takeoff_message, last_message, distance_travelled_km, max_speed_kt)

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
        )
        for row in df.iter_rows(named=True)
    ]


def _segments_to_flights_df(segments: list[FlightSegment]) -> pl.DataFrame:
    from data_engineering.flights.flight_type import FLIGHT_POLARS_SCHEMA

    if not segments:
        return pl.DataFrame(schema=FLIGHT_POLARS_SCHEMA)

    rows = []
    for segment in segments:
        rows.append(
            {
                "icao": segment.icao,
                "callsign": "",
                "registration": "",
                "takeoff_time": segment.takeoff_time,
                "takeoff_airport_ident": "",
                "landing_time": segment.landing_time,
                "landing_airport_ident": "",
                "pia": False,
                "ladd": False,
                "military": False,
                "interesting": False,
                "aircraft_type": "",
                "owner": "",
                "aircraft_description": "",
                "category": "",
            }
        )
    return (
        pl.DataFrame(rows)
        .select(list(FLIGHT_POLARS_SCHEMA.keys()))
        .cast(FLIGHT_POLARS_SCHEMA)
    )


def _day_bounds(target_date: date) -> tuple[datetime, datetime]:
    start = datetime.combine(target_date, datetime.min.time())
    return start, start + timedelta(days=1)


def _segment_takes_off_in_window(segment: FlightSegment, start: datetime, end: datetime) -> bool:
    return start <= segment.takeoff_time < end


def run_main(target_date: date = date(2026, 3, 1), icaos: Sequence[str] | None = None) -> pl.DataFrame:
    from data_engineering.adsb.read_adsb import scan_adsb

    day_start, day_end = _day_bounds(target_date)
    window_start = day_start - RUN_MAIN_CONTEXT
    window_end = day_end + RUN_MAIN_CONTEXT
    scan_dates = [target_date - timedelta(days=1), target_date, target_date + timedelta(days=1)]

    lf_adsb = pl.concat(
        [
            scan_adsb(scan_date, columns=["time", "icao", "lat", "lon", "ground_speed_kt", "on_ground", "baro_altitude_ft"])
            for scan_date in scan_dates
        ]
    ).filter(
        (pl.col("time") >= window_start)
        & (pl.col("time") < window_end)
    )
    if icaos is not None:
        lf_adsb = lf_adsb.filter(pl.col("icao").is_in(list(icaos)))

    df_adsb = (
        lf_adsb
        .collect()
        .filter(
            pl.col("time").is_not_null()
            & pl.col("icao").is_not_null()
            & pl.col("lat").is_not_null()
            & pl.col("lon").is_not_null()
        )
        .sort(["icao", "time"])
    )

    segments = []
    for _, df_icao in df_adsb.group_by("icao", maintain_order=True):
        segments.extend(identify_flight_segments(_rows_to_position_messages(df_icao)))

    segments = [
        segment
        for segment in segments
        if _segment_takes_off_in_window(segment, day_start, day_end)
    ]
    return _segments_to_flights_df(segments)
