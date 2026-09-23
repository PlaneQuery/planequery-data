import math
import os
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Sequence
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import polars as pl

from data_engineering.adsb.adsb_messages_types import AdsbMessageRow
from data_engineering.adsb.read_adsb import (
    ADSBSource,
    ensure_adsb_source_data,
    get_icaos_in_adsb,
    read_adsb,
    remove_adsb_parquet_days,
)
from flights.segmentation.readsb_legs import mark_leg_indexes


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
MAX_SHORT_SAME_AIRPORT_DURATION = timedelta(minutes=25)
MIN_FLIGHT_DURATION = timedelta(minutes=5)
MIN_FLIGHT_DISTANCE_KM = 5.0
MIN_AIRBORNE_RUN_MESSAGES = 2
MIN_AIRBORNE_RUN_DISTANCE_KM = 1.0
MIN_AIRBORNE_RUN_SPEED_KT = 30
GROUND_TRUTH_ICAOS_DATE = date(2026, 3, 1)
ADSB_ALGORITHM_COLUMNS = [
    "time",
    "icao",
    "callsign",
    "registration",
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
FLIGHT_IDENTITY_COLUMNS = (
    "callsign",
    "registration",
)
FLIGHT_IDENTITY_ALIASES = {
    "callsign": ("callsign", "flight"),
    "registration": ("registration", "r"),
}


@dataclass(frozen=True)
class AdsbPositionMessage:
    time: datetime
    icao: str
    lat: float
    lon: float
    ground_speed_kt: float | None = None
    track_deg: float | None = None
    on_ground: bool | None = None
    baro_altitude_ft: float | None = None
    geom_altitude_ft: int | None = None
    dbFlags: int | None = None
    pia: bool | None = None
    ladd: bool | None = None
    military: bool | None = None
    interesting: bool | None = None
    callsign: str | None = None
    registration: str | None = None
    aircraft_type: str | None = None
    owner: str | None = None
    aircraft_description: str | None = None
    category: str | None = None


@dataclass(frozen=True)
class FlightSegment:
    icao: str
    takeoff_time: datetime
    landing_time: datetime
    takeoff_message: AdsbMessageRow | AdsbPositionMessage
    landing_message: AdsbMessageRow | AdsbPositionMessage
    distance_travelled_km: float = 0.0
    db_flags_counts: tuple[tuple[int, int], ...] = ()
    flight_identity_counts: tuple[
        tuple[str, tuple[tuple[str, int], ...]],
        ...,
    ] = ()
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


def _message_flight_identity(
    message: AdsbMessageRow | AdsbPositionMessage,
    column: str,
) -> str | None:
    for attr in FLIGHT_IDENTITY_ALIASES[column]:
        value = _non_empty_string(getattr(message, attr, None))
        if value is not None:
            return value
    return None


def _flight_identity_counts_for_messages(
    messages: Sequence[AdsbMessageRow | AdsbPositionMessage],
) -> tuple[tuple[str, tuple[tuple[str, int], ...]], ...]:
    return tuple(
        (
            column,
            tuple(
                Counter(
                    value
                    for message in messages
                    if (value := _message_flight_identity(message, column)) is not None
                ).items()
            ),
        )
        for column in FLIGHT_IDENTITY_COLUMNS
    )


def _flight_identity_mode_from_counts(
    flight_identity_counts: tuple[tuple[str, tuple[tuple[str, int], ...]], ...],
) -> dict[str, str]:
    modes = {column: "" for column in FLIGHT_IDENTITY_COLUMNS}
    for column, value_counts in flight_identity_counts:
        if value_counts:
            modes[column] = max(value_counts, key=lambda item: item[1])[0]
    return modes


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


def _aircraft_metadata_mode_from_counts(
    aircraft_metadata_counts: tuple[tuple[str, tuple[tuple[str, int], ...]], ...],
) -> dict[str, str]:
    modes = {column: "" for column in AIRCRAFT_METADATA_COLUMNS}
    for column, value_counts in aircraft_metadata_counts:
        if value_counts:
            modes[column] = max(value_counts, key=lambda item: item[1])[0]
    return modes


def _baro_altitude_int_ft(message: AdsbMessageRow | AdsbPositionMessage) -> int | None:
    return _optional_int(
        getattr(message, "baro_altitude_ft", getattr(message, "alt_baro", None))
    )


def _geom_altitude_ft(message: AdsbMessageRow | AdsbPositionMessage) -> int | None:
    return _optional_int(
        getattr(message, "geom_altitude_ft", getattr(message, "alt_geom", None))
    )


def _flight_endpoint_messages(
    leg_messages: Sequence[AdsbMessageRow | AdsbPositionMessage],
) -> (
    tuple[
        AdsbMessageRow | AdsbPositionMessage,
        AdsbMessageRow | AdsbPositionMessage,
    ]
    | None
):
    """Locate takeoff and landing within a readsb marker-delimited leg.

    readsb's leg markers are separators between flights, commonly placed in
    the middle of a ground dwell. They are not takeoff or landing events.
    Prefer the strongest sustained airborne run and its surrounding ground
    transitions while retaining the leg endpoints as fallbacks for partial
    traces. This prevents isolated bad ``on_ground`` values from becoming
    takeoff or landing events.
    """
    airborne_runs: list[tuple[int, int, int, int]] = []
    run_start = None
    airborne_message_count = 0

    for index, message in enumerate(leg_messages):
        on_ground = _optional_bool(getattr(message, "on_ground", None))
        if on_ground is False:
            if run_start is None:
                run_start = index
            airborne_message_count += 1
        elif on_ground is True and run_start is not None:
            airborne_runs.append(
                (run_start, index - 1, index, airborne_message_count)
            )
            run_start = None
            airborne_message_count = 0

    if run_start is not None:
        airborne_runs.append(
            (
                run_start,
                len(leg_messages) - 1,
                len(leg_messages) - 1,
                airborne_message_count,
            )
        )

    best_run = None
    best_score = None
    for start, end, landing, message_count in airborne_runs:
        if message_count < MIN_AIRBORNE_RUN_MESSAGES:
            continue

        run_messages = leg_messages[start : end + 1]
        distance_km = sum(
            haversine_km(previous.lat, previous.lon, current.lat, current.lon)
            for previous, current in zip(run_messages, run_messages[1:])
        )
        max_speed_kt = max(
            (
                speed
                for message in run_messages
                if (
                    speed := _optional_int(
                        getattr(message, "ground_speed_kt", None)
                    )
                )
                is not None
            ),
            default=None,
        )
        has_altitude = any(
            _baro_altitude_int_ft(message) is not None
            or _geom_altitude_ft(message) is not None
            for message in run_messages
        )
        has_airborne_evidence = (
            distance_km >= MIN_AIRBORNE_RUN_DISTANCE_KM
            or (
                max_speed_kt is not None
                and max_speed_kt >= MIN_AIRBORNE_RUN_SPEED_KT
            )
            or has_altitude
        )
        if not has_airborne_evidence:
            continue

        duration = run_messages[-1].time - run_messages[0].time
        score = (distance_km, duration.total_seconds(), message_count)
        if best_score is None or score > best_score:
            best_run = (start, landing)
            best_score = score

    if best_run is None and any(
        _optional_bool(getattr(message, "on_ground", None)) is True
        for message in leg_messages
    ):
        return None

    if best_run is None:
        return leg_messages[0], leg_messages[-1]

    takeoff_index, landing_index = best_run
    return leg_messages[takeoff_index], leg_messages[landing_index]


def identify_flight_segments(
    messages: Sequence[AdsbMessageRow | AdsbPositionMessage],
) -> list[FlightSegment]:
    """Split an aircraft trace using readsb's ``mark_legs`` algorithm."""
    sorted_messages = sorted(messages, key=lambda message: message.time)
    if len(sorted_messages) < 2:
        return []

    boundaries = [0, *mark_leg_indexes(sorted_messages), len(sorted_messages)]
    segments = []
    for start, end in zip(boundaries, boundaries[1:]):
        leg_messages = sorted_messages[start:end]
        if len(leg_messages) < 2:
            continue

        distance_travelled_km = sum(
            haversine_km(previous.lat, previous.lon, current.lat, current.lon)
            for previous, current in zip(leg_messages, leg_messages[1:])
        )
        endpoint_messages = _flight_endpoint_messages(leg_messages)
        if endpoint_messages is None:
            continue
        takeoff_message, landing_message = endpoint_messages
        if landing_message.time - takeoff_message.time < MIN_FLIGHT_DURATION:
            continue
        if distance_travelled_km < MIN_FLIGHT_DISTANCE_KM:
            continue

        segments.append(
            FlightSegment(
                icao=takeoff_message.icao,
                takeoff_time=takeoff_message.time,
                landing_time=landing_message.time,
                takeoff_message=takeoff_message,
                landing_message=landing_message,
                distance_travelled_km=distance_travelled_km,
                db_flags_counts=_db_flags_counts_for_messages(leg_messages),
                flight_identity_counts=_flight_identity_counts_for_messages(
                    leg_messages
                ),
                aircraft_metadata_counts=_aircraft_metadata_counts_for_messages(
                    leg_messages
                ),
            )
        )
    return segments


def _rows_to_position_messages(df: pl.DataFrame) -> list[AdsbPositionMessage]:
    return [
        AdsbPositionMessage(
            time=row["time"],
            icao=row["icao"],
            lat=row["lat"],
            lon=row["lon"],
            ground_speed_kt=row.get("ground_speed_kt"),
            track_deg=row.get("track_deg"),
            on_ground=row.get("on_ground"),
            baro_altitude_ft=row.get("baro_altitude_ft"),
            geom_altitude_ft=row.get("geom_altitude_ft"),
            dbFlags=row.get("dbFlags"),
            pia=row.get("pia"),
            ladd=row.get("ladd"),
            military=row.get("military"),
            interesting=row.get("interesting"),
            callsign=row.get("callsign"),
            registration=row.get("registration"),
            aircraft_type=row.get("aircraft_type"),
            owner=row.get("owner"),
            aircraft_description=row.get("aircraft_description"),
            category=row.get("category"),
        )
        for row in df.iter_rows(named=True)
    ]


def _get_airport_lookup():
    from airports.airport_lookup import AirportLookup
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
    return haversine_km(
        message.lat,
        message.lon,
        airport.lat,
        airport.lon,
    )


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
    from flights.flight_type import FLIGHT_POLARS_SCHEMA

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
        flight_identity = _flight_identity_mode_from_counts(
            segment.flight_identity_counts
        )
        rows.append(
            {
                "icao": segment.icao,
                **flight_identity,
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


def _segment_takes_off_in_window(
    segment: FlightSegment,
    start: datetime,
    end: datetime,
) -> bool:
    return start <= segment.takeoff_time < end


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

    segments = []
    for df_icao in icao_dfs:
        segments.extend(identify_flight_segments(_rows_to_position_messages(df_icao)))

    return segments


def _sfdps_or_bts_icaos(target_date: date) -> list[str]:
    from flights.flight_source import get_flights
    from flights.providers.bts import BTSFlightSource
    from flights.providers.sfdps import SFDPSFlightSource

    source_icaos = set()
    for source in (SFDPSFlightSource(), BTSFlightSource()):
        df_flights = get_flights(source, target_date)
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


def _ground_truth_icaos() -> list[str]:
    from flights.ground_truth.bts_and_sfdps_to_gold import (
        get_ground_truth_flights_for_day,
    )

    ground_truth_flights = get_ground_truth_flights_for_day(GROUND_TRUTH_ICAOS_DATE)
    return sorted(
        ground_truth_flights.get_column("icao")
        .drop_nulls()
        .cast(pl.String)
        .str.to_lowercase()
        .unique()
        .to_list()
    )


def _limit_icaos_to_ground_truth(icaos: Sequence[str]) -> list[str]:
    ground_truth_icaos = set(_ground_truth_icaos())
    return [icao for icao in icaos if icao in ground_truth_icaos]


def create_flights_from_adsb(
    df_adsb: pl.DataFrame,
    target_date: date,
) -> pl.DataFrame:
    """Segment ADS-B messages and assign endpoint-nearest airports for one day."""
    day_start, day_end = _day_bounds(target_date)
    segments = _segment_adsb_df(df_adsb)
    segments = [
        segment
        for segment in segments
        if _segment_takes_off_in_window(segment, day_start, day_end)
    ]
    return _filter_short_same_airport_flights(_segments_to_flights_df(segments))


def process_icaos(
    day_start: datetime,
    day_end: datetime,
    icaos: list[str],
    source: ADSBSource = ADSBSource.ADSBLOL,
) -> pl.DataFrame:
    source = ADSBSource(source)
    df_adsb = read_adsb(day_start, day_end, icaos=icaos, source=source)
    flights_df = create_flights_from_adsb(df_adsb, day_start.date())
    if len(flights_df) == 0:
        print(f"Empty chunk: processed {len(icaos)} ICAOs, found 0 flights")
        return flights_df
    print(f"Processed {len(icaos)} ICAOs, found {len(flights_df)} flights")
    return flights_df


def _chunks_by_size(items: list[str], chunk_size: int) -> list[list[str]]:
    return [
        items[start:start + chunk_size]
        for start in range(0, len(items), chunk_size)
    ]


def _number_of_workers_default() -> int:
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return 1
    return max(1, cpu_count // 2)


def create_flights_for_day(
    target_date: date = date(2026, 3, 1),
    icaos: Sequence[str] | None = None,
    pia_or_american_ladd_only: bool = False,
    sfdps_or_bts_only: bool = False,
    ground_truth_only: bool = False,
    source: ADSBSource = ADSBSource.ADSBLOL,
    max_workers: int = _number_of_workers_default(),
) -> pl.DataFrame:
    source = ADSBSource(source)
    print(f"number of workers: {max_workers}")

    day_start, day_end = _day_bounds(target_date)
    created_parquet_days = ensure_adsb_source_data(
        day_start,
        day_end,
        source=source,
        pia_or_american_ladd_only=pia_or_american_ladd_only,
    )
    available_icaos = sorted(get_icaos_in_adsb(target_date, source=source))
    if icaos is None:
        icaos = available_icaos
    else:
        requested_icaos = set(icaos)
        icaos = [icao for icao in available_icaos if icao in requested_icaos]
    if sfdps_or_bts_only:
        icaos = _limit_icaos_to_sfdps_or_bts(target_date, icaos)
    if ground_truth_only:
        icaos = _limit_icaos_to_ground_truth(icaos)
    icao_lists = _chunks_by_size(icaos, 500)
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp.get_context("spawn"),
    ) as executor:
        futures = [
            executor.submit(
                process_icaos,
                day_start,
                day_end,
                icao_list,
                source,
            )
            for icao_list in icao_lists
        ]
    flights_dfs = [future.result() for future in futures]
    flights_df = pl.concat(flights_dfs)
    if source == ADSBSource.ADSBLOL:
        remove_adsb_parquet_days(created_parquet_days)
    return flights_df
