from flights.flight_type import flights_df_to_flights
import polars as pl
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal
from airports.models import Airport
from airports.runtime import airport_catalog_for_date
from flights.airport_reference import airport_from_reference
from planequery_data.utils import haversine

ADSB_MATCHING_COLUMNS = [
    "time",
    "icao",
    "lat",
    "lon",
    "on_ground",
    "ground_speed_kt",
    "baro_altitude_ft",
]

MAX_AIRPORT_DISTANCE_KM = 10
MAX_AIRPORT_DISTANCE_WITH_TIME_GRACE_KM = 100
AIRPORT_DISTANCE_GRACE_KM_PER_MINUTE = 8
AIRPORT_DWELL_GRACE_PERIOD = timedelta(minutes=10)
MAX_DWELL_GROUND_SPEED_KT = 40
MAX_DWELL_BARO_ALTITUDE_FT = 1000
@dataclass
class _AdsbRow:
    time: datetime
    lat: float
    lon: float
    on_ground: bool | None = None
    ground_speed_kt: float | None = None
    baro_altitude_ft: int | None = None

from bisect import bisect_left, bisect_right

def closest_time_row_idx(adsb_rows, target_time, lte=True):
    times = [row.time for row in adsb_rows]

    idx = (
        bisect_right(times, target_time) - 1
        if lte
        else bisect_left(times, target_time)
    )

    if idx < 0 or idx >= len(adsb_rows):
        return None

    return idx

def allowed_airport_distance_km(row_time: datetime, target_times: list[datetime]) -> float:
    time_gap_minutes = min(
        abs((row_time - target_time).total_seconds()) / 60
        for target_time in target_times
    )
    return min(
        MAX_AIRPORT_DISTANCE_WITH_TIME_GRACE_KM,
        MAX_AIRPORT_DISTANCE_KM
        + time_gap_minutes * AIRPORT_DISTANCE_GRACE_KM_PER_MINUTE,
    )

def adsb_row_matches_airport(
    row: _AdsbRow,
    airport: Airport | None,
    target_times: list[datetime],
    use_time_grace: bool = True,
) -> bool:
    if airport is None:
        return False
    distance_km = haversine(row.lat, row.lon, airport.lat, airport.lon)
    allowed_distance_km = (
        allowed_airport_distance_km(row.time, target_times)
        if use_time_grace
        else MAX_AIRPORT_DISTANCE_KM
    )
    return distance_km <= allowed_distance_km

def adsb_rows_match_airport(
    adsb_rows: list[_AdsbRow],
    airport: Airport | None,
    target_times: list[datetime],
    use_time_grace: bool = True,
) -> bool:
    return all(
        adsb_row_matches_airport(row, airport, target_times, use_time_grace)
        for row in adsb_rows
    )

def adsb_row_is_dwelling_at_airport(
    row: _AdsbRow,
    airport: Airport | None,
) -> bool:
    if airport is None:
        return False
    if haversine(row.lat, row.lon, airport.lat, airport.lon) > MAX_AIRPORT_DISTANCE_KM:
        return False
    if row.ground_speed_kt is not None and row.ground_speed_kt > MAX_DWELL_GROUND_SPEED_KT:
        return False
    if row.baro_altitude_ft is not None and row.baro_altitude_ft > MAX_DWELL_BARO_ALTITUDE_FT:
        return False
    return True

def adsb_rows_dwell_at_airport(
    adsb_rows: list[_AdsbRow],
    airport: Airport | None,
    start_time: datetime,
    end_time: datetime | None = None,
) -> bool:
    dwell_start = start_time + AIRPORT_DWELL_GRACE_PERIOD
    dwell_end = None if end_time is None else end_time - AIRPORT_DWELL_GRACE_PERIOD
    return all(
        adsb_row_is_dwelling_at_airport(row, airport)
        for row in adsb_rows
        if row.time >= dwell_start and (dwell_end is None or row.time <= dwell_end)
    )

def flights_match_adsb(
    df_flights: pl.DataFrame, df_adsb: pl.DataFrame, start_dt: datetime
) -> bool | Literal["unknown"]:
    if df_flights.is_empty() or df_adsb.is_empty():
        return False

    airport_catalog = airport_catalog_for_date(start_dt.date())
    flights = flights_df_to_flights(df_flights)
    optional_adsb_cols = ["on_ground", "ground_speed_kt", "baro_altitude_ft"]
    adsb_cols = ["time", "lat", "lon"] + [
        col for col in optional_adsb_cols if col in df_adsb.columns
    ]
    adsb_rows = [
        _AdsbRow(
            time=row["time"],
            lat=row["lat"],
            lon=row["lon"],
            on_ground=row.get("on_ground"),
            ground_speed_kt=row.get("ground_speed_kt"),
            baro_altitude_ft=row.get("baro_altitude_ft"),
        )
        for row in df_adsb.select(adsb_cols).iter_rows(named=True)
    ]
    assert any(flight.landing_time <= flight.takeoff_time for flight in flights) == False # TODO: may need to change.

    # Check that stable takeoff airport IDs match stable landing airport IDs.
    # 2. check that from when flights say plane is landed plane did not move
    # 3. check that last adsb message and first adsb message is near the airport. 
    previous_flight = None
    for flight in flights:
        takeoff_airport = airport_from_reference(
            airport_catalog,
            flight.takeoff_airport_id,
        )
        landing_airport = airport_from_reference(
            airport_catalog,
            flight.landing_airport_id,
        )
        if previous_flight:
            previous_landing_airport = airport_from_reference(
                airport_catalog,
                previous_flight.landing_airport_id,
            )
            if (
                previous_flight.landing_airport_id
                != flight.takeoff_airport_id
            ):
                return False
            landing_row_idx = closest_time_row_idx(adsb_rows, previous_flight.landing_time, True)
            if landing_row_idx is None:
                return False
            if not adsb_rows_dwell_at_airport(
                adsb_rows[landing_row_idx:],
                previous_landing_airport,
                previous_flight.landing_time,
                flight.takeoff_time,
            ):
                return False
        else:
            landing_row_idx = 0

        takeoff_row_idx = closest_time_row_idx(adsb_rows, flight.takeoff_time, False)
        if takeoff_row_idx is None:
            return False

        if not adsb_rows_match_airport(
            adsb_rows[landing_row_idx : takeoff_row_idx + 1],
            takeoff_airport,
            [flight.takeoff_time]
            if previous_flight is None
            else [previous_flight.landing_time, flight.takeoff_time],
            use_time_grace=previous_flight is not None,
        ):
            if previous_flight is None and flight.takeoff_time < start_dt + timedelta(hours=8):
                return "unknown"
            return False

        current_landing_row_idx = closest_time_row_idx(adsb_rows, flight.landing_time, True)
        if current_landing_row_idx is None:
            return False
        if not adsb_rows_match_airport(
            [adsb_rows[current_landing_row_idx]],
            landing_airport,
            [flight.landing_time],
        ):
            return False

        previous_flight =  flight

    last_flight = flights[-1]
    landing_row_idx = closest_time_row_idx(adsb_rows, last_flight.landing_time, True)
    if landing_row_idx is None:
        return False

    last_landing_airport = airport_from_reference(
        airport_catalog,
        last_flight.landing_airport_id,
    )

    if not adsb_rows_match_airport(
        adsb_rows[landing_row_idx:],
        last_landing_airport,
        [last_flight.landing_time],
    ):
        return False
    return adsb_rows_dwell_at_airport(
        adsb_rows[landing_row_idx:],
        last_landing_airport,
        last_flight.landing_time,
    )

def add_adsb_match_column(
    df_flights_full: pl.DataFrame,
    df_adsb_full: pl.DataFrame,
    start_dt: datetime,
) -> pl.DataFrame:
    """Add a nullable ADS-B match result for every flight's ICAO.

    ``None`` represents an indeterminate (``"unknown"``) match, while ``True``
    and ``False`` represent definite results.
    """
    if df_flights_full.is_empty():
        return df_flights_full.with_columns(
            pl.Series("adsb_matched", [], dtype=pl.Boolean)
        )

    match_by_icao = {}
    for key, df_flights in df_flights_full.group_by("icao"):
        icao = key[0]
        df_adsb = df_adsb_full.filter(pl.col("icao") == icao).sort("time")
        match_by_icao[icao] = flights_match_adsb(df_flights, df_adsb, start_dt)

    return df_flights_full.with_columns(
        pl.Series(
            "adsb_matched",
            [
                None if match_by_icao[icao] == "unknown" else match_by_icao[icao]
                for icao in df_flights_full.get_column("icao")
            ],
            dtype=pl.Boolean,
        )
    )


def get_matching_icaos_in_flights(
    df_flights_full: pl.DataFrame,
    df_adsb_full: pl.DataFrame,
    start_dt: datetime,
) -> pl.DataFrame:
    return add_adsb_match_column(df_flights_full, df_adsb_full, start_dt).filter(
        pl.col("adsb_matched") == True
    ).drop("adsb_matched")


    
'''
# We may want to make exception for e.g https://globe.adsbexchange.com/?icao=a25ec5&lat=36.525&lon=-96.369&zoom=6.2&showTrace=2026-03-01&leg=1&timestamp=1772386508  Where it did get back to same airport. This is a “touch and go”
'''
