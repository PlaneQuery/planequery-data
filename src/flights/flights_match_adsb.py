from data_engineering.flights.flight_type import flights_df_to_flights
import polars as pl
from dataclasses import dataclass
from datetime import datetime, timedelta
from airports.airport_lookup import AirportLookup
from utils import haversine

MAX_AIRPORT_DISTANCE_KM = 10
MAX_AIRPORT_DISTANCE_WITH_TIME_GRACE_KM = 100
AIRPORT_DISTANCE_GRACE_KM_PER_MINUTE = 8
AIRPORT_DWELL_GRACE_PERIOD = timedelta(minutes=10)
MAX_DWELL_GROUND_SPEED_KT = 40
MAX_DWELL_BARO_ALTITUDE_FT = 1000
airport_lookup = AirportLookup()
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
    airport_ident: str,
    target_times: list[datetime],
    use_time_grace: bool = True,
) -> bool:
    airport_coordinates = airport_lookup.get_airport_coordinates(airport_ident)
    if airport_coordinates is None:
        return False

    airport_lat, airport_lon = airport_coordinates
    distance_km = haversine(row.lat, row.lon, airport_lat, airport_lon)
    allowed_distance_km = (
        allowed_airport_distance_km(row.time, target_times)
        if use_time_grace
        else MAX_AIRPORT_DISTANCE_KM
    )
    return distance_km <= allowed_distance_km

def adsb_rows_match_airport(
    adsb_rows: list[_AdsbRow],
    airport_ident: str,
    target_times: list[datetime],
    use_time_grace: bool = True,
) -> bool:
    return all(
        adsb_row_matches_airport(row, airport_ident, target_times, use_time_grace)
        for row in adsb_rows
    )

def adsb_row_is_dwelling_at_airport(row: _AdsbRow, airport_ident: str) -> bool:
    airport_coordinates = airport_lookup.get_airport_coordinates(airport_ident)
    if airport_coordinates is None:
        return False

    airport_lat, airport_lon = airport_coordinates
    if haversine(row.lat, row.lon, airport_lat, airport_lon) > MAX_AIRPORT_DISTANCE_KM:
        return False
    if row.ground_speed_kt is not None and row.ground_speed_kt > MAX_DWELL_GROUND_SPEED_KT:
        return False
    if row.baro_altitude_ft is not None and row.baro_altitude_ft > MAX_DWELL_BARO_ALTITUDE_FT:
        return False
    return True

def adsb_rows_dwell_at_airport(
    adsb_rows: list[_AdsbRow],
    airport_ident: str,
    start_time: datetime,
    end_time: datetime | None = None,
) -> bool:
    dwell_start = start_time + AIRPORT_DWELL_GRACE_PERIOD
    dwell_end = None if end_time is None else end_time - AIRPORT_DWELL_GRACE_PERIOD
    return all(
        adsb_row_is_dwelling_at_airport(row, airport_ident)
        for row in adsb_rows
        if row.time >= dwell_start and (dwell_end is None or row.time <= dwell_end)
    )

def flights_match_adsb(df_flights: pl.DataFrame, df_adsb: pl.DataFrame) -> bool: # TODO: have an option to check accuracy of airprots as well. 
    if df_flights.is_empty() or df_adsb.is_empty():
        return False

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
    assert any(flight.landing_time <= flight.takeoff_time for flight in flights) == False

    # check that takeoff_airport idents match landing airport idents
    # 2. check that from when flights say plane is landed plane did not move
    # 3. check that last adsb message and first adsb message is near the airport. 
    previous_flight = None
    for flight in flights:
        if previous_flight:
            if previous_flight.landing_airport_ident != flight.takeoff_airport_ident:
                return False
            landing_row_idx = closest_time_row_idx(adsb_rows, previous_flight.landing_time, True)
            if landing_row_idx is None:
                return False
            if not adsb_rows_dwell_at_airport(
                adsb_rows[landing_row_idx:],
                previous_flight.landing_airport_ident,
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
            flight.takeoff_airport_ident,
            [flight.takeoff_time]
            if previous_flight is None
            else [previous_flight.landing_time, flight.takeoff_time],
            use_time_grace=previous_flight is not None,
        ):
            return False

        current_landing_row_idx = closest_time_row_idx(adsb_rows, flight.landing_time, True)
        if current_landing_row_idx is None:
            return False
        if not adsb_rows_match_airport(
            [adsb_rows[current_landing_row_idx]],
            flight.landing_airport_ident,
            [flight.landing_time],
        ):
            return False

        previous_flight =  flight

    last_flight = flights[-1]
    landing_row_idx = closest_time_row_idx(adsb_rows, last_flight.landing_time, True)
    if landing_row_idx is None:
        return False

    if not adsb_rows_match_airport(
        adsb_rows[landing_row_idx:],
        last_flight.landing_airport_ident,
        [last_flight.landing_time],
    ):
        return False
    return adsb_rows_dwell_at_airport(
        adsb_rows[landing_row_idx:],
        last_flight.landing_airport_ident,
        last_flight.landing_time,
    )
    # 1. check that all takeoff_landing airport idents equal each other

def get_matching_icaos_in_flights(df_flights_full: pl.DataFrame, df_adsb_full: pl.DataFrame):
    # TODO: this could be faster if you only load the icaos you need for df_adsb instead of full thing. 
    icaos = set()
    for key, df_flights in df_flights_full.group_by("icao"):
        icao = key[0]
        df_adsb = df_adsb_full.filter(pl.col("icao") == icao).sort("time")
        if flights_match_adsb(df_flights, df_adsb):
            icaos.add(icao)
    df = df_flights_full.filter(pl.col("icao").is_in(icaos))
    return df


    
'''
# We may want to make exception for e.g https://globe.adsbexchange.com/?icao=a25ec5&lat=36.525&lon=-96.369&zoom=6.2&showTrace=2026-03-01&leg=1&timestamp=1772386508  Where it did get back to same airport. This is a “touch and go”
'''
