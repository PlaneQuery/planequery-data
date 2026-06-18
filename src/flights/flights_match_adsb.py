from data_engineering.flights.flight_type import flights_df_to_flights
import polars as pl
from dataclasses import dataclass
from datetime import datetime
from airports.airportLookup import AirportLookup
import math
from utils import haversine

MAX_AIRPORT_DISTANCE_KM = 10
airport_lookup = AirportLookup()
@dataclass
class _AdsbRow:
    time: datetime
    lat: float
    lon: float

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

def flights_match_adsb(df_flights: pl.DataFrame, df_adsb: pl.DataFrame) -> bool: 
    if df_flights.is_empty() or df_adsb.is_empty():
        return False

    flights = flights_df_to_flights(df_flights)
    adsb_rows = [
        _AdsbRow(time=row["time"], lat=row["lat"], lon=row["lon"])
        for row in df_adsb.select(["time", "lat", "lon"]).iter_rows(named=True)
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
        else:
            landing_row_idx = 0

        takeoff_row_idx = closest_time_row_idx(adsb_rows, flight.takeoff_time, False)
        if takeoff_row_idx is None:
            return False
    
        airport_lat = adsb_rows[takeoff_row_idx].lat
        airport_lon = adsb_rows[takeoff_row_idx].lon

        if previous_flight:
            landing_row = adsb_rows[landing_row_idx]
            airport_lat = landing_row.lat
            airport_lon = landing_row.lon

        in_between_rows = adsb_rows[landing_row_idx:takeoff_row_idx]
        for row in in_between_rows:
            distance = haversine(row.lat, row.lon, airport_lat, airport_lon)
            if distance > MAX_AIRPORT_DISTANCE_KM:
                return False

        previous_flight =  flight

    last_flight = flights[-1]
    landing_row_idx = closest_time_row_idx(adsb_rows, last_flight.landing_time, True)
    if landing_row_idx is None:
        return False

    landing_row = adsb_rows[landing_row_idx]
    airport_lat = landing_row.lat
    airport_lon = landing_row.lon

    for row in adsb_rows[landing_row_idx:]:
        distance = haversine(row.lat, row.lon, airport_lat, airport_lon)
        if distance > MAX_AIRPORT_DISTANCE_KM:
            return False
    return True
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


    
