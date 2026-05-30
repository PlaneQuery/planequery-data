from data_engineering.flights.flight_type import flights_df_to_flights
import polars as pl
from dataclasses import dataclass
from datetime import datetime
from airports.airportLookup import AirportLookup
import math
from utils import haversine

MAX_AIRPORT_DISTANCE_KM = 16
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

    def get_airport_coords(ident: str) -> tuple[float, float] | None:
        airport = airport_lookup.get_Airport_from_airport_icao(ident)
        if airport is None:
            return None
        return airport.coordinates

    def is_near_airport(row: _AdsbRow, airport_coords: tuple[float, float]) -> bool:
        distance = haversine(row.lat, row.lon, airport_coords[0], airport_coords[1])
        return distance <= MAX_AIRPORT_DISTANCE_KM

    assert any(flight.landing_time <= flight.takeoff_time for flight in flights) == False

    # check that takeoff_airport idents match landing airport idents
    # 2. check that from when flights say plane is landed plane did not move
    # 3. check that last adsb message and first adsb message is near the airport. 
    previous_flight = None
    for flight in flights:
        if previous_flight:
            assert previous_flight.landing_airport_ident == flight.takeoff_airport_ident
            landing_row_idx = closest_time_row_idx(adsb_rows, previous_flight.landing_time, True)
            assert landing_row_idx is not None
        else:
            landing_row_idx = 0

        takeoff_airport_coords = get_airport_coords(flight.takeoff_airport_ident)
        assert takeoff_airport_coords is not None

        takeoff_row_idx = closest_time_row_idx(adsb_rows, flight.takeoff_time, False)
        assert takeoff_row_idx is not None

        in_between_rows = adsb_rows[landing_row_idx:takeoff_row_idx]
        for row in in_between_rows:
            distance = haversine(row.lat, row.lon, *takeoff_airport_coords)
            if distance > 10:
                return False

        previous_flight =  flight
    return True
    # 1. check that all takeoff_landing airport idents equal each other


    
