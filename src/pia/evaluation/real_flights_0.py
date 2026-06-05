# See if it can identify the Bezos and Elon
# 1. did my flights algorihtm get them correct. Will they be in perfect icaos.
from datetime import date, datetime
import polars as pl
from data_engineering.flights.flight_type import Flight, flights_df_to_flights, get_flights
from pia.ladd_aircraft_pia_candidates import get_ladd_pia_candidates


elon_flights = [
    Flight(
        icao="a835af",
        takeoff_time=datetime(2025, 5, 15, 6, 7),
        landing_time=datetime(2025, 5, 15, 6, 7),
        takeoff_airport_ident="BIKF",
        landing_airport_ident="KAUS",
    ),
    Flight(
        icao="a03c16",
        takeoff_time=datetime(2025, 5, 18, 23, 52),
        landing_time=datetime(2025, 5, 19, 3, 5),
        takeoff_airport_ident="KAUS",
        landing_airport_ident="KSJC",
        pia=True,
    ),
    Flight(
        icao="a03c16",
        takeoff_time=datetime(2025, 5, 20, 0, 0),
        landing_time=datetime(2025, 5, 20, 5, 1),
        takeoff_airport_ident="KSJC",
        landing_airport_ident="KHHR",
        pia=True,
    ),
]

elon_pia_icaos = set(flight.icao for flight in elon_flights if flight.pia)
mapping_dict_0 = {}

df_flights_0 = get_flights(date(2025, 5, 14), end_date=date(2025, 5, 21))
# df_flights_0 = df_flights_0.filter(pl.col("pia"))
flights_0 = flights_df_to_flights(df_flights_0)
for flight in flights_0:
    if flight.icao in elon_pia_icaos:
        mapping_dict_0[flight.flight_id] = "a835af"