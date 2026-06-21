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

# these all must map to 1 icao. I don't know who's yet. 
a039a2_flights = [
    Flight(
        icao="a039a2",
        callsign="DCM2493",
        takeoff_time=datetime(2025, 5, 14, 13, 16, 59, 420000),
        landing_time=datetime(2025, 5, 14, 14, 52, 14, 460000),
        takeoff_airport_ident="KJWN",
        landing_airport_ident="KOKC",
        ladd=False,
        pia=True,
    ),
    Flight(
        icao="a039a2",
        callsign="DCM2493",
        takeoff_time=datetime(2025, 5, 14, 21, 46, 34, 580000),
        landing_time=datetime(2025, 5, 14, 22, 56, 23, 540000),
        takeoff_airport_ident="KOKC",
        landing_airport_ident="KBTR",
        ladd=False,
        pia=True,
    ),
    Flight(
        icao="a039a2",
        callsign="DCM2493",
        takeoff_time=datetime(2025, 5, 14, 23, 29, 19, 780000),
        landing_time=datetime(2025, 5, 14, 23, 49, 9, 970000),
        takeoff_airport_ident="KBTR",
        landing_airport_ident="KHSA",
        ladd=False,
        pia=True,
    ),
    Flight(
        icao="a039a2",
        callsign="DCM2493",
        takeoff_time=datetime(2025, 5, 20, 13, 27, 47, 480000),
        landing_time=datetime(2025, 5, 20, 14, 48, 33, 100000),
        takeoff_airport_ident="KHSA",
        landing_airport_ident="KFPR",
        ladd=False,
        pia=True,
    ),
    Flight(
        icao="a039a2",
        callsign="DCM2493",
        takeoff_time=datetime(2025, 5, 20, 16, 16, 9, 320000),
        landing_time=datetime(2025, 5, 20, 17, 55, 2, 620000),
        takeoff_airport_ident="KFPR",
        landing_airport_ident="5MS5",
        ladd=False,
        pia=True,
    ), # THE LADD flight is after this. 
]
