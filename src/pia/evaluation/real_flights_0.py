# See if it can identify the Bezos and Elon
# 1. did my flights algorihtm get them correct. Will they be in perfect icaos.
from datetime import date, datetime
from flights.algorithm import get_flights_range
from flights.flight_type import Flight, flights_df_to_flights


def flight(**kwargs) -> Flight:
    defaults = {
        "last_lat": 0.0,
        "last_lon": 0.0,
        "last_baro_altitude_ft": 0,
        "last_geom_altitude_ft": 0,
    }
    defaults.update(kwargs)
    return Flight(**defaults)


elon_flights = [
    flight(
        icao="a835af",
        takeoff_time=datetime(2025, 5, 15, 6, 7),
        landing_time=datetime(2025, 5, 15, 6, 7),
        takeoff_airport_ident="BIKF",
        landing_airport_ident="KAUS",
    ),
    flight(
        icao="a03c16",
        takeoff_time=datetime(2025, 5, 18, 23, 52),
        landing_time=datetime(2025, 5, 19, 3, 5),
        takeoff_airport_ident="KAUS",
        landing_airport_ident="KSJC",
        pia=True,
    ),
    flight(
        icao="a03c16",
        takeoff_time=datetime(2025, 5, 20, 0, 0),
        landing_time=datetime(2025, 5, 20, 5, 1),
        takeoff_airport_ident="KSJC",
        landing_airport_ident="KHHR",
        pia=True,
    ),
]

elon_pia_icaos = set(flight.icao for flight in elon_flights if flight.pia)
_flights_0 = None
_mapping_dict_0 = None

def _load_flights_0():
    global _flights_0, _mapping_dict_0
    if _flights_0 is None or _mapping_dict_0 is None:
        df_flights_0 = get_flights_range(date(2025, 5, 14), date(2025, 5, 21))
        # df_flights_0 = df_flights_0.filter(pl.col("pia"))
        _flights_0 = flights_df_to_flights(df_flights_0)
        _mapping_dict_0 = {}
        for flight in _flights_0:
            if flight.icao in elon_pia_icaos:
                _mapping_dict_0[flight] = "a835af"
    return _flights_0, _mapping_dict_0

def __getattr__(name: str):
    if name == "flights_0":
        flights, _ = _load_flights_0()
        return flights
    if name == "mapping_dict_0":
        _, mapping_dict = _load_flights_0()
        return mapping_dict
    raise AttributeError(name)

# these all must map to 1 icao. I don't know who's yet. 
a039a2_flights = [
    flight(
        icao="a039a2",
        callsign="DCM2493",
        takeoff_time=datetime(2025, 5, 14, 13, 16, 59, 420000),
        landing_time=datetime(2025, 5, 14, 14, 52, 14, 460000),
        takeoff_airport_ident="KJWN",
        landing_airport_ident="KOKC",
        ladd=False,
        pia=True,
    ),
    flight(
        icao="a039a2",
        callsign="DCM2493",
        takeoff_time=datetime(2025, 5, 14, 21, 46, 34, 580000),
        landing_time=datetime(2025, 5, 14, 22, 56, 23, 540000),
        takeoff_airport_ident="KOKC",
        landing_airport_ident="KBTR",
        ladd=False,
        pia=True,
    ),
    flight(
        icao="a039a2",
        callsign="DCM2493",
        takeoff_time=datetime(2025, 5, 14, 23, 29, 19, 780000),
        landing_time=datetime(2025, 5, 14, 23, 49, 9, 970000),
        takeoff_airport_ident="KBTR",
        landing_airport_ident="KHSA",
        ladd=False,
        pia=True,
    ),
    flight(
        icao="a039a2",
        callsign="DCM2493",
        takeoff_time=datetime(2025, 5, 20, 13, 27, 47, 480000),
        landing_time=datetime(2025, 5, 20, 14, 48, 33, 100000),
        takeoff_airport_ident="KHSA",
        landing_airport_ident="KFPR",
        ladd=False,
        pia=True,
    ),
    flight(
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
