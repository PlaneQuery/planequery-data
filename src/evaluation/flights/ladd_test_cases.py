from data_engineering.flights.flight_type import Flight
from datetime import datetime
elon_flights = [
    Flight(
        icao="a03c39",
        takeoff_time=datetime(2025, 5, 31, 17),
        landing_time=datetime(2025, 5, 31, 21),
        takeoff_airport_ident="KIAD",
        landing_airport_ident="KMSO",
        pia=True,
    ),

    Flight(
        icao="a03c39",
        takeoff_time=datetime(2025, 6, 1, 1),
        landing_time=datetime(2025, 6, 1, 3),
        takeoff_airport_ident="KMSO",
        landing_airport_ident="KSJC",
        pia=True,
    ),

    Flight(
        icao="a03c0e",
        takeoff_time=datetime(2025, 6, 1, 18),
        landing_time=datetime(2025, 6, 1, 20),
        takeoff_airport_ident="KSJC",
        landing_airport_ident="KMSO",
        pia=True,
    ),

    Flight(
        icao="a03c0e",
        takeoff_time=datetime(2025, 6, 1, 21),
        landing_time=datetime(2025, 6, 1, 23, 56),
        takeoff_airport_ident="KMSO",
        landing_airport_ident="KAUS",
        pia=True,
    ),

    Flight(
        icao="a03c10",
        takeoff_time=datetime(2025, 6, 2, 1, 26),
        landing_time=datetime(2025, 6, 2, 3, 9),
        takeoff_airport_ident="KAUS",
        landing_airport_ident="KSAV",
        pia=True,
    ),
]