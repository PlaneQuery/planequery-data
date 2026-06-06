import sys
import os
from pathlib import Path
import unittest
from datetime import datetime
import polars as pl

from data_engineering.flights.flight_type import Flight, FLIGHT_POLARS_SCHEMA

flights_taylor_swift = [
    Flight(icao="a03b95", takeoff_time=datetime(2025,5,29,21,47), takeoff_airport_ident="KBNA",  landing_time=datetime(2025,5,30,1,37),  landing_airport_ident="KBUR"),
    Flight(icao="a03b95", takeoff_time=datetime(2025,5,30,18,16), takeoff_airport_ident="KBUR",  landing_time=datetime(2025,5,30,21,38), landing_airport_ident="KM33"),
]

flights_elon_musk = [
    Flight(icao="a835af", takeoff_time=datetime(2025,5,15,6,7),    takeoff_airport_ident="BIKF",  landing_time=datetime(2025,5,15,6,7),   landing_airport_ident="KAUS"),
    Flight(icao="a03c16", takeoff_time=datetime(2025,5,18,23,52),  takeoff_airport_ident="KAUS",  landing_time=datetime(2025,5,19,3,5),   landing_airport_ident="KSJC"),
    Flight(icao="a03c16", takeoff_time=datetime(2025,5,20,0,0),    takeoff_airport_ident="KSJC",  landing_time=datetime(2025,5,20,5,1),   landing_airport_ident="KHHR"),
    Flight(icao="a03c16", takeoff_time=datetime(2025,5,20,5,3,16), takeoff_airport_ident="KHHR",  landing_time=datetime(2025,5,20,5,28),  landing_airport_ident="KAUS"),
    Flight(icao="a03bee", takeoff_time=datetime(2025,5,21,3,40),   takeoff_airport_ident="KAUS",  landing_time=datetime(2025,5,21,5,52),  landing_airport_ident="KIAD"),
    Flight(icao="a03c35", takeoff_time=datetime(2025,5,23,5,19),   takeoff_airport_ident="KIAD",  landing_time=datetime(2025,5,23,10,0),  landing_airport_ident="KSJC"),
    Flight(icao="a03bd6", takeoff_time=datetime(2025,5,27,2,26),   takeoff_airport_ident="KSJC",  landing_time=datetime(2025,5,27,5,29),  landing_airport_ident="KBRO"),
    Flight(icao="a03c3a", takeoff_time=datetime(2025,5,28,20,7),   takeoff_airport_ident="KBRO",  landing_time=datetime(2025,5,28,23,12), landing_airport_ident="KSJC"),
    Flight(icao="a03c39", takeoff_time=datetime(2025,5,30,8),      takeoff_airport_ident="KSJC",  landing_time=datetime(2025,5,30,12),    landing_airport_ident="KIAD"),
    Flight(icao="a03c39", takeoff_time=datetime(2025,5,31,17),     takeoff_airport_ident="KIAD",  landing_time=datetime(2025,5,31,21),    landing_airport_ident="KMSO"),
    Flight(icao="a03c39", takeoff_time=datetime(2025,6,1,1),       takeoff_airport_ident="KMSO",  landing_time=datetime(2025,6,1,3),      landing_airport_ident="KSJC"),
    Flight(icao="a03c0e", takeoff_time=datetime(2025,6,1,18),      takeoff_airport_ident="KSJC",  landing_time=datetime(2025,6,1,20),     landing_airport_ident="KMSO"),
    Flight(icao="a03c0e", takeoff_time=datetime(2025,6,1,21),      takeoff_airport_ident="KMSO",  landing_time=datetime(2025,6,1,23,56),  landing_airport_ident="KAUS"),
    Flight(icao="a03c10", takeoff_time=datetime(2025,6,2,1,26),    takeoff_airport_ident="KAUS",  landing_time=datetime(2025,6,2,3,9),    landing_airport_ident="KSAV"),
    Flight(icao="a835af", takeoff_time=datetime(2025,6,12,13,0),   takeoff_airport_ident="KSAV",  landing_time=datetime(2025,6,12,14,6),  landing_airport_ident="KSAV"),
    Flight(icao="a835af", takeoff_time=datetime(2025,6,12,15,24),  takeoff_airport_ident="KSAV",  landing_time=datetime(2025,6,12,16,12), landing_airport_ident="KSAV"),
]

flights_jeff_bezos = [
    Flight(icao="a03bb1", takeoff_time=datetime(2025,6,2,3,15),  takeoff_airport_ident="KVNY",  landing_time=datetime(2025,6,2,5,15),  landing_airport_ident="KBFI"),
    Flight(icao="a03bb1", takeoff_time=datetime(2025,6,2,17,53), takeoff_airport_ident="KBFI",  landing_time=datetime(2025,6,2,18,28), landing_airport_ident="KMWH"),
    Flight(icao="a03bb1", takeoff_time=datetime(2025,6,2,18,43), takeoff_airport_ident="KMWH",  landing_time=datetime(2025,6,2,19,24), landing_airport_ident="KBLI"),
    Flight(icao="a03bb1", takeoff_time=datetime(2025,6,2,19,49), takeoff_airport_ident="KBLI",  landing_time=datetime(2025,6,2,20,14), landing_airport_ident="KPAE"),
]
custom_flights = [flights_taylor_swift, flights_elon_musk, flights_jeff_bezos]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", type=lambda s: datetime.strptime(s, "%Y-%m-%d"), action="append", dest="days", help="Date(s) to run tests on (YYYY-MM-DD), can be repeated")
    unittest.main()
