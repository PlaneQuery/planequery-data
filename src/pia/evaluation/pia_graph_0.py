from datetime import datetime, timedelta
import os
from pathlib import Path
from data_engineering.flights.flight_type import TestFlight

initial_time = datetime(2026,1,1,1)
# I probably want initial state in future. 
start = [
    TestFlight(icao="0", takeoff_time=initial_time, landing_time=initial_time + timedelta(hours=2), takeoff_airport_ident="A", landing_airport_ident="B"),
    TestFlight(icao="1", takeoff_time=initial_time, landing_time=initial_time + timedelta(hours=2), takeoff_airport_ident="A", landing_airport_ident="B"),
    TestFlight(icao="2", takeoff_time=initial_time, landing_time=initial_time + timedelta(hours=2), takeoff_airport_ident="A", landing_airport_ident="B"),
]

complete_flight_path_1 =[
    TestFlight(icao="pia_0", takeoff_time=initial_time + timedelta(hours=4), landing_time=initial_time + timedelta(hours=6), takeoff_airport_ident="B", landing_airport_ident="C"),
    TestFlight(icao="0", takeoff_time=initial_time + timedelta(hours=7), landing_time=initial_time + timedelta(hours=8), takeoff_airport_ident="C", landing_airport_ident="D")
]
complete_flight_path_2 = [
    TestFlight(icao="pia_1", takeoff_time=initial_time+timedelta(hours=4), landing_time=initial_time+timedelta(hours=6), takeoff_airport_ident="B", landing_airport_ident="E"),
    TestFlight(icao="1", takeoff_time=initial_time+timedelta(hours=7), landing_time=initial_time+timedelta(hours=8), takeoff_airport_ident="E", landing_airport_ident="F")
]
complete_flight_path_3 = [
    TestFlight(icao="2", takeoff_time=initial_time+timedelta(hours=4), landing_time=initial_time+timedelta(hours=6), takeoff_airport_ident="B", landing_airport_ident="G")
]

flights_0 = start + complete_flight_path_1 + complete_flight_path_2 + complete_flight_path_3
test_mapping_dict_0 = {'pia_0_2026-01-01_05-00': '0', 'pia_1_2026-01-01_05-00': '1'}

flights_2 = [
    TestFlight(icao="3", takeoff_time=initial_time, landing_time=initial_time + timedelta(hours=2), takeoff_airport_ident="A", landing_airport_ident="B"),
    TestFlight(icao="3", takeoff_time=initial_time + timedelta(hours=4), landing_time=initial_time + timedelta(hours=6), takeoff_airport_ident="B", landing_airport_ident="Q")
] + flights_0
# for number 2 I can probably just add to already existing flights. 
# flights_2 = flights + [
#     TestFlight(icao="0", takeoff_time=initial_time + timedelta(hours=10), landing_time=initial_time+timedelta(hours=12), takeoff_airport_ident="D", landing_airport_ident="Q"),
#     TestFlight(icao="pia_1", takeoff_time=initial_time + timedelta(hours=10), landing_time=initial_time+timedelta(hours=12), takeoff_airport_ident)
# ]

flights_2_category_test= flights_0

# TODO: make something where the ladd plane then lifts off elminating it from a possible pia contender. Since new resolver constraint will mean it has to keep phyiscal nodes. 

# We can do an extreme setup of having like 5 or 6 icaos leave the place let’s do them in time that the edge_score would favor and then see. They can leave like 6 days later 1 icao
start = [
    TestFlight(icao="0", takeoff_time=initial_time, landing_time=initial_time + timedelta(hours=2), takeoff_airport_ident="A", landing_airport_ident="B"),
    TestFlight(icao="1", takeoff_time=initial_time, landing_time=initial_time + timedelta(hours=2), takeoff_airport_ident="A", landing_airport_ident="B"),
    TestFlight(icao="2", takeoff_time=initial_time, landing_time=initial_time + timedelta(hours=2), takeoff_airport_ident="A", landing_airport_ident="B"),
]
complete_flight_path_1 =[
    TestFlight(icao="pia_0", takeoff_time=initial_time + timedelta(hours=4), landing_time=initial_time + timedelta(hours=6), takeoff_airport_ident="B", landing_airport_ident="C"),
]
more = [
    TestFlight(icao="0", takeoff_time=initial_time + timedelta(hours=3) + timedelta(days=20), landing_time=initial_time + timedelta(hours=4) + timedelta(days=20), takeoff_airport_ident="B", landing_airport_ident="C"),
    TestFlight(icao="1",takeoff_time=initial_time + timedelta(hours=3) + timedelta(days=20), landing_time=initial_time + timedelta(hours=4) + timedelta(days=20), takeoff_airport_ident="B", landing_airport_ident="D"),
]

flights_3 = start + complete_flight_path_1 + more
test_mapping_dict_3 = {'pia_0_2026-01-01_05-00': '2'}