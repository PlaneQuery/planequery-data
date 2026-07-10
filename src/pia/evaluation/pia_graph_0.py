import hashlib
from datetime import datetime, timedelta

from data_engineering.flights.flight_type import Flight

initial_time = datetime(2026, 1, 1, 1)

def test_flight(
    icao: str,
    takeoff_time: datetime = initial_time,
    landing_time: datetime | None = None,
    takeoff_airport_ident: str = "A",
    landing_airport_ident: str = "B",
    last_lat: float = 0.0,
    last_lon: float = 0.0,
    last_baro_altitude_ft: int = 0,
    last_geom_altitude_ft: int = 0,
    **kwargs,
) -> Flight:
    if landing_time is None:
        landing_time = takeoff_time + timedelta(hours=2)

    flight_id = icao + "_" + takeoff_time.strftime("%Y-%m-%d_%H-%M")
    defaults = {
        "pia": icao.startswith("pia"),
        "callsign": hashlib.sha1(flight_id.encode()).hexdigest()[:8].upper(),
    }
    defaults.update(kwargs)

    return Flight(
        icao=icao,
        takeoff_time=takeoff_time,
        landing_time=landing_time,
        takeoff_airport_ident=takeoff_airport_ident,
        landing_airport_ident=landing_airport_ident,
        last_lat=last_lat,
        last_lon=last_lon,
        last_baro_altitude_ft=last_baro_altitude_ft,
        last_geom_altitude_ft=last_geom_altitude_ft,
        **defaults,
    )

# I probably want initial state in future. 
start = [
    test_flight(icao="0"),
    test_flight(icao="1"),
    test_flight(icao="2"),
]

complete_flight_path_1 =[
    test_flight(icao="pia_0", takeoff_time=initial_time + timedelta(hours=4), landing_time=initial_time + timedelta(hours=6), takeoff_airport_ident="B", landing_airport_ident="C"),
    test_flight(icao="0", takeoff_time=initial_time + timedelta(hours=7), landing_time=initial_time + timedelta(hours=8), takeoff_airport_ident="C", landing_airport_ident="D")
]
complete_flight_path_2 = [
    test_flight(icao="pia_1", takeoff_time=initial_time+timedelta(hours=4), landing_time=initial_time+timedelta(hours=6), takeoff_airport_ident="B", landing_airport_ident="E"),
    test_flight(icao="1", takeoff_time=initial_time+timedelta(hours=7), landing_time=initial_time+timedelta(hours=8), takeoff_airport_ident="E", landing_airport_ident="F")
]
complete_flight_path_3 = [
    test_flight(icao="2", takeoff_time=initial_time+timedelta(hours=4), landing_time=initial_time+timedelta(hours=6), takeoff_airport_ident="B", landing_airport_ident="G")
]

flights_0 = start + complete_flight_path_1 + complete_flight_path_2 + complete_flight_path_3
test_mapping_dict_0 = {complete_flight_path_1[0]: "0", complete_flight_path_2[0]: "1"}

flights_2 = [
    test_flight(icao="3"),
    test_flight(icao="3", takeoff_time=initial_time + timedelta(hours=4), landing_time=initial_time + timedelta(hours=6), takeoff_airport_ident="B", landing_airport_ident="Q")
] + flights_0
# for number 2 I can probably just add to already existing flights. 
# flights_2 = flights + [
#     test_flight(icao="0", takeoff_time=initial_time + timedelta(hours=10), landing_time=initial_time+timedelta(hours=12), takeoff_airport_ident="D", landing_airport_ident="Q"),
#     test_flight(icao="pia_1", takeoff_time=initial_time + timedelta(hours=10), landing_time=initial_time+timedelta(hours=12), takeoff_airport_ident)
# ]

flights_2_category_test= flights_0

# TODO: make something where the ladd plane then lifts off elminating it from a possible pia contender. Since new resolver constraint will mean it has to keep phyiscal nodes. 

# We can do an extreme setup of having like 5 or 6 icaos leave the place let’s do them in time that the edge_score would favor and then see. They can leave like 6 days later 1 icao
start = [
    test_flight(icao="0"),
    test_flight(icao="1"),
    test_flight(icao="2"),
]
complete_flight_path_1 =[
    test_flight(icao="pia_0", takeoff_time=initial_time + timedelta(hours=4), landing_time=initial_time + timedelta(hours=6), takeoff_airport_ident="B", landing_airport_ident="C"),
]
more = [
    test_flight(icao="0", takeoff_time=initial_time + timedelta(hours=3) + timedelta(days=20), landing_time=initial_time + timedelta(hours=4) + timedelta(days=20), takeoff_airport_ident="B", landing_airport_ident="C"),
    test_flight(icao="1", takeoff_time=initial_time + timedelta(hours=3) + timedelta(days=20), landing_time=initial_time + timedelta(hours=4) + timedelta(days=20), takeoff_airport_ident="B", landing_airport_ident="D"),
]

flights_3 = start + complete_flight_path_1 + more
test_mapping_dict_3 = {complete_flight_path_1[0]: "2"}
