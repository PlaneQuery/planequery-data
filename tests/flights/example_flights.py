from datetime import date, datetime
from data_engineering.adsb.read_adsb import read_adsb
from flights.flight_type import Flight
from src.flights.evaluation.flights_test_package import FlightsTestPackage


A0205C_DATE = date(2026, 4, 1)
A0205C_ICAO = "a0205c"
A008C9_DATE = date(2026, 3, 1)
A008C9_ICAO = "a008c9"
AIRPORT_IDS = {
    "KCHS": 3450,
    "KCLT": 3457,
    "KIAH": 3604,
    "KMEM": 3675,
    "KMSP": 3709,
    "KORD": 3754,
    "KRDU": 3844,
    "MMOX": 4736,
}



def _a0205c_flight(
    *,
    callsign: str,
    takeoff_time: datetime,
    takeoff_airport_ident: str,
    landing_time: datetime,
    landing_airport_ident: str,
) -> Flight:
    return Flight(
        icao=A0205C_ICAO,
        registration="N107SY",
        callsign=callsign,
        aircraft_type="E75L",
        aircraft_description="EMBRAER ERJ-170-200 (long wing)",
        owner="SKYWEST AIRLINES INC",
        takeoff_time=takeoff_time,
        takeoff_airport_id=AIRPORT_IDS[takeoff_airport_ident],
        takeoff_airport_ident=takeoff_airport_ident,
        landing_time=landing_time,
        landing_airport_id=AIRPORT_IDS[landing_airport_ident],
        landing_airport_ident=landing_airport_ident,
        military=False,
        interesting=False,
        pia=False,
        ladd=False,
    )


# The confirmed real flight history for N107SY on 2026-04-01. Keep the
# sub-second times here so this can also serve as ground truth for the flights
# algorithm tests.
A0205C_FLIGHTS = [
    _a0205c_flight(
        callsign="SKW5841",
        takeoff_time=datetime(2026, 4, 1, 0, 16, 16, 520000),
        takeoff_airport_ident="KORD",
        landing_time=datetime(2026, 4, 1, 1, 40, 11, 480000),
        landing_airport_ident="KMEM",
    ),
    _a0205c_flight(
        callsign="SKW5235",
        takeoff_time=datetime(2026, 4, 1, 2, 57, 36, 260000),
        takeoff_airport_ident="KMEM",
        landing_time=datetime(2026, 4, 1, 4, 12, 58, 170000),
        landing_airport_ident="KIAH",
    ),
    _a0205c_flight(
        callsign="SKW420W",
        takeoff_time=datetime(2026, 4, 1, 12, 18, 39, 20000),
        takeoff_airport_ident="KIAH",
        landing_time=datetime(2026, 4, 1, 14, 15, 6, 671331),
        landing_airport_ident="MMOX",
    ),
    _a0205c_flight(
        callsign="SKW5876",
        takeoff_time=datetime(2026, 4, 1, 15, 44, 10, 895234),
        takeoff_airport_ident="MMOX",
        landing_time=datetime(2026, 4, 1, 17, 53, 10, 540000),
        landing_airport_ident="KIAH",
    ),
    _a0205c_flight(
        callsign="SKW5620",
        takeoff_time=datetime(2026, 4, 1, 20, 11, 49, 990000),
        takeoff_airport_ident="KIAH",
        landing_time=datetime(2026, 4, 1, 22, 21, 37, 90000),
        landing_airport_ident="KCLT",
    ),
]

A0205C_TEST = FlightsTestPackage(
    flights=A0205C_FLIGHTS,
    start_dt=datetime.combine(A0205C_DATE, datetime.min.time()),
    end_dt=datetime.combine(A0205C_DATE, datetime.max.time()),
)


# The confirmed real flight history for A008C9 on 2026-03-01.
A008C9_FLIGHTS = [
    Flight(
        icao=A008C9_ICAO,
        callsign="",
        registration="",
        takeoff_time=datetime(2026, 3, 1, 0, 1, 21, 820000),
        takeoff_airport_id=AIRPORT_IDS["KRDU"],
        takeoff_airport_ident="KRDU",
        landing_time=datetime(2026, 3, 1, 2, 32, 54, 110000),
        landing_airport_id=AIRPORT_IDS["KMSP"],
        landing_airport_ident="KMSP",
        first_message_time=datetime(2026, 3, 1, 0, 0, 4, 280000),
        first_lat=35.892368,
        first_lon=-78.777938,
        last_message_time=datetime(2026, 3, 1, 2, 34, 55, 250000),
        last_lat=44.885822,
        last_lon=-93.207468,
        pia=False,
        ladd=False,
        military=False,
        interesting=False,
        aircraft_type="BCS1",
        owner="BANK OF UTAH TRUSTEE",
        aircraft_description="AIRBUS A220-100",
        category="A3",
    ),
    Flight(
        icao=A008C9_ICAO,
        callsign="",
        registration="",
        takeoff_time=datetime(2026, 3, 1, 16, 44, 31, 300000),
        takeoff_airport_id=AIRPORT_IDS["KMSP"],
        takeoff_airport_ident="KMSP",
        landing_time=datetime(2026, 3, 1, 18, 50, 4, 310000),
        landing_airport_id=AIRPORT_IDS["KCHS"],
        landing_airport_ident="KCHS",
        first_message_time=datetime(2026, 3, 1, 2, 34, 55, 250000),
        first_lat=44.885822,
        first_lon=-93.207468,
        last_message_time=datetime(2026, 3, 1, 18, 50, 4, 310000),
        last_lat=32.914032,
        last_lon=-80.048859,
        last_baro_altitude_ft=-125,
        pia=False,
        ladd=False,
        military=False,
        interesting=False,
        aircraft_type="BCS1",
        owner="BANK OF UTAH TRUSTEE",
        aircraft_description="AIRBUS A220-100",
        category="A3",
    ),
    Flight(
        icao=A008C9_ICAO,
        callsign="",
        registration="",
        takeoff_time=datetime(2026, 3, 1, 20, 5, 43, 730000),
        takeoff_airport_id=AIRPORT_IDS["KCHS"],
        takeoff_airport_ident="KCHS",
        landing_time=datetime(2026, 3, 1, 22, 56, 28, 320000),
        landing_airport_id=AIRPORT_IDS["KMSP"],
        landing_airport_ident="KMSP",
        first_message_time=datetime(2026, 3, 1, 20, 5, 40, 940000),
        first_lat=32.887247,
        first_lon=-80.044313,
        first_baro_altitude_ft=75,
        last_message_time=datetime(2026, 3, 1, 22, 58, 32, 870000),
        last_lat=44.883445,
        last_lon=-93.203268,
        pia=False,
        ladd=False,
        military=False,
        interesting=False,
        aircraft_type="BCS1",
        owner="BANK OF UTAH TRUSTEE",
        aircraft_description="AIRBUS A220-100",
        category="A3",
    ),
]

A008C9_TEST = FlightsTestPackage(
    flights=A008C9_FLIGHTS,
    start_dt=datetime.combine(A008C9_DATE, datetime.min.time()),
    end_dt=datetime.combine(A008C9_DATE, datetime.max.time()),
)

Flights_Test_Packages: list[FlightsTestPackage] = [
    A008C9_TEST,
]
