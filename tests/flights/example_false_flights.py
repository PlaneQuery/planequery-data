
from datetime import date, datetime
from data_engineering.adsb.read_adsb import ADSBSource, read_adsb
from data_engineering.flights.flight_type import Flight
from src.flights.evaluation.flights_test_package import FlightsTestPackage

# A reduced a0205c itinerary from the flight-data export. The final flight
# crosses midnight, so this package reads through the following day.
A0205C_ICAO = "a0205c"
A0205C_FLIGHTS = [ # WHY IS THIS NOT PASSING THE FLIGHTS_MATCH_ADSB now ??? 
    Flight(
        icao=A0205C_ICAO,
        registration="N107SY",
        takeoff_time=datetime(2026, 4, 1, 0, 16),
        takeoff_airport_ident="KORD",
        landing_time=datetime(2026, 4, 1, 1, 40),
        landing_airport_ident="KMEM",
        category="A3",
    ),
    Flight(
        icao=A0205C_ICAO,
        registration="N107SY",
        takeoff_time=datetime(2026, 4, 1, 2, 57),
        takeoff_airport_ident="KMEM",
        landing_time=datetime(2026, 4, 1, 4, 13),
        landing_airport_ident="KIAH",
        category="A3",
    ),
    Flight(
        icao=A0205C_ICAO,
        registration="N107SY",
        takeoff_time=datetime(2026, 4, 1, 20, 11),
        takeoff_airport_ident="KIAH",
        landing_time=datetime(2026, 4, 1, 22, 20),
        landing_airport_ident="KCLT",
        category="A3",
    ),

]

A0205C_TEST = FlightsTestPackage(
    flights=A0205C_FLIGHTS,
    start_dt=datetime(2026, 4, 1, 0, 0),
    end_dt=datetime(2026, 4, 2, 0, 0),
    adsb_source=ADSBSource.ADSBLOL,
    is_accurate=False
)

False_Flights_Test_Packages: list[FlightsTestPackage] = [
    A0205C_TEST,
]
