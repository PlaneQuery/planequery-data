# calculate percentage chance that the flight is contionus when takeoff/landing airport ident disagree

from airports.airportLookup import AirportLookup
from data_engineering.flights.flight_type import Flight
airport_lookup = AirportLookup()

def is_teleport():
    
flights: list[Flight] = []
previous_landing_airport_ident = None
for flight in flights:
    if flight.takeoff_airport_ident == flight.landing_airport_ident:
        continue
    if previous_landing_airport_ident and previous_landing_airport_ident != flight.takeoff_airport_ident:
        continue

# things to consider: airport distance, airport elevation diffrernece( future), is island, airport_size