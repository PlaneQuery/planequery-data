# calculate percentage chance that the flight is contionus when takeoff/landing airport ident disagree

from pyparsing import ai

from airports.airportLookup import AirportLookup
from data_engineering.flights.flight_type import Flight
from utils import haversine
airport_lookup = AirportLookup()

def make_flights_continuous(flights:list[Flight]):    
    previous_flight: Flight | None = None
    for flight in flights:
        if previous_flight is None:
            previous_flight = flight
            continue
        elif previous_flight.landing_airport_ident == flight.takeoff_airport_ident:
            previous_flight = flight
            continue
        elif previous_flight.landing_airport_ident is None or flight.takeoff_airport_ident is None:
            previous_flight = flight
            continue
        try:
            takeoff_airport = airport_lookup.ident_index[flight.takeoff_airport_ident]
            landing_airport = airport_lookup.ident_index[previous_flight.landing_airport_ident]
        except KeyError:
            previous_flight = flight
            continue

        takeoff_airport_in_island = airport_lookup.airport_ident_is_in_us_faa_island_airspace(flight.takeoff_airport_ident)
        landing_airport_in_island = airport_lookup.airport_ident_is_in_us_faa_island_airspace(previous_flight.landing_airport_ident)
        
        distance_between_airports = haversine(takeoff_airport.lat, takeoff_airport.lon, landing_airport.lat, landing_airport.lon)

        if distance_between_airports > 100 and takeoff_airport_in_island == landing_airport_in_island == False:
            previous_flight = flight
            continue
        score = 50 # if score <= 50 then use takeoff, if score > 50 then use landing
        if flight.takeoff_airport_ident_distance_km is not None and previous_flight.landing_airport_ident_distance_km is not None:
            if flight.takeoff_airport_ident_distance_km > previous_flight.landing_airport_ident_distance_km:
                score += 20
            elif flight.takeoff_airport_ident_distance_km < previous_flight.landing_airport_ident_distance_km:
                score -= 20
        
        if takeoff_airport.type > landing_airport.type:
            score -= 20
        elif takeoff_airport.type < landing_airport.type:
            score += 20
            

        if takeoff_airport_in_island and not landing_airport_in_island:
            score += 20
        elif not takeoff_airport_in_island and landing_airport_in_island:
            score -= 20
        
        if score <= 50:
            previous_flight.landing_airport_ident = flight.takeoff_airport_ident
        else:
            flight.takeoff_airport_ident = previous_flight.landing_airport_ident
        
        previous_flight = flight

# things to consider: airport distance, airport elevation diffrernece( future), is island, airport_size