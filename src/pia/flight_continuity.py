from airports.airport_lookup import AirportLookup
from flights.flight_type import Flight
from planequery_data.utils import haversine


airport_lookup = AirportLookup()


def make_flights_continuous(flights: list[Flight]):
    previous_flight: Flight | None = None
    teleports = []
    for flight in flights:
        if previous_flight is None:
            previous_flight = flight
            continue
        if previous_flight.landing_airport_ident == flight.takeoff_airport_ident:
            previous_flight = flight
            continue
        if previous_flight.landing_airport_ident is None or flight.takeoff_airport_ident is None:
            previous_flight = flight
            continue
        try:
            takeoff_airport = airport_lookup.ident_index[flight.takeoff_airport_ident]
            landing_airport = airport_lookup.ident_index[previous_flight.landing_airport_ident]
        except KeyError:
            previous_flight = flight
            continue

        takeoff_in_island = airport_lookup.airport_ident_is_in_us_faa_island_airspace(
            flight.takeoff_airport_ident
        )
        landing_in_island = airport_lookup.airport_ident_is_in_us_faa_island_airspace(
            previous_flight.landing_airport_ident
        )
        takeoff_in_us = airport_lookup.airport_ident_is_in_us_faa_airspace(
            flight.takeoff_airport_ident
        )
        landing_in_us = airport_lookup.airport_ident_is_in_us_faa_airspace(
            previous_flight.landing_airport_ident
        )
        if not takeoff_in_us or not landing_in_us:
            previous_flight = flight
            continue

        distance_km = haversine(
            takeoff_airport.latitude_deg,
            takeoff_airport.longitude_deg,
            landing_airport.latitude_deg,
            landing_airport.longitude_deg,
        )
        if distance_km > 100 and not takeoff_in_island and not landing_in_island:
            teleports.append((previous_flight, flight))
            previous_flight = flight
            continue

        score = 50
        takeoff_distance = getattr(flight, "takeoff_airport_ident_distance_km", None)
        landing_distance = getattr(
            previous_flight,
            "landing_airport_ident_distance_km",
            None,
        )
        if takeoff_distance is not None and landing_distance is not None:
            if takeoff_distance > landing_distance:
                score += 20
            elif takeoff_distance < landing_distance:
                score -= 20

        if takeoff_airport.type > landing_airport.type:
            score -= 20
        elif takeoff_airport.type < landing_airport.type:
            score += 20

        if takeoff_in_island and not landing_in_island:
            score += 20
        elif not takeoff_in_island and landing_in_island:
            score -= 20

        if score <= 50:
            previous_flight.landing_airport_ident = flight.takeoff_airport_ident
        else:
            flight.takeoff_airport_ident = previous_flight.landing_airport_ident
        previous_flight = flight
    return teleports
