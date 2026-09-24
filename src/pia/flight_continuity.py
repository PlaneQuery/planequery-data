from airports.models import AirportType
from airports.runtime import airport_catalog_for_date
from flights.airport_reference import airport_from_reference
from flights.flight_type import Flight
from planequery_data.utils import haversine


AIRPORT_SIZE_RANK = {
    AirportType.CLOSED: -1,
    AirportType.BALLOONPORT: 0,
    AirportType.HELIPORT: 0,
    AirportType.SEAPLANE_BASE: 0,
    AirportType.SMALL_AIRPORT: 0,
    AirportType.MEDIUM_AIRPORT: 1,
    AirportType.LARGE_AIRPORT: 2,
}


def make_flights_continuous(flights: list[Flight]):
    previous_flight: Flight | None = None
    teleports = []
    for flight in flights:
        if previous_flight is None:
            previous_flight = flight
            continue

        if previous_flight.landing_airport_id == flight.takeoff_airport_id:
            previous_flight = flight
            continue

        landing_airport = airport_from_reference(
            airport_catalog_for_date(previous_flight.takeoff_time.date()),
            previous_flight.landing_airport_id,
        )
        takeoff_airport = airport_from_reference(
            airport_catalog_for_date(flight.takeoff_time.date()),
            flight.takeoff_airport_id,
        )
        takeoff_in_island = takeoff_airport.is_in_us_faa_island_airspace
        landing_in_island = landing_airport.is_in_us_faa_island_airspace
        if (
            not takeoff_airport.is_in_us_faa_airspace
            or not landing_airport.is_in_us_faa_airspace
        ):
            previous_flight = flight
            continue

        distance_km = haversine(
            takeoff_airport.lat,
            takeoff_airport.lon,
            landing_airport.lat,
            landing_airport.lon,
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

        takeoff_size = AIRPORT_SIZE_RANK[takeoff_airport.type]
        landing_size = AIRPORT_SIZE_RANK[landing_airport.type]
        if takeoff_size > landing_size:
            score -= 20
        elif takeoff_size < landing_size:
            score += 20

        if takeoff_in_island and not landing_in_island:
            score += 20
        elif not takeoff_in_island and landing_in_island:
            score -= 20

        if score <= 50:
            previous_flight.landing_airport_id = takeoff_airport.id
            previous_flight.landing_airport_ident = takeoff_airport.ident
        else:
            flight.takeoff_airport_id = landing_airport.id
            flight.takeoff_airport_ident = landing_airport.ident
        previous_flight = flight
    return teleports
