import math
import polars as pl
import numpy as np
from airports.airportLookup import AirportLookup
from data_engineering.flights.flight_type import Flight
from utils import haversine

AIRPORT_RADIUS_KM = 50.0
airport_lookup = AirportLookup()

# For a given flight ( I can take in flight object) I output all the possible airports and their distances etc.. the features then return
def create_flight_airport_features(flight: Flight):
    assert flight.first_lat is not None and flight.first_lon is not None, "Flight must have first_lat and first_lon"
    rows = []
    potential_airports = airport_lookup.getAirportsWithinRadius(flight.first_lat, flight.first_lon, AIRPORT_RADIUS_KM)
    for airport in potential_airports:
        distance_km = haversine(flight.first_lat, flight.first_lon, airport.lat, airport.lon)
        # elevation_diff = flight.first_baro_altitude_ft - airport.elevation_ft
        airport_features = [distance_km]
        label = 1 if airport.ident == flight.takeoff_airport_ident else 0
        rows.append((airport_features, label, airport.ident))
    return rows

def create_features_from_adsb_for_flight():
    pass

def build_training_data(flights: list[Flight]):
    X_rows = []
    Y_rows = []
    group_rows = []
    airport_ident_rows = []
    for flight in flights:
        flight_features = create_flight_airport_features(flight)
        for row in flight_features:
            X_rows.append(row[0])
            Y_rows.append(row[1])
            airport_ident_rows.append(row[2])
            group_rows.append(flight.flight_id)

    X = np.asarray(X_rows)
    Y = np.asarray(Y_rows)
    groups = np.asarray(group_rows)
    airport_idents = np.asarray(airport_ident_rows)
    
    return X, Y, groups, airport_idents

def build_inference_data(flights: list[Flight]):
    rows = []
    flight_ids = []
    for flight in flights:
        airport_features = create_flight_airport_features(flight)
        rows.extend(airport_features)
        flight_ids.extend([flight.flight_id] * len(airport_features))
    X = np.asarray([row[0] for row in rows])
    flight_ids = np.asarray(flight_ids)
    airport_idents = np.asarray([row[2] for row in rows]) # this is for now when I'm doing testing on known data.
    return X, flight_ids, airport_idents