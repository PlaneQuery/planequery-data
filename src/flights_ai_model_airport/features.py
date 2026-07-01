from datetime import timedelta
import math
import polars as pl
import numpy as np
from adsb.adsb_messages_types import AdsbMessage, adsb_messages_from_parquet_df
from airports.airportLookup import Airport, AirportLookup
from data_engineering.adsb.read_adsb import read_adsb
from data_engineering.flights.flight_type import Flight, flights_df_to_flights
from data_engineering.flights.flight_type import add_flight_id_col, flights_df_to_flights, get_flights

from utils import haversine

AIRPORT_RADIUS_KM = 150.0
airport_lookup = AirportLookup()

# For a given flight ( I can take in flight object) I output all the possible airports and their distances etc.. the features then return
def create_flight_airport_features(first_lat, first_lon, first_baro_altitude_ft: int, takeoff_airport_ident):
    rows = []
    potential_airports: list[Airport] = airport_lookup.getAirportsWithinRadius(first_lat, first_lon, AIRPORT_RADIUS_KM)
    if len(potential_airports) == 0:
        assert takeoff_airport_ident is not None
        airport = airport_lookup.get_Airport_from_airport_ident(takeoff_airport_ident)
        assert airport is not None
        potential_airports = [airport]
        assert len(potential_airports) > 0, f"Could not find airport for takeoff_airport_ident: {takeoff_airport_ident}"
    for airport in potential_airports:
        distance_km = haversine(first_lat, first_lon, airport.lat, airport.lon)
        elevation_above_airport = max(first_baro_altitude_ft - airport.elevation_ft, 0)
        # elevation_diff = first_baro_altitude_ft - airport.elevation_ft
        airport_features = [distance_km, elevation_above_airport]
        label = 1 if airport.ident == takeoff_airport_ident else 0
        rows.append((airport_features, label, airport.ident))
    assert len(rows) > 0, f"No airports found within radius for first_lat: {first_lat}, first_lon: {first_lon}, first_baro_altitude_ft: {first_baro_altitude_ft}, takeoff_airport_ident: {takeoff_airport_ident}"
    return rows


def extract_from_messages(messages: list[AdsbMessage]):
    # need to get first non null track, baro_altitude_ft, veritcal_rate
    i = 0
    first_time = messages[0].time
    first_lat = messages[0].lat
    first_lon = messages[0].lon
    first_baro_altitude_ft = None
    while i < len(messages):
        m = messages[i]
        if m.baro_altitude_ft is not None:
            first_baro_altitude_ft = m.baro_altitude_ft
            break
        if m.time - first_time > timedelta(minutes=5):
            pass


        i += 1
    assert first_baro_altitude_ft is not None, f"No baro_altitude_ft found in {len(messages)} messages"
    return first_time, first_lat, first_lon, first_baro_altitude_ft

def create_features_for_flight(flight: Flight):
    pass
def build_training_data(df_flights: pl.DataFrame):
    X, Y, flight_id_rows, airport_ident_rows = process_flights(df_flights)
    return X, Y, flight_id_rows, airport_ident_rows

def process_flights(df_flights: pl.DataFrame):
    df_flights = add_flight_id_col(df_flights)
    target_date = df_flights.get_column("takeoff_time").dt.date().to_list()[0]
    icaos = df_flights.get_column("icao").unique().to_list()
    df_adsb_full = read_adsb(target_date, icaos=icaos)
    flights = flights_df_to_flights(df_flights)
    X_rows = []
    Y_rows = []
    flight_id_rows = []
    airport_ident_rows = []
    for flight in flights:
        if flight.first_message_time is None or flight.last_message_time is None:
            continue
        df_adsb = df_adsb_full.filter(
            (pl.col("icao") == flight.icao)
            & (pl.col("time") >= flight.first_message_time)
            & (pl.col("time") <= flight.last_message_time)
        )
        if len(df_adsb) < 2:
            continue
        if df_adsb.get_column("baro_altitude_ft").is_not_null().sum() < 2:
            continue
        adsb_messages = adsb_messages_from_parquet_df(df_adsb)
        if adsb_messages[-1].time - adsb_messages[0].time < timedelta(minutes=15):
            continue
        _first_time, first_lat, first_lon, first_baro_altitude_ft = extract_from_messages(adsb_messages) # TODO: Use some sort of features type.
        rows = create_flight_airport_features(first_lat, first_lon, first_baro_altitude_ft, flight.takeoff_airport_ident)
        for row in rows:
            X_rows.append(row[0])
            Y_rows.append(row[1])
            airport_ident_rows.append(row[2])
            flight_id_rows.append(flight.flight_id)

    X = np.asarray(X_rows)
    Y = np.asarray(Y_rows)
    flight_id_rows = np.asarray(flight_id_rows)
    airport_ident_rows = np.asarray(airport_ident_rows)
    return X, Y, flight_id_rows, airport_ident_rows

def build_inference_data(df_flights: pl.DataFrame):
    X, Y, flight_id_rows, airport_ident_rows = process_flights(df_flights)
    return X, flight_id_rows, airport_ident_rows

'''
ground_speed_first
vertical_rate_first
bearing_from_airport_to_first_point: diffrence between track and bearing. Track being direction plane is pointed. Bearing being the degree from current point it needs to be to land. 
canidate_ahead as biniary version of previous. 
required_descent_ft_per_min

Maybe:
distance_projected_1min_to_airport_km
distance_projected_2min_to_airport_km
distance_projected_5min_to_airport_km
distance_projected_10min_to_airport_km

'''