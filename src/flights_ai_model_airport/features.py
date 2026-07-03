from datetime import datetime, timedelta
import math
import polars as pl
import numpy as np
from adsb.adsb_messages_types import AdsbMessage, adsb_messages_from_parquet_df
from airports.airportLookup import Airport, Airport_Size_Ranking, AirportLookup
from data_engineering.adsb.read_adsb import read_adsb
from data_engineering.flights.flight_type import FLIGHT_POLARS_SCHEMA, Flight, flights_df_to_flights, pia_or_american_ladd_icao_filter
from data_engineering.flights.flight_type import add_flight_id_col, flights_df_to_flights, get_flights
from datetime import date
from data_engineering.flights.sfdps_to_flights import get_sfdps_flights_day
from flights.flights_comparison import df_flights_comparision
from flights.flights_match_adsb import get_matching_icaos_in_flights
from utils import angle_diff_deg, haversine, initial_bearing_deg, normalize_deg

AIRPORT_RADIUS_KM = 150.0
AIRCRAFT_CATEGORY_RANKING = {
    "A1": 1,
    "A2": 2,
    "A3": 3,
    "A4": 4,
    "A5": 5,
    "A6": 6,
    "A7": 7,
    "B1": 8,
    "B2": 9,
    "B3": 10,
    "B4": 11,
    "B5": 12,
    "B6": 13,
    "B7": 14,
}
airport_lookup = AirportLookup()


def aircraft_category_to_int(category: str) -> int:
    return AIRCRAFT_CATEGORY_RANKING.get(category, 0)


def get_training_flights(dt: date):
    df_sfdps_flights = get_sfdps_flights_day(dt)
    df_sfdps_flights = df_sfdps_flights.filter(pia_or_american_ladd_icao_filter())
    df_adsb = read_adsb(dt, icaos=df_sfdps_flights.get_column("icao").to_list())
    df_sfdps_flights = get_matching_icaos_in_flights(df_sfdps_flights, df_adsb)
    df_flights = get_flights(dt)
    df = df_flights_comparision(df_flights, df_sfdps_flights, datetime(dt.year, dt.month, dt.day), datetime(dt.year, dt.month, dt.day) + timedelta(days=1), compare_airports=False)
    df = df.filter(pl.col("match_status") == "both")
    df = df.with_columns(pl.col("takeoff_airport_ident_df1").alias("takeoff_airport_ident"))
    df = df.select(FLIGHT_POLARS_SCHEMA.keys())
    return df

# For a given flight ( I can take in flight object) I output all the possible airports and their distances etc.. the features then return
def create_flight_airport_features(first_lat, first_lon, first_baro_altitude_ft: int, first_track, flight: Flight):
    takeoff_airport_ident = flight.takeoff_airport_ident
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
        airport_track_features = create_airport_track_features(first_lat, first_lon, first_track, airport.lat, airport.lon, distance_km)
        # elevation_diff = first_baro_altitude_ft - airport.elevation_ft
        airport_type = Airport_Size_Ranking[airport.type.name]
        aircraft_category = aircraft_category_to_int(flight.category)
        airport_features = [distance_km, elevation_above_airport, aircraft_category, airport_type] + list(airport_track_features)
        label = 1 if airport.ident == takeoff_airport_ident else 0
        rows.append((airport_features, label, airport.ident))
    assert len(rows) > 0, f"No airports found within radius for first_lat: {first_lat}, first_lon: {first_lon}, first_baro_altitude_ft: {first_baro_altitude_ft}, takeoff_airport_ident: {takeoff_airport_ident}"
    return rows

def create_airport_track_features(first_lat, first_lon, first_track, airport_lat, airport_lon, distance_km):
    bearing_airport_to_first_deg = initial_bearing_deg(airport_lat, airport_lon, first_lat, first_lon)
    bearing_first_to_airport_deg = initial_bearing_deg(first_lat, first_lon, airport_lat, airport_lon)

    takeoff_track_error_deg = angle_diff_deg(first_track, bearing_first_to_airport_deg)
    reverse_track_deg = normalize_deg(first_track + 180.0)
    airport_behind_error_deg = angle_diff_deg(reverse_track_deg, bearing_airport_to_first_deg)

    error_rad = math.radians(takeoff_track_error_deg)

    along_track_from_airport_km = distance_km * math.cos(error_rad)
    cross_track_from_airport_km = abs(distance_km * math.sin(error_rad))

    candidate_is_behind = int(airport_behind_error_deg <= 60.0)

    return bearing_airport_to_first_deg, bearing_first_to_airport_deg, takeoff_track_error_deg, airport_behind_error_deg, along_track_from_airport_km, cross_track_from_airport_km, candidate_is_behind

def extract_from_messages(messages: list[AdsbMessage]):
    # need to get first non null track, baro_altitude_ft, veritcal_rate
    i = 0
    first_time = messages[0].time
    first_lat = messages[0].lat
    first_lon = messages[0].lon
    first_baro_altitude_ft = None
    first_track = None
    while i < len(messages):
        m = messages[i]
        if m.baro_altitude_ft is not None:
            first_baro_altitude_ft = m.baro_altitude_ft
        if m.track_deg is not None:
            assert m.track_deg >= 0 and m.track_deg <= 360, f"Invalid track_deg: {m.track_deg}"
            first_track = m.track_deg  
        if first_baro_altitude_ft is not None and first_track is not None:
            break
        if m.time - first_time > timedelta(minutes=5):
            pass


        i += 1
    assert first_baro_altitude_ft is not None, f"No baro_altitude_ft found in {len(messages)} messages"
    assert first_track is not None, f"No track_deg found in {len(messages)} messages"
    return first_time, first_lat, first_lon, first_baro_altitude_ft, first_track

def create_features_for_flight(flight: Flight):
    pass
def build_training_data(training_dates: list[date]):
    X_parts = []
    Y_parts = []
    flight_id_parts = []
    airport_ident_parts = []
    for dt in training_dates:
        df_flights = get_training_flights(dt)
        df_flights = add_flight_id_col(df_flights)
        df_flights = df_flights.filter(pia_or_american_ladd_icao_filter())
        X, Y, flight_id_rows, airport_ident_rows = process_flights(df_flights)
        if len(X) == 0:
            continue
        X_parts.append(X)
        Y_parts.append(Y)
        flight_id_parts.append(flight_id_rows)
        airport_ident_parts.append(airport_ident_rows)

    return (
        np.concatenate(X_parts),
        np.concatenate(Y_parts),
        np.concatenate(flight_id_parts),
        np.concatenate(airport_ident_parts),
    )

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
        if df_adsb.get_column("track_deg").is_not_null().sum() < 2:
            continue
        adsb_messages = adsb_messages_from_parquet_df(df_adsb)
        if adsb_messages[-1].time - adsb_messages[0].time < timedelta(minutes=15):
            continue
        first_time, first_lat, first_lon, first_baro_altitude_ft, first_track = extract_from_messages(adsb_messages) # TODO: Use some sort of features type.
        rows = create_flight_airport_features(first_lat, first_lon, first_baro_altitude_ft, first_track, flight)
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
