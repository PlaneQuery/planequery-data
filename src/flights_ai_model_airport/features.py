from datetime import datetime, timedelta
import math

import polars as pl
import numpy as np
from data_engineering.adsb.adsb_messages_types import AdsbMessage, adsb_messages_from_parquet_df
from airports.airport_lookup import Airport, Airport_Size_Ranking, AirportLookup
from data_engineering.adsb.read_adsb import read_adsb
from data_engineering.flights.flight_type import FLIGHT_POLARS_SCHEMA, Flight, flights_df_to_flights
from data_engineering.flights.flight_type import add_flight_id_col, flights_df_to_flights, get_flights
from datetime import date
from data_engineering.flights.sfdps_to_flights import get_sfdps_flights_day
from flights.flights_comparison import df_flights_comparision
from flights.flights_match_adsb import ADSB_MATCHING_COLUMNS, get_matching_icaos_in_flights
from flights_ai_model_airport.flights_ai_model_utils import AIRPORT_MODEL_ENDPOINTS
from utils import angle_diff_deg, haversine, initial_bearing_deg, normalize_deg

AIRPORT_RADIUS_KM = 150.0
FEATURE_NAMES = [
    "distance_km",
    "elevation_above_airport_ft",
    "aircraft_category",
    "airport_type",
    "bearing_airport_to_point_deg",
    "bearing_point_to_airport_deg",
    "track_error_deg",
    "direction_error_deg",
    "along_track_km",
    "cross_track_km",
    "candidate_in_expected_direction",
]
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
    df_adsb = read_adsb(
        dt,
        icaos=df_sfdps_flights.get_column("icao").drop_nulls().unique().to_list(),
        columns=ADSB_MATCHING_COLUMNS,
    )
    df_sfdps_flights = get_matching_icaos_in_flights(
        df_sfdps_flights, df_adsb, datetime(dt.year, dt.month, dt.day)
    )
    df_flights = get_flights(dt, no_airports_model=True)
    df = df_flights_comparision(df_flights, df_sfdps_flights, datetime(dt.year, dt.month, dt.day), datetime(dt.year, dt.month, dt.day) + timedelta(days=1), compare_airports=False)
    df = df.filter(pl.col("match_status") == "both")
    df = df.with_columns(
        pl.col("takeoff_airport_ident_df1").alias("takeoff_airport_ident"),
        pl.col("landing_airport_ident_df1").alias("landing_airport_ident"),
    )
    df = df.select(FLIGHT_POLARS_SCHEMA.keys())
    return df

# For a given flight ( I can take in flight object) I output all the possible airports and their distances etc.. the features then return
def create_flight_airport_features(
    lat: float,
    lon: float,
    baro_altitude_ft: int,
    track: float,
    flight: Flight,
    endpoint: str = "takeoff",
):
    target_airport_ident = _flight_airport_ident(flight, endpoint)
    rows = []
    potential_airports: list[Airport] = airport_lookup.getAirportsWithinRadius(lat, lon, AIRPORT_RADIUS_KM)
    # During inference, target_airport_ident is the endpoint selected by the
    # distance-based algorithm. Keep that strong baseline available even when
    # the endpoint point is sparse and more than AIRPORT_RADIUS_KM away. During
    # training this also guarantees that every candidate group has a positive
    # example.
    if target_airport_ident is not None:
        target_airport = airport_lookup.get_Airport_from_airport_ident(target_airport_ident)
        if (
            target_airport is not None
            and all(airport.ident != target_airport_ident for airport in potential_airports)
        ):
            potential_airports.append(target_airport)
    assert len(potential_airports) > 0, (
        f"No airports found within radius for lat: {lat}, lon: {lon}, "
        f"baro_altitude_ft: {baro_altitude_ft}, "
        f"{endpoint}_airport_ident: {target_airport_ident}"
    )
    for airport in potential_airports:
        distance_km = haversine(lat, lon, airport.lat, airport.lon)
        elevation_above_airport = max(baro_altitude_ft - airport.elevation_ft, 0)
        airport_track_features = create_airport_track_features(lat, lon, track, airport.lat, airport.lon, distance_km, endpoint)
        # elevation_diff = first_baro_altitude_ft - airport.elevation_ft
        airport_type = Airport_Size_Ranking[airport.type.name]
        aircraft_category = aircraft_category_to_int(flight.category)
        airport_features = [distance_km, elevation_above_airport, aircraft_category, airport_type] + list(airport_track_features)
        label = 1 if airport.ident == target_airport_ident else 0
        rows.append((airport_features, label, airport.ident))
    assert len(rows) > 0, f"No airports found within radius for lat: {lat}, lon: {lon}, baro_altitude_ft: {baro_altitude_ft}, {endpoint}_airport_ident: {target_airport_ident}"
    return rows

def create_airport_track_features(lat, lon, track, airport_lat, airport_lon, distance_km, phase: str = "takeoff"):

    bearing_airport_to_point_deg = initial_bearing_deg(airport_lat, airport_lon, lat, lon)
    bearing_point_to_airport_deg = initial_bearing_deg(lat, lon, airport_lat, airport_lon)

    if phase == "takeoff":
        expected_track_deg = bearing_airport_to_point_deg
        reverse_track_deg = normalize_deg(track + 180.0)
        direction_error_deg = angle_diff_deg(reverse_track_deg, bearing_point_to_airport_deg)
    elif phase == "landing":
        expected_track_deg = bearing_point_to_airport_deg
        direction_error_deg = angle_diff_deg(track, bearing_point_to_airport_deg)
    else:
        raise ValueError(f"Unsupported airport model endpoint: {phase}")

    track_error_deg = angle_diff_deg(track, expected_track_deg)
    error_rad = math.radians(track_error_deg)

    along_track_km = distance_km * math.cos(error_rad)
    cross_track_km = abs(distance_km * math.sin(error_rad))

    candidate_in_expected_direction = int(direction_error_deg <= 60.0)

    return bearing_airport_to_point_deg, bearing_point_to_airport_deg, track_error_deg, direction_error_deg, along_track_km, cross_track_km, candidate_in_expected_direction

def _flight_airport_ident(flight: Flight, endpoint: str) -> str:
    if endpoint == "takeoff":
        return flight.takeoff_airport_ident
    if endpoint == "landing":
        return flight.landing_airport_ident
    raise ValueError(f"Unsupported airport model endpoint: {endpoint}")

def extract_from_messages(messages: list[AdsbMessage], endpoint: str = "takeoff"):
    if endpoint not in AIRPORT_MODEL_ENDPOINTS:
        raise ValueError(f"Unsupported airport model endpoint: {endpoint}")

    # Parquet row order is not part of its contract. In particular, OpenSky
    # cache generation deduplicates and writes with maintain_order=False.
    ordered_messages = sorted(messages, key=lambda message: message.time)
    messages_to_scan = (
        ordered_messages
        if endpoint == "takeoff"
        else reversed(ordered_messages)
    )

    # Keep position, altitude, and track aligned to one observation instead of
    # combining an endpoint position with motion data from a different time.
    for message in messages_to_scan:
        if message.baro_altitude_ft is None or message.track_deg is None:
            continue
        assert 0 <= message.track_deg <= 360, f"Invalid track_deg: {message.track_deg}"
        return (
            message.time,
            message.lat,
            message.lon,
            message.baro_altitude_ft,
            message.track_deg,
        )

    raise ValueError(
        f"No message with both baro_altitude_ft and track_deg found in "
        f"{len(messages)} messages"
    )

def create_features_for_flight(flight: Flight):
    pass
def build_training_data(training_dates: list[date], endpoint: str = "takeoff"):
    X_parts = []
    Y_parts = []
    flight_id_parts = []
    airport_ident_parts = []
    for dt in training_dates:
        df_flights = get_training_flights(dt)
        df_flights = add_flight_id_col(df_flights)
        icaos = df_flights.get_column("icao").unique().to_list()
        df_adsb_full = read_adsb(dt, icaos=icaos)
        X, Y, flight_id_rows, airport_ident_rows = process_flights(
            df_flights,
            df_adsb_full,
            endpoint,
        )
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

def process_flights(
    df_flights: pl.DataFrame,
    df_adsb_full: pl.DataFrame,
    endpoint: str = "takeoff",
):
    df_flights = add_flight_id_col(df_flights)
    icaos = df_flights.get_column("icao").unique().to_list()
    df_adsb_full = df_adsb_full.filter(pl.col("icao").is_in(icaos))
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
        ).sort("time")
        if len(df_adsb) < 2:
            continue
        if df_adsb.get_column("baro_altitude_ft").is_not_null().sum() < 2:
            continue
        if df_adsb.get_column("track_deg").is_not_null().sum() < 2:
            continue
        adsb_messages = adsb_messages_from_parquet_df(df_adsb)
        if adsb_messages[-1].time - adsb_messages[0].time < timedelta(minutes=15):
            continue
        try:
            _endpoint_time, lat, lon, baro_altitude_ft, track = extract_from_messages(
                adsb_messages,
                endpoint,
            )
        except ValueError:
            # Leave the distance-based airport unchanged when no single
            # observation has the data needed to score candidates reliably.
            continue
        rows = create_flight_airport_features(lat, lon, baro_altitude_ft, track, flight, endpoint)
        for row in rows:
            X_rows.append(row[0])
            Y_rows.append(row[1])
            airport_ident_rows.append(row[2])
            flight_id_rows.append(flight.flight_id)

    X = (
        np.asarray(X_rows, dtype=np.float64)
        if X_rows
        else np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
    )
    Y = np.asarray(Y_rows)
    flight_id_rows = np.asarray(flight_id_rows)
    airport_ident_rows = np.asarray(airport_ident_rows)
    return X, Y, flight_id_rows, airport_ident_rows

def build_inference_data(
    df_flights: pl.DataFrame,
    df_adsb_full: pl.DataFrame,
    endpoint: str = "takeoff",
):
    X, Y, flight_id_rows, airport_ident_rows = process_flights(
        df_flights,
        df_adsb_full,
        endpoint,
    )
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
