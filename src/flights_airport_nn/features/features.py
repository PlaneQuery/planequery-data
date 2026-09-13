from attr import dataclass
import polars as pl
import numpy as np
from airports.airport_lookup import Airport, AirportLookup
from data_engineering.adsb.adsb_messages_types import AdsbMessage, adsb_messages_from_parquet_df, closest_time_row_idx
from data_engineering.adsb.read_adsb import read_adsb
from data_engineering.flights.flight_type import Flight, flights_df_to_flights
from flights.ground_truth.bts_and_sfdps_to_gold import get_ground_truth_flights_for_day
from datetime import date, timedelta

from flights_airport_nn.features.ground_truth_join_segmentation_algorithm import flights_features_df
from utils import haversine

FEATURE_PERIOD = timedelta(minutes=15)
airport_lookup = AirportLookup()
NUMBER_OF_FEATURES = 2 # distance and time delta
# TODO: Must use segmentation takeoff/landing time now the ground_truth takeoff/landing time. 

def features_df_to_numpy():
    pass

def normalize_features():
    pass

@dataclass
class AirportCandidateFeatures:
    predicted_airport: Airport
    features: list[tuple]

@dataclass
class FlightFeatures():
    flight_id: str
    true_airport: Airport
    candidates: list[AirportCandidateFeatures]

@dataclass
class PlaneFeatures():
    icao: str
    flight_features: list[FlightFeatures]
@dataclass
class Features():
    plane_features: list[PlaneFeatures]
    def max_features_length(self):
        return max(len(c.features) for p in self.plane_features for f in p.flight_features for c in f.candidates)

    # probably do some numpy creation here. Be able to split features here. Or maybe split in seperate function...

def create_features_for_flight(f: Flight, messages: list[AdsbMessage]):
    all_airport_candidate_features: list[AirportCandidateFeatures] = []
    takeoff_time = f.takeoff_time
    idx = closest_time_row_idx(messages, takeoff_time, lte=False)
    assert idx is not None
    first_m = messages[idx]
    next_time = first_m.time + FEATURE_PERIOD
    idx_final = closest_time_row_idx(messages, next_time)
    assert idx_final is not None
    sub_messages = messages[idx:idx_final+1]
    true_airport = f.takeoff_airport
    possible_airports = airport_lookup.getAirportsWithinRadius(first_m.lat, first_m.lon, 100)
    if true_airport not in possible_airports:
        possible_airports.append(true_airport)
    features = []
    for airport in possible_airports:
        features = []
        airport_lat = airport.lat
        airport_lon = airport.lon
        for j in range(len(sub_messages)):
            m = sub_messages[j]
            distance = haversine(m.lat, m.lon, airport_lat, airport_lon)
            if j == 0:
                time_delta = 0
            else:
                time_delta = (m.time - sub_messages[j-1].time).total_seconds()
            features.append((distance,time_delta))
        acf = AirportCandidateFeatures(predicted_airport=airport, features=features)
        all_airport_candidate_features.append(acf)
    ff = FlightFeatures(flight_id=f.flight_id, true_airport=true_airport, candidates=all_airport_candidate_features)
    return ff

def create_features_for_plane(flights: list[Flight], messages: list[AdsbMessage]):
    # probably return array of arrays for each leadup (post takeoff_time messages). For now only takeoffs. 
    all_flight_features: list[FlightFeatures] = []
    for i in range(len(flights)):
        ff = create_features_for_flight(flights[i], messages)
        all_flight_features.append(ff)
    pf = PlaneFeatures(icao=flights[0].icao, flight_features=all_flight_features)
    return pf
        # we could find a few minutes. 

# def message_sequences_for_flights 
# for each message_sequence I do stuff for each airport candidate. 

def create_features(df_flights: pl.DataFrame, df_adsb: pl.DataFrame):
    # load ground truth flights
    # load matching adsb
    icaos = df_flights["icao"].unique().to_list()
    plane_features: list[PlaneFeatures] = []
    for icao in icaos:
        flights = flights_df_to_flights(df_flights.filter(pl.col("icao") == icao))
        messages = adsb_messages_from_parquet_df(df_adsb.filter(pl.col("icao") == icao))
        plane_features.append(create_features_for_plane(flights, messages))
    features = Features(plane_features=plane_features)
    return features

def create_training_features_np(flight_features: list[FlightFeatures], max_message_length: int):
    samples: list[np.ndarray] = []
    targets = []
    flight_mapping = dict()
    for f in flight_features:
        true_airport = f.true_airport
        for candidate in f.candidates:
            airport = candidate.predicted_airport
            features = candidate.features
            samples.append(np.asarray(features))
            targets.append(1 if airport == true_airport else 0)
        flight_mapping[f.flight_id] = (len(samples) - len(f.candidates), len(samples))
    X = np.zeros((len(samples), max_message_length, NUMBER_OF_FEATURES), dtype=np.float32)
    mask = np.zeros((len(samples), max_message_length), dtype=bool)

    for index, sample in enumerate(samples):
        length = len(sample)
        X[index, :length, :] = sample
        mask[index, :length] = True

    y = np.asarray(targets, dtype=np.float32)

    return X, mask, y, flight_mapping

def get_features(): # for now dataframe in future will be numpy array
    pass

def create_features_for_day(target_dt: date):
    df_flights = flights_features_df(target_dt)
    df_adsb = read_adsb(target_dt, icaos=df_flights["icao"].unique().to_list(), source="adsblol")
    features = create_features(df_flights, df_adsb)
    max_message_length = features.max_features_length()
    plane_features = features.plane_features
    # icao_range mapping to do train/test split
    all_flight_features = []
    for pf in plane_features:
        all_flight_features.extend(pf.flight_features)
    X, mask, y,flight_mapping = create_training_features_np(all_flight_features, max_message_length)
    return X, mask, y

if __name__ == "__main__":
    create_features_for_day(date(2026, 3, 1))