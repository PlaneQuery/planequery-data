from attr import dataclass
import polars as pl
import numpy as np
from airports.airport_lookup import Airport, AirportLookup
from data_engineering.adsb.adsb_messages_types import AdsbMessage, adsb_messages_from_parquet_df, closest_time_row_idx
from data_engineering.adsb.read_adsb import read_adsb
from data_engineering.flights.flight_type import Flight, flights_df_to_flights
from flights.ground_truth.bts_and_sfdps_to_gold import get_ground_truth_flights_for_day
from datetime import date, timedelta
from functools import cached_property

from flights_airport_nn.features.ground_truth_join_segmentation_algorithm import flights_features_df
from planequery_data.utils import haversine

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
    flight_id: str
    predicted_airport: Airport
    true_airport: Airport
    features: list[tuple]

    def features_to_numpy(self, max_length: int):
        X = np.zeros((max_length, NUMBER_OF_FEATURES), dtype=np.float32)
        y = np.float32(self.predicted_airport == self.true_airport)
        mask = np.zeros((max_length), dtype=bool)
        features_np = np.asarray(self.features)
        X[0:len(features_np), :] = features_np
        mask[0:len(features_np)] = True
        return X, mask, y
@dataclass
class FlightFeatures():
    flight_id: str
    true_airport: Airport
    candidates: list[AirportCandidateFeatures]

@dataclass
class PlaneFeatures():
    icao: str
    flight_features: list[FlightFeatures]
class Features():

    def __init__(self, df_flights: pl.DataFrame, df_adsb: pl.DataFrame):
        icaos = df_flights["icao"].unique().to_list()
        flights_by_id: dict[str, Flight] = dict()
        plane_features: list[PlaneFeatures] = []
        for icao in icaos:
            flights = flights_df_to_flights(df_flights.filter(pl.col("icao") == icao))
            for flight in flights:
                flights_by_id[flight.flight_id] = flight
            messages = adsb_messages_from_parquet_df(df_adsb.filter(pl.col("icao") == icao))
            plane_features.append(create_features_for_plane(flights, messages))
        self.plane_features = plane_features
        self.flights_by_id = flights_by_id
        self.mapping = dict()
    
    @cached_property
    def max_messages_length(self):
        return max(len(c.features) for p in self.plane_features for f in p.flight_features for c in f.candidates)
    
    def numpy_features(self):
        X = []
        mask = []
        y = []
        candidate_rows: list[AirportCandidateFeatures] = []
        for pf in self.plane_features:
            for ff in pf.flight_features:
                for acf in ff.candidates:
                    x_tmp, mask_tmp, y_tmp = acf.features_to_numpy(self.max_messages_length)
                    X.append(x_tmp)
                    mask.append(mask_tmp)
                    y.append(y_tmp)
                    candidate_rows.append(acf)
        X = np.asarray(X, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        y = np.asarray(y, dtype=np.float32)
        return X, mask, y, candidate_rows



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
        acf = AirportCandidateFeatures(flight_id=f.flight_id, predicted_airport=airport, true_airport=true_airport, features=features)
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
    features = Features(df_flights, df_adsb)
    return features


# I can split by date
def get_features(target_dt: date):

    pass

def create_features_for_day(target_dt: date):
    df_flights = flights_features_df(target_dt)
    df_adsb = read_adsb(target_dt, icaos=df_flights["icao"].unique().to_list(), source="adsblol")
    features = create_features(df_flights, df_adsb)
    return features

if __name__ == "__main__":
    create_features_for_day(date(2026, 3, 1))