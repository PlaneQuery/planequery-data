from attr import dataclass
import polars as pl
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


def create_features_for_plane(flights: list[Flight], messages: list[AdsbMessage]):
    # probably return array of arrays for each leadup (post takeoff_time messages). For now only takeoffs. 
    all_flight_features: list[FlightFeatures] = []
    for i in range(len(flights)):
        all_airport_candidate_features = []
        f = flights[i]
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
        flight_features = FlightFeatures(flight_id=f.flight_id, true_airport=true_airport, candidates=all_airport_candidate_features)

        all_flight_features.append(flight_features)
    return all_flight_features
        # we could find a few minutes. 

# def message_sequences_for_flights 
# for each message_sequence I do stuff for each airport candidate. 

def create_features(df_flights: pl.DataFrame, df_adsb: pl.DataFrame):
    # load ground truth flights
    # load matching adsb
    icaos = df_flights["icao"].unique().to_list()
    for icao in icaos:
        flights = flights_df_to_flights(df_flights.filter(pl.col("icao") == icao))
        messages = adsb_messages_from_parquet_df(df_adsb.filter(pl.col("icao") == icao))
        flight_features = create_features_for_plane(flights, messages)




def get_features(): # for now dataframe in future will be numpy array
    pass

def create_features_for_day(target_dt: date):
    df_flights = flights_features_df(target_dt)
    df_adsb = read_adsb(target_dt, icaos=df_flights["icao"].unique().to_list(), source="adsblol")
    # print(df_flights)
    # print(df_adsb)

if __name__ == "__main__":
    create_features_for_day(date(2026, 3, 1))