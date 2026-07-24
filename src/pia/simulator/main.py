# need to check if airports ident ever leaves USA
# Parameters N planes are using PIA
# They can change PIA callsign either, every flight, every day, every X days
# change some flights in df_flights to pia flights with pia icao: pia_0343, will genereate unique pias
import polars as pl
from data_engineering.flights.flight_type import flights_df_to_flights, flights_to_flights_df
from airports.airport_lookup import AirportLookup
import hashlib
import random

NUMBER_OF_PIA_PLANES = 10
_airport_lookup = None

def _get_airport_lookup() -> AirportLookup:
    global _airport_lookup
    if _airport_lookup is None:
        _airport_lookup = AirportLookup()
    return _airport_lookup


def make_pia_icao(flight_id: str) -> str:
    return "pia_" + hashlib.md5(flight_id.encode()).hexdigest()[:8]

def flight_is_in_united_states_airspace(flight) -> bool:
    airport_lookup = _get_airport_lookup()
    return (
        airport_lookup.airport_ident_is_in_us_faa_airspace(flight.takeoff_airport_ident)
        and airport_lookup.airport_ident_is_in_us_faa_airspace(flight.landing_airport_ident)
    )

# We'll start with every X days

def simulate_pia_flights(df: pl.DataFrame):
    gold_mapping = dict()
    icaos_to_use = (
        df.group_by("icao")
        .len()
        .filter(pl.col("len") > 2)
        .get_column("icao")
        .sort()
        .sample(n=NUMBER_OF_PIA_PLANES, seed=20)
        .to_list()
    )
    print(icaos_to_use)
    def process_group(df: pl.DataFrame):
        flights = flights_df_to_flights(df)
        for i in range(1, len(flights) - 1):
            flight = flights[i]
            # if flight in United States airspace
            if random.random() > 0.1:
                pass
                # continue
            if flight.icao not in icaos_to_use:
                continue
            if flight_is_in_united_states_airspace(flight) == False:
                continue
            real_icao = flight.icao
            flight.icao = make_pia_icao(flight.flight_id)
            flight.ladd = False
            flight.pia = True
            gold_mapping[flight] = real_icao

        df = flights_to_flights_df(flights)
        return df

    df = df.group_by("icao").map_groups(process_group)
    return df, gold_mapping
