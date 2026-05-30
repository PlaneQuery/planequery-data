# need to check if airports ident ever leaves USA
# Parameters N planes are using PIA
# They can change PIA callsign either, every flight, every day, every X days
# change some flights in df_flights to pia flights with pia icao: pia_0343, will genereate unique pias
import polars as pl
NUMBER_OF_PIA_PLANES = 10
from data_engineering.flights.flight_type import flights_df_to_flights, flights_to_flights_df
from airports.airportLookup import AirportLookup

airport_lookup = AirportLookup()

import hashlib

def make_pia_icao(flight_id: str) -> str:
    return "pia_" + hashlib.md5(flight_id.encode()).hexdigest()[:8]

def flight_is_in_united_states_airspace(flight) -> bool:
    return (
        airport_lookup.airport_ident_is_in_us_faa_airspace(flight.takeoff_airport_ident)
        and airport_lookup.airport_ident_is_in_us_faa_airspace(flight.landing_airport_ident)
    )

# We'll start with every X days
def process_group(df: pl.DataFrame):
    flights = flights_df_to_flights(df)
    for flight in flights:
        # if flight in United States airspace
        if flight_is_in_united_states_airspace(flight) == False:
            continue
        flight.icao = make_pia_icao(flight.flight_id)
        flight.ladd = False
        flight.pia = True

    df = flights_to_flights_df(flights)
    return df

def main(df: pl.DataFrame):
    df = df.group_by("icao").map_groups(process_group)
    return df
