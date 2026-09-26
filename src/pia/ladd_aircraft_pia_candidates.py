from datetime import date
from flights.algorithm import get_flights_range
from flights.airport_reference import airport_from_reference
from flights.flight_type import flights_df_to_flights
import polars as pl
from airports.runtime import airport_catalog_for_date
from planequery_data.utils import haversine

def potential_pia_flight_count(group):
    # group = group.sort("takeoff_time") # uncessary alread sorted 
    flights = flights_df_to_flights(group)
    previous_flight = None
    count = 0
    for flight in flights:
        if previous_flight is not None:
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
            if (
                landing_airport is not None
                and takeoff_airport is not None
                and landing_airport.is_in_us_faa_airspace
                and takeoff_airport.is_in_us_faa_airspace
            ):
                distance = haversine(
                    landing_airport.lat,
                    landing_airport.lon,
                    takeoff_airport.lat,
                    takeoff_airport.lon,
                )
                if distance > 600: # cranking the distance is good.
                    count+=1
        previous_flight = flight
    return count

def get_ladd_pia_candidates(df: pl.DataFrame) -> set[str]:
    candidates = set()
    for icao_key, group in df.group_by("icao"):
        icao = icao_key[0]
        total_count = group.height
        if total_count == 0:
            continue
        if potential_pia_flight_count(group) / total_count > 0.0:
            candidates.add(icao)
    return candidates

check_icaos = {"aa314f", "a029e0", "a81b13", "a835af"}
if __name__ == "__main__":
    df = get_flights_range(date(2025, 1, 1), date(2026, 1, 1))
    df = df.filter((pl.col("ladd")) & (pl.col("category") != "A7"))
    df = df.unique(subset=["icao", "takeoff_time"])
    total_count = df.get_column("icao").n_unique()
    print(total_count)
    candidates = get_ladd_pia_candidates(df)
    # print percentage
    print(len(candidates))
    print(len(candidates) / total_count)
    for icao in check_icaos:
        print(icao, icao in candidates)

# TODO: Have tests that check if elon musk, jeff bezos, taylor swift, markzuckerberg jets are in this. 
