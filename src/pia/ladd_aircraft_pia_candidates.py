from datetime import date
from data_engineering.flights.flight_type import flights_df_to_flights, get_flights
import polars as pl
from airports.airportLookup import AirportLookup
from utils import haversine
airport_lookup = AirportLookup()
def potential_pia_flight_count(group):
    # group = group.sort("takeoff_time") # uncessary alread sorted 
    flights = flights_df_to_flights(group)
    previous_flight = None
    count = 0
    for flight in flights:
        if previous_flight is not None:
            if previous_flight.landing_airport_ident != flight.takeoff_airport_ident \
            and airport_lookup.airport_ident_is_in_us_faa_airspace(previous_flight.landing_airport_ident) \
            and airport_lookup.airport_ident_is_in_us_faa_airspace(flight.takeoff_airport_ident):
                distance = haversine(
                    previous_flight.landing_airport_lat,
                    previous_flight.landing_airport_lon,
                    flight.takeoff_airport_lat,
                    flight.takeoff_airport_lon,
                )
                if distance > 600: # cranking the distance is good.
                    count+=1
        previous_flight = flight
    return count

def add_airport_coords(df: pl.DataFrame, prefix: str) -> pl.DataFrame:
    airport_col = f"{prefix}_airport_ident"
    lat_col = f"{prefix}_airport_lat"
    lon_col = f"{prefix}_airport_lon"

    def get_coords(airport_ident: str | None) -> dict[str, float | None]:
        coords = airport_lookup.get_airport_coordinates(airport_ident)

        if coords is None:
            return {lat_col: None, lon_col: None}

        lat, lon = coords
        return {lat_col: lat, lon_col: lon}

    return (
        df.with_columns(
            pl.col(airport_col)
            .map_elements(
                get_coords,
                return_dtype=pl.Struct({
                    lat_col: pl.Float64,
                    lon_col: pl.Float64,
                }),
            )
            .alias("_coords")
        )
        .unnest("_coords")
    )

def get_ladd_pia_candidates(df: pl.DataFrame) -> set[str]:
    df = add_airport_coords(df, "takeoff")
    df = add_airport_coords(df, "landing")
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
    df = get_flights(date(2025, 1, 1), end_date = date(2026, 1, 1))
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
