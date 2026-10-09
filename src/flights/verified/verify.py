import polars as pl

from flights.flight_type import Flight, flights_df_to_flights


def process_icao(flights: list[Flight]): # TODO: Add this check to verification. PRINT HOW MANY ICAOS THIS ELIMINATES
    for i in range(len(flights) - 1):
        if i > 0:
            f0 = flights[i - 1]
            f1 = flights[i]
            if f0.landing_airport_ident != f1.takeoff_airport_ident:
                return False
            
    return True

def create_reference_flights(df_flights: pl.DataFrame):
    df_flights.sort(["icao", "takeoff_time"])
    keep_icaos = [
    icao
    for (icao,), group_df in df_flights.group_by("icao", maintain_order=True)
    if process_icao(flights_df_to_flights(group_df))
    ]
    final_df_flights = df_flights.filter(
        pl.col("icao").is_in(keep_icaos)
    )
    return final_df_flights
# assert takeoff_time is after landing_time


# The sucess of this is that adsbx network has better precision than adsblol network. Recall should also be better and close to 100% 