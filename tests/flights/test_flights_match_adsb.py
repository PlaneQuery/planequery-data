from datetime import date
import polars as pl
from data_engineering.adsb.read_adsb import read_adsb, read_icaos_from_adsb
from flights.flights_match_adsb import flights_match_adsb
from data_engineering.flights.flight_type import get_flights
# might save this to data/tests
# find an ladd plane that has a lot of flights but their all accurate. 

def test_flights_match_adsb():
    # good one
    target_date = date(2026,3,1)
    icao = "a63b3c"
    df = get_flights(target_date)
    df_flights = df.filter(pl.col("icao") == icao)
    df = read_icaos_from_adsb(target_date, [icao])
    df_adsb = df.filter(pl.col("icao") == icao)

    assert flights_match_adsb(df_flights, df_adsb) == True
    # bad one
    pass

if __name__ == "__main__":
    test_flights_match_adsb()

# TODO: Allow read_adsb to do lazy and just read one icao worht of data. 