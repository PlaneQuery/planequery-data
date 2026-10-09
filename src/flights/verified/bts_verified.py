from dataclasses import dataclass
import polars as pl
from adsb.adsb_callsign_history import get_callsign_history
from airports.catalog import AirportCatalogView
from data_engineering.adsb.read_adsb import read_adsb
from flights.flight_source import NonADSBFlightSource, get_flights
from flights.providers.bts import BTSFlightSource

def validation(df_history: pl.DataFrame, df_flights: pl.DataFrame): # TODO: move to bts verfied/refernce file
    df_history = df_history.select(["icao", "callsign"])
    df_flights = df_flights.select(["icao", "callsign"])
    icaos = df_flights.get_column("icao").unique().to_list()
    valid_icaos = []
    for icao in icaos:
        df_0 = df_history.filter(pl.col("icao") == icao)
        df_1 = df_flights.filter(pl.col("icao") == icao)
        if len(df_0) != len(df_1):
            continue
        if len(df_1) < 4:
            continue
        if df_0.get_column("callsign").to_list() != df_1.get_column("callsign").to_list():
            continue
        valid_icaos.append(icao)
    return valid_icaos

@dataclass(frozen=True)
class BTSVerifiedFlightSource(NonADSBFlightSource):
    name: str = "bts_verified"
    version: int = 1

    def create_flights_for_day(self, dt, airport_catalog: AirportCatalogView) -> pl.DataFrame:
        df_flights = get_flights(BTSFlightSource(), dt)
        icaos = df_flights.get_column("icao").unique().to_list()
        df_adsb = read_adsb(dt, icaos=icaos)
        df_history = get_callsign_history(df_adsb)
        valid_icaos = validation(df_history, df_flights)
        df_verified = df_flights.filter(pl.col("icao").is_in(valid_icaos))
        return df_verified
