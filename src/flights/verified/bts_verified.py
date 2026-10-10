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

def check_consecutive_airports(df: pl.DataFrame):
    valid_icaos = (
    df
    .group_by("icao", maintain_order=True)
    .agg(
        pl.col("landing_airport_ident")
        .shift(1)
        .eq_missing(pl.col("takeoff_airport_ident"))
        .slice(1)
        .all()
        .alias("is_valid")
    )
    .filter(pl.col("is_valid"))
    .get_column("icao")
    .to_list()
    )
    return valid_icaos
@dataclass(frozen=True)
class BTSVerifiedFlightSource(NonADSBFlightSource):
    name: str = "bts_verified"
    version: int = 1

    def create_flights_for_day(self, dt, airport_catalog: AirportCatalogView) -> pl.DataFrame:
        df_flights = get_flights(BTSFlightSource(), dt)
        assert df_flights.select(["icao", "takeoff_time"]).equals(
        df_flights.sort(["icao", "takeoff_time"]).select(["icao", "takeoff_time"])
        )
        num_icaos = df_flights.get_column("icao").n_unique()
        valid_icaos = check_consecutive_airports(df_flights)
        print(f"Filtered {num_icaos - len(valid_icaos)} ICAOs out of {num_icaos} due to non-consecutive airports")
        df_flights = df_flights.filter(pl.col("icao").is_in(valid_icaos))
        icaos = df_flights.get_column("icao").unique().to_list()
        df_adsb = read_adsb(dt, icaos=icaos)
        df_history = get_callsign_history(df_adsb)
        valid_icaos = validation(df_history, df_flights)
        df_verified = df_flights.filter(pl.col("icao").is_in(valid_icaos))
        print(f"Filtered {len(icaos) - len(valid_icaos)} ICAOs out of {len(icaos)} due to callsign history mismatch")
        return df_verified

if __name__ == "__main__":
    from airports.catalog import AirportCatalogView
    from datetime import date
    airport_catalog = None
    df_verified = BTSVerifiedFlightSource().create_flights_for_day(date(2026, 3, 1), airport_catalog)
    print(df_verified)