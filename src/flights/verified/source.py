from collections.abc import Callable
from dataclasses import dataclass

import polars as pl


from airports.catalog import AirportCatalogView

from data_engineering.adsb.read_adsb import read_adsb
from flights.flight_source import NonADSBFlightSource, get_flights
from flights.providers.bts import BTSFlightSource
from flights.providers.sfdps import SFDPSFlightSource
from flights.verified.verification import verify_flights

@dataclass(frozen=True, init=False)
class VerifiedFlightSource(NonADSBFlightSource):
    source: NonADSBFlightSource

    def __init__(self, source: NonADSBFlightSource):
        if isinstance(source, SFDPSFlightSource):
            version = 3
        elif isinstance(source, BTSFlightSource):
            version = 1
        else:
            raise ValueError(
                "VerifiedFlightSource only supports BTS and SFDPS sources"
            )
        object.__setattr__(self, "name", f"{source.name}-verified")
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "source", source)

    def create_flights_for_day(self, dt, airport_catalog: AirportCatalogView) -> pl.DataFrame:
        df_flights = get_flights(self.source, dt)
        assert df_flights.select(["icao", "takeoff_time"]).equals(
        df_flights.sort(["icao", "takeoff_time"]).select(["icao", "takeoff_time"])
        )
        df_adsb = read_adsb(dt)
        return verify_flights(airport_catalog, df_flights, df_adsb)

if __name__ == "__main__":
    from flights.sources import BTS_SOURCE
    from airports.runtime import airport_catalog_for_date
    from airports.catalog import AirportCatalogView
    from datetime import date
    dt = date(2026, 3, 1)
    airport_catalog = airport_catalog_for_date(dt)
    df_verified = VerifiedFlightSource(BTS_SOURCE).create_flights_for_day(dt, airport_catalog)
    print(df_verified)
