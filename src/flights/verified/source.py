from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta

import polars as pl


from airports.catalog import AirportCatalogView

from data_engineering.adsb.read_adsb import ADSBSource, read_adsb
from flights.flight_source import NonADSBFlightSource, get_flights_for_catalog
from flights.providers.bts import BTSFlightSource
from flights.providers.eurocontrol import EurocontrolFlightSource
from flights.providers.eurocontrol_callsigns import assign_callsigns
from flights.providers.sfdps import SFDPSFlightSource
from flights.verified.verification import verify_flights

@dataclass(frozen=True, init=False)
class VerifiedFlightSource(NonADSBFlightSource):
    """Verify reference flights against a fixed ADSB.lol observation source."""

    source: NonADSBFlightSource
    adsb_source: ADSBSource = field(default=ADSBSource.ADSBLOL, init=False)
    target_day_adsb: bool = field(default=False, kw_only=True)

    def __init__(
        self,
        source: NonADSBFlightSource,
        *,
        target_day_adsb: bool = False,
    ):
        if isinstance(source, SFDPSFlightSource):
            version = 8
        elif isinstance(source, BTSFlightSource):
            version = 2
        elif isinstance(source, EurocontrolFlightSource):
            version = 1
        else:
            raise ValueError(
                "VerifiedFlightSource only supports BTS, SFDPS, and Eurocontrol sources"
            )
        resolved_adsb_source = ADSBSource.ADSBLOL
        if (
            isinstance(source, EurocontrolFlightSource)
            and source.callsign_source is not resolved_adsb_source
        ):
            source = replace(source, callsign_source=resolved_adsb_source)
        object.__setattr__(self, "name", f"{source.name}-verified")
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "adsb_source", resolved_adsb_source)
        object.__setattr__(self, "target_day_adsb", target_day_adsb)

    def enrichment_adsb_source(self) -> ADSBSource:
        return self.adsb_source

    def create_flights_for_day(self, dt, airport_catalog: AirportCatalogView) -> pl.DataFrame:
        df_flights = get_flights_for_catalog(self.source, dt, airport_catalog).sort(
            "icao", "takeoff_time"
        )
        if df_flights.is_empty():
            return df_flights
        if isinstance(self.source, EurocontrolFlightSource):
            day_end = datetime.combine(dt + timedelta(days=1), datetime.min.time())
            # Keep the full departure day for sequence checks. Only extend
            # through overnight arrivals when target-day-only is disabled.
            aircraft_ends = df_flights.group_by("icao").agg(
                (
                    pl.lit(day_end)
                    if self.target_day_adsb
                    else pl.max_horizontal(pl.col("landing_time").max(), pl.lit(day_end))
                ).alias("_verification_end")
            )
            df_adsb = read_adsb(
                dt,
                end=aircraft_ends["_verification_end"].max(),
                columns=["icao", "time", "callsign"],
                icaos=df_flights["icao"].unique().to_list(),
                source=self.source.callsign_source,
            ).join(aircraft_ends, on="icao").filter(
                pl.col("time") < pl.col("_verification_end")
            ).drop("_verification_end").sort("icao", "time")
            # Verification may have fetched data missing when the base cache
            # was built. Refresh callsigns from the observations just loaded.
            df_flights = assign_callsigns(df_flights, df_adsb.lazy())
            return verify_flights(df_flights, df_adsb)
        df_adsb = read_adsb(
            dt,
            columns=["icao", "time", "callsign"],
            icaos=df_flights["icao"].unique().to_list(),
            source=self.adsb_source,
        ).sort("icao", "time")
        return verify_flights(df_flights, df_adsb)

if __name__ == "__main__":
    from flights.sources import BTS_SOURCE
    from airports.runtime import airport_catalog_for_date
    from airports.catalog import AirportCatalogView
    from datetime import date
    dt = date(2026, 3, 1)
    airport_catalog = airport_catalog_for_date(dt)
    df_verified = VerifiedFlightSource(BTS_SOURCE).create_flights_for_day(dt, airport_catalog)
    print(df_verified)
