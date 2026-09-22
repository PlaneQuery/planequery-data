from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

from data_engineering.adsb.read_adsb import ADSBSource, read_adsb
from data_engineering.utils import OUTPUT_DIR


@dataclass(frozen=True)
class FlightSource(ABC):
    name: str
    version: int = 1
    uses_adsb_source: bool = False

@dataclass(frozen=True)
class ADSBFlightSource(FlightSource):
    @abstractmethod
    def create_flights_for_day(
        self,
        df_adsb: pl.DataFrame,
    ) -> pl.DataFrame:
        ...


@dataclass(frozen=True)
class NonADSBFlightSource(FlightSource):
    @abstractmethod
    def create_flights_for_day(
        self,
        dt: date,
    ) -> pl.DataFrame:
        ...

def flight_path(
    source: FlightSource,
    dt: date,
    adsb_src: ADSBSource | None = None,
) -> Path:
    if source.uses_adsb_source and adsb_src is None:
        raise ValueError(f"{source.name} requires an ADS-B source")

    if not source.uses_adsb_source and adsb_src is not None:
        raise ValueError(f"{source.name} does not use an ADS-B source")

    path = (
        OUTPUT_DIR
        / "data"
        / "flights"
        / f"source={source.name}"
    )

    if adsb_src is not None:
        path /= f"adsb_source={adsb_src.value}"

    return (
        path
        / f"version={source.version}"
        / f"year={dt.year}"
        / f"month={dt.month:02d}"
        / f"day={dt.day:02d}"
        / "data.parquet"
    )


def get_flights(
    source: FlightSource,
    dt: date,
    adsb_src: ADSBSource | None = None,
    use_cache: bool = True,
) -> pl.DataFrame:
    path = flight_path(source, dt, adsb_src)
    if use_cache and path.exists():
        return pl.read_parquet(path)
    
    if isinstance(source, ADSBFlightSource):
        assert adsb_src is not None, f"{source.name} requires an ADS-B source"
        df_adsb = read_adsb(dt, source=adsb_src) # may need buffer
        return source.create_flights_for_day(df_adsb)
    elif isinstance(source, NonADSBFlightSource):
        return source.create_flights_for_day(dt)
    else:
        raise ValueError(f"Unknown FlightSource type: {type(source)}")

@dataclass(frozen=True)
class BTSFlightSource(NonADSBFlightSource):
    name: str = "bts"
    version: int = 3
    uses_adsb_source: bool = False

    def create_flights_for_day(
        self,
        dt: date,
    ) -> pl.DataFrame:
        from data_engineering.bts.read_bts_ontime_to_flights import (
            get_bts_flights_for_day,
        )
        return get_bts_flights_for_day(dt)



