from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import polars as pl

from data_engineering.adsb.read_adsb import ADSBSource, read_adsb
from data_engineering.utils import OUTPUT_DIR


@dataclass(frozen=True)
class FlightSource(ABC):
    name: str
    version: int = 1


@dataclass(frozen=True)
class ADSBFlightSource(FlightSource):
    def adsb_read_interval(
        self,
        dt: date,
    ) -> tuple[date | datetime, datetime | None]:
        return dt, None

    @abstractmethod
    def create_flights_for_day(
        self,
        dt: date,
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
    assert isinstance(source, ADSBFlightSource) == (adsb_src is not None)

    path = OUTPUT_DIR / "data" / "flights" / f"source={source.name}"

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
        start, end = source.adsb_read_interval(dt)
        df_adsb = read_adsb(start, end, source=adsb_src)
        return source.create_flights_for_day(dt, df_adsb)
    if isinstance(source, NonADSBFlightSource):
        return source.create_flights_for_day(dt)
    raise ValueError(f"Unknown FlightSource type: {type(source)}")


def write_flights(
    source: FlightSource,
    dt: date,
    adsb_src: ADSBSource | None = None,
) -> pl.DataFrame:
    path = flight_path(source, dt, adsb_src)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_flights = get_flights(source, dt, adsb_src, use_cache=False)
    df_flights.write_parquet(path)
    return df_flights
