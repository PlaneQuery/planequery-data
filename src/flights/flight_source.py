from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import polars as pl

from data_engineering.adsb.read_adsb import (
    ADSBSource,
    exclude_invalid_icaos,
    read_adsb,
)
from data_engineering.utils import OUTPUT_DIR


@dataclass(frozen=True)
class FlightSource(ABC):
    name: str
    version: int = 1
    # TODO: Can have missing days range

@dataclass(frozen=True)
class ADSBFlightSource(FlightSource):
    def adsb_icao_batch_size(self) -> int | None:
        """Maximum number of complete aircraft traces to load at once."""
        return None

    def adsb_icaos(self, dt: date) -> Sequence[str] | None:
        return None

    def adsb_columns(self) -> Sequence[str] | None:
        return None

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
        columns = source.adsb_columns()
        batch_size = source.adsb_icao_batch_size()
        source_icaos = source.adsb_icaos(dt)
        if batch_size is None:
            kwargs = {"source": adsb_src}
            if columns is not None:
                kwargs["columns"] = columns
            if source_icaos is not None:
                kwargs["icaos"] = source_icaos
            df_adsb = read_adsb(start, end, **kwargs)
            return source.create_flights_for_day(dt, df_adsb)
        if batch_size <= 0:
            raise ValueError("adsb_icao_batch_size must be positive")

        if source_icaos is None:
            icao_scan = read_adsb(
                start,
                end,
                source=adsb_src,
                columns=["icao"],
                lazy=True,
            )
            if adsb_src is ADSBSource.ADSBLOL:
                icao_scan = icao_scan.filter(exclude_invalid_icaos())
            icaos = (
                icao_scan.select("icao")
                .drop_nulls()
                .unique()
                .sort("icao")
                .collect(engine="streaming")
                .get_column("icao")
                .to_list()
            )
        else:
            icaos = sorted(set(source_icaos))
        batches = [
            icaos[offset : offset + batch_size]
            for offset in range(0, len(icaos), batch_size)
        ]
        if not batches:
            batches = [[]]

        flights = []
        for batch_index, icao_batch in enumerate(batches, 1):
            if len(batches) > 1 and (
                batch_index == 1
                or batch_index % 10 == 0
                or batch_index == len(batches)
            ):
                print(
                    f"Processing {source.name} ADS-B batch "
                    f"{batch_index}/{len(batches)}",
                    flush=True,
                )
            kwargs = {
                "source": adsb_src,
                "icaos": icao_batch,
            }
            if columns is not None:
                kwargs["columns"] = columns
            df_adsb = read_adsb(start, end, **kwargs)
            flights.append(source.create_flights_for_day(dt, df_adsb))
            del df_adsb
        return pl.concat(flights)
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
