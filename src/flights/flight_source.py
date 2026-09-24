import hashlib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

import polars as pl

from airports.catalog import AirportCatalogView
from airports.runtime import airport_catalog_for_date
from data_engineering.adsb.read_adsb import (
    ADSBSource,
    exclude_invalid_icaos,
    read_adsb,
)
from data_engineering.utils import OUTPUT_DIR
from flights.airport_catalog_manifest import (
    AirportCatalogManifest,
    airport_catalog_manifest_path,
)


@dataclass(frozen=True)
class FlightSource(ABC):
    name: str
    version: int = 1
    # TODO: Can have missing days range

    def artifact_cache_key(self) -> str | None:
        """Return the immutable source-input identity used in cache paths."""
        return None


@lru_cache(maxsize=None)
def _file_sha256(path: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_artifact_cache_key(label: str, path: str | Path) -> str:
    path = Path(path).resolve()
    stat = path.stat()
    digest = _file_sha256(str(path), stat.st_size, stat.st_mtime_ns)
    return f"{label}-sha256-{digest}"


@dataclass(frozen=True)
class ADSBFlightSource(FlightSource):
    def adsb_icao_batch_size(self) -> int | None:
        """Maximum number of complete aircraft traces to load at once."""
        return None

    def adsb_icaos(
        self,
        dt: date,
        airport_catalog: AirportCatalogView,
    ) -> Sequence[str] | None:
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
        airport_catalog: AirportCatalogView,
    ) -> pl.DataFrame:
        ...


@dataclass(frozen=True)
class NonADSBFlightSource(FlightSource):
    @abstractmethod
    def create_flights_for_day(
        self,
        dt: date,
        airport_catalog: AirportCatalogView,
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

    artifact_cache_key = source.artifact_cache_key()
    if artifact_cache_key is not None:
        path /= f"artifact={artifact_cache_key}"

    return (
        path
        / f"version={source.version}"
        / f"year={dt.year}"
        / f"month={dt.month:02d}"
        / f"day={dt.day:02d}"
        / "data.parquet"
    )


def _get_flights_for_catalog_at_path(
    source: FlightSource,
    dt: date,
    airport_catalog: AirportCatalogView,
    path: Path,
    adsb_src: ADSBSource | None = None,
    use_cache: bool = True,
) -> pl.DataFrame:
    if use_cache and path.exists():
        current_manifest = AirportCatalogManifest.from_provenance(
            airport_catalog.provenance
        )
        cached_manifest = AirportCatalogManifest.read(
            airport_catalog_manifest_path(path)
        )
        if cached_manifest.catalog_cache_key == current_manifest.catalog_cache_key:
            return pl.read_parquet(path)

    if isinstance(source, ADSBFlightSource):
        assert adsb_src is not None, f"{source.name} requires an ADS-B source"
        start, end = source.adsb_read_interval(dt)
        columns = source.adsb_columns()
        batch_size = source.adsb_icao_batch_size()
        source_icaos = source.adsb_icaos(dt, airport_catalog)
        if batch_size is None:
            kwargs = {"source": adsb_src}
            if columns is not None:
                kwargs["columns"] = columns
            if source_icaos is not None:
                kwargs["icaos"] = source_icaos
            df_adsb = read_adsb(start, end, **kwargs)
            return source.create_flights_for_day(dt, df_adsb, airport_catalog)
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
            flights.append(
                source.create_flights_for_day(dt, df_adsb, airport_catalog)
            )
            del df_adsb
        return pl.concat(flights)
    if isinstance(source, NonADSBFlightSource):
        return source.create_flights_for_day(dt, airport_catalog)
    raise ValueError(f"Unknown FlightSource type: {type(source)}")


def get_flights_for_catalog(
    source: FlightSource,
    dt: date,
    airport_catalog: AirportCatalogView,
    adsb_src: ADSBSource | None = None,
    use_cache: bool = True,
) -> pl.DataFrame:
    """Read or build a flight day using exactly ``airport_catalog``."""
    return _get_flights_for_catalog_at_path(
        source,
        dt,
        airport_catalog,
        flight_path(source, dt, adsb_src),
        adsb_src,
        use_cache,
    )


def get_flights(
    source: FlightSource,
    dt: date,
    adsb_src: ADSBSource | None = None,
    use_cache: bool = True,
) -> pl.DataFrame:
    airport_catalog = airport_catalog_for_date(dt)
    return get_flights_for_catalog(
        source,
        dt,
        airport_catalog,
        adsb_src,
        use_cache,
    )


def write_flights(
    source: FlightSource,
    dt: date,
    adsb_src: ADSBSource | None = None,
) -> pl.DataFrame:
    airport_catalog = airport_catalog_for_date(dt)
    return write_flights_for_catalog(
        source,
        dt,
        airport_catalog,
        adsb_src,
    )


def write_flights_for_catalog(
    source: FlightSource,
    dt: date,
    airport_catalog: AirportCatalogView,
    adsb_src: ADSBSource | None = None,
) -> pl.DataFrame:
    """Build and persist a flight day using exactly ``airport_catalog``."""
    path = flight_path(source, dt, adsb_src)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_flights = _get_flights_for_catalog_at_path(
        source,
        dt,
        airport_catalog,
        path,
        adsb_src,
        use_cache=False,
    )
    df_flights.write_parquet(path)
    AirportCatalogManifest.from_provenance(
        airport_catalog.provenance
    ).write(airport_catalog_manifest_path(path))
    return df_flights
