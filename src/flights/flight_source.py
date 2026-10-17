from __future__ import annotations

import multiprocessing
from abc import ABC, abstractmethod
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path

import polars as pl

from airports.catalog import AirportCatalogView
from airports.models import AirportCatalogProvenance
from airports.runtime import airport_catalog_for_date
from data_engineering.adsb.read_adsb import (
    ADSBSource,
    exclude_invalid_icaos,
    read_adsb,
)
from data_engineering.utils import OUTPUT_DIR
from flights.flight_manifest import (
    FlightManifest,
    flight_manifest_path,
)
from planequery_data.utils import date_range


@dataclass(frozen=True)
class FlightSource(ABC):
    name: str
    version: int = 1
    # TODO: Can have missing days range

    def flight_manifest(
        self,
        provenance: AirportCatalogProvenance,
        adsb_src: ADSBSource | None,
    ) -> FlightManifest:
        return FlightManifest.from_provenance(provenance, adsb_source=adsb_src)


@dataclass(frozen=True)
class ADSBFlightSource(FlightSource):
    def read_adsb(
        self,
        start: date | datetime,
        end: datetime | None = None,
        **kwargs,
    ) -> pl.DataFrame | pl.LazyFrame:
        """Read source input; override when canonical ADS-B loses required data."""
        return read_adsb(start, end, **kwargs)

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

    def adsb_read_interval_for_source(
        self,
        dt: date,
        adsb_src: ADSBSource,
    ) -> tuple[date | datetime, datetime | None]:
        """Return the provider-specific interval needed to build one flight day."""
        reusable_source = self.reusable_flight_source()
        if reusable_source is not None:
            return reusable_source.adsb_read_interval_for_source(dt, adsb_src)
        return self.adsb_read_interval(dt)

    def reusable_flight_source(self) -> ADSBFlightSource | None:
        """Flights from the same ADS-B provider that can replace segmentation."""
        return None

    def adsb_columns_for_reused_flights(self) -> Sequence[str] | None:
        """Columns needed in addition to cached flights; empty skips ADS-B reads."""
        return self.adsb_columns()

    def adsb_icao_batch_size_for_reused_flights(self) -> int | None:
        return self.adsb_icao_batch_size()

    def initialize_batch_worker(self, worker_count: int) -> None:
        """Configure process-local resources before processing ADS-B batches."""
        return None

    def create_flights_from_reused_flights(
        self,
        dt: date,
        flights: pl.DataFrame,
        df_adsb: pl.DataFrame,
        airport_catalog: AirportCatalogView,
    ) -> pl.DataFrame:
        raise NotImplementedError(
            f"{type(self).__name__} does not support reusable flights"
        )

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


@dataclass(frozen=True)
class _ADSBBatchTask:
    index: int
    icaos: tuple[str, ...]
    reused_flights: pl.DataFrame | None = None


_BATCH_WORKER_SOURCE: ADSBFlightSource | None = None
_BATCH_WORKER_DATE: date | None = None
_BATCH_WORKER_CATALOG: AirportCatalogView | None = None
_BATCH_WORKER_ADSB_SOURCE: ADSBSource | None = None


def _initialize_adsb_batch_worker(
    source: ADSBFlightSource,
    dt: date,
    expected_provenance: AirportCatalogProvenance,
    adsb_src: ADSBSource,
    worker_count: int,
) -> None:
    global _BATCH_WORKER_SOURCE
    global _BATCH_WORKER_DATE
    global _BATCH_WORKER_CATALOG
    global _BATCH_WORKER_ADSB_SOURCE

    airport_catalog = airport_catalog_for_date(dt)
    if airport_catalog.provenance != expected_provenance:
        raise RuntimeError("Batch worker loaded a different airport catalog")
    source.initialize_batch_worker(worker_count)
    _BATCH_WORKER_SOURCE = source
    _BATCH_WORKER_DATE = dt
    _BATCH_WORKER_CATALOG = airport_catalog
    _BATCH_WORKER_ADSB_SOURCE = adsb_src


def _create_adsb_batch(
    source: ADSBFlightSource,
    dt: date,
    airport_catalog: AirportCatalogView,
    adsb_src: ADSBSource,
    task: _ADSBBatchTask,
) -> tuple[int, pl.DataFrame]:
    reused = task.reused_flights is not None
    columns = (
        source.adsb_columns_for_reused_flights()
        if reused
        else source.adsb_columns()
    )
    start, end = source.adsb_read_interval_for_source(dt, adsb_src)
    kwargs = {
        "source": adsb_src,
        "icaos": list(task.icaos),
    }
    if columns is not None:
        kwargs["columns"] = columns
    if reused and columns is not None and len(columns) == 0:
        df_adsb = pl.DataFrame()
    else:
        df_adsb = source.read_adsb(start, end, **kwargs)
    if reused:
        assert task.reused_flights is not None
        flights = source.create_flights_from_reused_flights(
            dt,
            task.reused_flights,
            df_adsb,
            airport_catalog,
        )
    else:
        flights = source.create_flights_for_day(dt, df_adsb, airport_catalog)
    return task.index, flights


def _create_adsb_batch_in_worker(
    task: _ADSBBatchTask,
) -> tuple[int, pl.DataFrame]:
    assert _BATCH_WORKER_SOURCE is not None
    assert _BATCH_WORKER_DATE is not None
    assert _BATCH_WORKER_CATALOG is not None
    assert _BATCH_WORKER_ADSB_SOURCE is not None
    return _create_adsb_batch(
        _BATCH_WORKER_SOURCE,
        _BATCH_WORKER_DATE,
        _BATCH_WORKER_CATALOG,
        _BATCH_WORKER_ADSB_SOURCE,
        task,
    )


def _create_adsb_batches(
    source: ADSBFlightSource,
    dt: date,
    airport_catalog: AirportCatalogView,
    adsb_src: ADSBSource,
    tasks: Sequence[_ADSBBatchTask],
    batch_workers: int,
) -> pl.DataFrame:
    worker_count = min(batch_workers, len(tasks))
    if worker_count == 1:
        indexed_parts = [
            _create_adsb_batch(source, dt, airport_catalog, adsb_src, task)
            for task in tasks
        ]
    else:
        print(
            f"Processing {len(tasks)} {source.name} ADS-B batches with "
            f"{worker_count} within-day workers",
            flush=True,
        )
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_adsb_batch_worker,
            initargs=(
                replace(source),
                dt,
                airport_catalog.provenance,
                adsb_src,
                worker_count,
            ),
        ) as executor:
            indexed_parts = list(
                executor.map(_create_adsb_batch_in_worker, tasks)
            )

    indexed_parts.sort(key=lambda item: item[0])
    return pl.concat([part for _, part in indexed_parts])


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


def _read_cached_flights_for_catalog(
    source: FlightSource,
    dt: date,
    airport_catalog: AirportCatalogView,
    adsb_src: ADSBSource | None,
) -> pl.DataFrame | None:
    path = flight_path(source, dt, adsb_src)
    manifest_path = flight_manifest_path(path)
    if not path.is_file() or not manifest_path.is_file():
        return None

    current_manifest = source.flight_manifest(airport_catalog.provenance, adsb_src)
    cached_manifest = FlightManifest.read(manifest_path)
    if cached_manifest.flight_cache_key != current_manifest.flight_cache_key:
        return None
    return pl.read_parquet(path)


def _reusable_flights_for_source(
    source: ADSBFlightSource,
    dt: date,
    airport_catalog: AirportCatalogView,
    adsb_src: ADSBSource,
) -> pl.DataFrame | None:
    reusable_source = source.reusable_flight_source()
    if reusable_source is None:
        return None

    flights = _read_cached_flights_for_catalog(
        reusable_source,
        dt,
        airport_catalog,
        adsb_src,
    )
    if flights is not None:
        print(
            f"Reusing cached {reusable_source.name} flights for {source.name} "
            f"on {dt}",
            flush=True,
        )
    return flights


def _create_flights_from_reused_flights(
    source: ADSBFlightSource,
    dt: date,
    reusable_flights: pl.DataFrame,
    airport_catalog: AirportCatalogView,
    adsb_src: ADSBSource,
    batch_workers: int,
) -> pl.DataFrame:
    batch_size = source.adsb_icao_batch_size_for_reused_flights()
    if batch_size is None:
        batch_size = max(1, reusable_flights.height)
    if batch_size <= 0:
        raise ValueError("adsb_icao_batch_size must be positive")

    icaos = (
        reusable_flights.get_column("icao")
        .drop_nulls()
        .unique()
        .sort()
        .to_list()
    )
    batches = [
        icaos[offset : offset + batch_size]
        for offset in range(0, len(icaos), batch_size)
    ]
    if not batches:
        batches = [[]]

    tasks = [
        _ADSBBatchTask(
            index=batch_index,
            icaos=tuple(icao_batch),
            reused_flights=reusable_flights.filter(
                pl.col("icao").is_in(icao_batch)
            ),
        )
        for batch_index, icao_batch in enumerate(batches)
    ]
    return _create_adsb_batches(
        source,
        dt,
        airport_catalog,
        adsb_src,
        tasks,
        batch_workers,
    )


def _create_flights_for_catalog(
    source: FlightSource,
    dt: date,
    airport_catalog: AirportCatalogView,
    adsb_src: ADSBSource | None = None,
    batch_workers: int = 1,
) -> pl.DataFrame:
    if batch_workers < 1:
        raise ValueError("batch_workers must be positive")

    if isinstance(source, ADSBFlightSource):
        assert adsb_src is not None, f"{source.name} requires an ADS-B source"
        reusable_flights = _reusable_flights_for_source(
            source,
            dt,
            airport_catalog,
            adsb_src,
        )
        if reusable_flights is not None:
            return _create_flights_from_reused_flights(
                source,
                dt,
                reusable_flights,
                airport_catalog,
                adsb_src,
                batch_workers,
            )
        start, end = source.adsb_read_interval_for_source(dt, adsb_src)
        columns = source.adsb_columns()
        batch_size = source.adsb_icao_batch_size()
        source_icaos = source.adsb_icaos(dt, airport_catalog)
        if batch_size is None:
            kwargs = {"source": adsb_src}
            if columns is not None:
                kwargs["columns"] = columns
            if source_icaos is not None:
                kwargs["icaos"] = source_icaos
            df_adsb = source.read_adsb(start, end, **kwargs)
            return source.create_flights_for_day(dt, df_adsb, airport_catalog)
        if batch_size <= 0:
            raise ValueError("adsb_icao_batch_size must be positive")

        if source_icaos is None:
            icao_scan = source.read_adsb(
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
        assert icaos, (
            f"No ADS-B ICAOs found for {source.name} on {dt} "
            f"from {adsb_src.value}"
        )
        batches = [
            icaos[offset : offset + batch_size]
            for offset in range(0, len(icaos), batch_size)
        ]

        tasks = [
            _ADSBBatchTask(batch_index, tuple(icao_batch))
            for batch_index, icao_batch in enumerate(batches)
        ]
        return _create_adsb_batches(
            source,
            dt,
            airport_catalog,
            adsb_src,
            tasks,
            batch_workers,
        )
    if isinstance(source, NonADSBFlightSource):
        return source.create_flights_for_day(dt, airport_catalog)
    raise ValueError(f"Unknown FlightSource type: {type(source)}")


def get_flights_for_catalog(
    source: FlightSource,
    dt: date,
    airport_catalog: AirportCatalogView,
    adsb_src: ADSBSource | None = None,
    use_cache: bool = True,
    batch_workers: int = 1,
) -> pl.DataFrame:
    """Read a valid cache or build and persist a flight day.

    ``use_cache=False`` skips this source's cache and does not persist the result.
    """
    if use_cache:
        cached_flights = _read_cached_flights_for_catalog(
            source,
            dt,
            airport_catalog,
            adsb_src,
        )
        if cached_flights is not None:
            return cached_flights
        return write_flights_for_catalog(
            source,
            dt,
            airport_catalog,
            adsb_src,
            batch_workers,
        )
    return _create_flights_for_catalog(
        source,
        dt,
        airport_catalog,
        adsb_src,
        batch_workers,
    )


def get_flights(
    source: FlightSource,
    dt: date,
    adsb_src: ADSBSource | None = None,
    use_cache: bool = True,
    batch_workers: int = 1,
) -> pl.DataFrame:
    airport_catalog = airport_catalog_for_date(dt)
    return get_flights_for_catalog(
        source,
        dt,
        airport_catalog,
        adsb_src,
        use_cache,
        batch_workers,
    )


def get_flights_range(
    source: FlightSource,
    start_date: date,
    end_date: date,
    adsb_src: ADSBSource | None = None,
    use_cache: bool = True,
    batch_workers: int = 1,
) -> pl.DataFrame:
    """Get flights for each day in ``[start_date, end_date)``."""
    return pl.concat([
        get_flights(
            source,
            target_date,
            adsb_src,
            use_cache,
            batch_workers,
        )
        for target_date in date_range(start_date, end_date)
    ])


def write_flights(
    source: FlightSource,
    dt: date,
    adsb_src: ADSBSource | None = None,
    batch_workers: int = 1,
) -> pl.DataFrame:
    airport_catalog = airport_catalog_for_date(dt)
    return write_flights_for_catalog(
        source,
        dt,
        airport_catalog,
        adsb_src,
        batch_workers,
    )


def write_flights_for_catalog(
    source: FlightSource,
    dt: date,
    airport_catalog: AirportCatalogView,
    adsb_src: ADSBSource | None = None,
    batch_workers: int = 1,
) -> pl.DataFrame:
    """Build and persist a flight day using exactly ``airport_catalog``."""
    path = flight_path(source, dt, adsb_src)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_flights = _create_flights_for_catalog(
        source,
        dt,
        airport_catalog,
        adsb_src,
        batch_workers,
    )
    df_flights.write_parquet(path)
    source.flight_manifest(airport_catalog.provenance, adsb_src).write(
        flight_manifest_path(path)
    )
    return df_flights
