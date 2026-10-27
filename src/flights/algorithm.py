import argparse
from dataclasses import dataclass
from datetime import date, datetime
from typing import Sequence

import polars as pl

from airports.catalog import AirportCatalogView
from airports.runtime import airport_catalog_for_date
from data_engineering.adsb.read_adsb import ADSBSource
from flights.flight_manifest import (
    FlightManifest,
    flight_manifest_path,
)
from flights.flight_source import (
    ADSBFlightSource,
    flight_path,
)
from flights.segmentation import algorithm as segmentation
from planequery_data.utils import date_range


@dataclass(frozen=True)
class AlgorithmNearestFlightSource(ADSBFlightSource):
    name: str = "algorithm-nearest"
    version: int = 8

    def adsb_icao_batch_size(self) -> int:
        return 500

    def adsb_columns(self):
        return segmentation.ADSB_ALGORITHM_COLUMNS

    def adsb_read_interval(self, dt: date) -> tuple[datetime, datetime]:
        return segmentation._adsb_read_bounds(dt)

    def adsb_read_interval_for_source(
        self,
        dt: date,
        adsb_src: ADSBSource,
    ) -> tuple[datetime, datetime]:
        return segmentation._adsb_read_bounds(dt, adsb_src)

    def create_flights_for_day(
        self,
        dt: date,
        df_adsb: pl.DataFrame,
        airport_catalog: AirportCatalogView,
    ) -> pl.DataFrame:
        return segmentation.create_flights_from_adsb(
            df_adsb,
            dt,
            airport_catalog,
        )


def create_flights_for_day(
    target_date: date,
    airport_catalog: AirportCatalogView,
    icaos: Sequence[str] | None = None,
    pia_or_american_ladd_only: bool = False,
    sfdps_or_bts_only: bool = False,
    ground_truth_only: bool = False,
    adsb_src: ADSBSource = ADSBSource.ADSBLOL,
    max_workers: int | None = None,
) -> pl.DataFrame:
    if max_workers is None:
        max_workers = segmentation._number_of_workers_default()
    return segmentation.create_flights_for_day(
        target_date=target_date,
        icaos=icaos,
        pia_or_american_ladd_only=pia_or_american_ladd_only,
        sfdps_or_bts_only=sfdps_or_bts_only,
        ground_truth_only=ground_truth_only,
        source=adsb_src,
        max_workers=max_workers,
        airport_catalog=airport_catalog,
    )


def default_max_workers() -> int:
    return segmentation._number_of_workers_default()


def write_flights_for_day(
    target_date: date = date(2026, 3, 1),
    icaos: Sequence[str] | None = None,
    pia_or_american_ladd_only: bool = False,
    sfdps_or_bts_only: bool = False,
    ground_truth_only: bool = False,
    adsb_src: ADSBSource = ADSBSource.ADSBLOL,
    max_workers: int | None = None,
) -> pl.DataFrame:
    adsb_src = ADSBSource(adsb_src)
    airport_catalog = airport_catalog_for_date(target_date)
    flights_df = create_flights_for_day(
        target_date=target_date,
        airport_catalog=airport_catalog,
        icaos=icaos,
        pia_or_american_ladd_only=pia_or_american_ladd_only,
        sfdps_or_bts_only=sfdps_or_bts_only,
        ground_truth_only=ground_truth_only,
        adsb_src=adsb_src,
        max_workers=max_workers,
    )
    output_path = flight_path(AlgorithmNearestFlightSource(), target_date, adsb_src)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flights_df.write_parquet(output_path)
    FlightManifest.from_provenance(
        airport_catalog.provenance,
        adsb_source=adsb_src,
    ).write(flight_manifest_path(output_path))
    return flights_df


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("start_date", type=date.fromisoformat)
    parser.add_argument("end_date", type=date.fromisoformat)
    parser.add_argument(
        "--source",
        type=ADSBSource,
        choices=tuple(ADSBSource),
        default=ADSBSource.ADSBLOL,
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=default_max_workers(),
    )
    parser.add_argument("--pia-or-american-ladd-only", action="store_true")
    parser.add_argument("--sfdps-or-bts-only", action="store_true")
    parser.add_argument("--ground-truth-only", action="store_true")
    parser.add_argument("--icaos", nargs="+")
    args = parser.parse_args()
    assert args.end_date > args.start_date
    assert args.max_workers > 0

    for target_date in date_range(args.start_date, args.end_date):
        write_flights_for_day(
            target_date=target_date,
            icaos=args.icaos,
            pia_or_american_ladd_only=args.pia_or_american_ladd_only,
            sfdps_or_bts_only=args.sfdps_or_bts_only,
            ground_truth_only=args.ground_truth_only,
            adsb_src=args.source,
            max_workers=args.max_workers,
        )


if __name__ == "__main__":
    cli()
