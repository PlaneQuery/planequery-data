import argparse
import os
from dataclasses import dataclass
from datetime import date, datetime
from typing import Sequence

import polars as pl

from data_engineering.adsb.read_adsb import ADSBSource
from flights.flight_source import (
    ADSBFlightSource,
    flight_path,
    get_flights as get_source_flights,
)
from flights.segmentation import algorithm as segmentation
from planequery_data.utils import date_range


@dataclass(frozen=True)
class AlgorithmNearestFlightSource(ADSBFlightSource):
    name: str = "algorithm-nearest"
    version: int = 3

    def adsb_icao_batch_size(self) -> int:
        return 500

    def adsb_columns(self):
        return segmentation.ADSB_ALGORITHM_COLUMNS

    def adsb_read_interval(self, dt: date) -> tuple[datetime, datetime]:
        return segmentation._day_bounds(dt)

    def create_flights_for_day(
        self,
        dt: date,
        df_adsb: pl.DataFrame,
    ) -> pl.DataFrame:
        return segmentation.create_flights_from_adsb(df_adsb, dt)


def get_flights_range(
    start_date: date,
    end_date: date,
    adsb_src: ADSBSource = ADSBSource.ADSBLOL,
) -> pl.DataFrame:
    assert end_date > start_date
    return pl.concat([
        get_source_flights(
            AlgorithmNearestFlightSource(),
            target_date,
            adsb_src,
        )
        for target_date in date_range(start_date, end_date)
    ])


def create_flights_for_day(
    target_date: date,
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
    )


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
    flights_df = create_flights_for_day(
        target_date=target_date,
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
        default=max(1, (os.cpu_count() or 1) // 2),
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
