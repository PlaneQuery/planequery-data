"""Assign neural-network airport identifiers to one day of segmented flights."""

import argparse
from datetime import date
from pathlib import Path

import polars as pl

from airports.runtime import airport_catalog_for_date
from data_engineering.adsb.read_adsb import ADSBSource, read_adsb
from flights.algorithm import AlgorithmNearestFlightSource
from flights.flight_manifest import (
    FlightManifest,
    flight_manifest_path,
)
from flights.airport_nn.features import COLUMNS
from flights.airport_nn.source import (
    DEFAULT_MODEL_PATH,
    REUSED_FLIGHT_BATCH_SIZE,
    AlgorithmNNFlightSource,
)
from flights.flight_source import flight_path, get_flights_for_catalog


def derive_day(
    day: date,
    source: ADSBSource = ADSBSource.ADSBLOL,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    flights: pl.DataFrame | None = None,
    batch_aircraft: int = REUSED_FLIGHT_BATCH_SIZE,
) -> Path:
    source = ADSBSource(source)
    assert batch_aircraft > 0
    airport_catalog = airport_catalog_for_date(day)
    if flights is None:
        flights = get_flights_for_catalog(
            AlgorithmNearestFlightSource(),
            day,
            airport_catalog,
            source,
        )

    nn_source = AlgorithmNNFlightSource(model_path=Path(model_path))
    aircraft = flights.get_column("icao").drop_nulls().unique().sort().to_list()
    parts = []
    for start in range(0, len(aircraft), batch_aircraft):
        icaos = aircraft[start : start + batch_aircraft]
        subset = flights.filter(pl.col("icao").is_in(icaos))
        messages = read_adsb(day, source=source, icaos=icaos, columns=COLUMNS)
        parts.append(
            nn_source.create_flights_from_reused_flights(
                day,
                subset,
                messages,
                airport_catalog,
            )
        )

    result = pl.concat(parts) if parts else flights
    output = flight_path(
        nn_source,
        day,
        source,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output)
    FlightManifest.from_provenance(
        airport_catalog.provenance,
        adsb_source=source,
    ).write(flight_manifest_path(output))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("date", type=date.fromisoformat)
    parser.add_argument(
        "--source",
        type=ADSBSource,
        choices=tuple(ADSBSource),
        default=ADSBSource.ADSBLOL,
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--batch-aircraft",
        type=int,
        default=REUSED_FLIGHT_BATCH_SIZE,
    )
    args = parser.parse_args()
    print(derive_day(args.date, args.source, args.model, batch_aircraft=args.batch_aircraft))
