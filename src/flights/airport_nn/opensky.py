"""Assign NN airports to published OpenSky flights using OpenSky observations."""

from dataclasses import dataclass
from datetime import date, timedelta
from functools import cached_property
from pathlib import Path

import polars as pl

from airports.catalog import AirportCatalogView
from data_engineering.adsb.read_adsb import ADSBSource, read_adsb
from flights.airport_nn.source import DEFAULT_MODEL_PATH
from flights.flight_source import NonADSBFlightSource, get_flights_for_catalog
from flights.providers.opensky import OpenSkyTrinoFlightSource


FLIGHT_BATCH_SIZE = 2048


@dataclass(frozen=True)
class OpenSkyNNFlightSource(NonADSBFlightSource):
    name: str = "opensky-nn"
    version: int = 5
    model_path: Path = DEFAULT_MODEL_PATH

    def enrichment_adsb_source(self) -> ADSBSource:
        return ADSBSource.OPENSKY

    @cached_property
    def model(self):
        from flights.airport_nn.model import load_model

        return load_model(self.model_path)

    def create_flights_for_day(
        self,
        dt: date,
        airport_catalog: AirportCatalogView,
    ) -> pl.DataFrame:
        flights = get_flights_for_catalog(
            OpenSkyTrinoFlightSource(), dt, airport_catalog
        )
        if flights.is_empty():
            return flights

        from flights.airport_nn.features import COLUMNS
        from flights.airport_nn.inference import predict_flights

        model = self.model
        parts = []
        # Batch the published flights, including aircraft with no observations.
        # Slicing preserves their original row order and bounds each ADS-B read.
        for batch in flights.iter_slices(FLIGHT_BATCH_SIZE):
            messages = read_adsb(
                start=batch["takeoff_time"].min(),
                # Canonical timestamps have millisecond precision and reads are
                # end-exclusive. Include lastseen, even on overnight arrivals.
                end=batch["landing_time"].max() + timedelta(milliseconds=1),
                source=ADSBSource.OPENSKY,
                icaos=batch["icao"].drop_nulls().unique().to_list(),
                columns=COLUMNS,
            )
            parts.append(predict_flights(batch, messages, model, airport_catalog))
        return pl.concat(parts)
