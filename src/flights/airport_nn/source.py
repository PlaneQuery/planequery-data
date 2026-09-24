from dataclasses import dataclass
from datetime import date, datetime
from functools import cached_property
from pathlib import Path

import polars as pl

from airports.catalog import AirportCatalogView
from data_engineering.utils import OUTPUT_DIR
from flights.algorithm import AlgorithmNearestFlightSource
from flights.flight_source import ADSBFlightSource, file_artifact_cache_key


DEFAULT_MODEL_PATH = (
    OUTPUT_DIR
    / "data/models/flights_airport_nn/algorithm-airport-refactored/model.pt"
)


@dataclass(frozen=True)
class AlgorithmNNFlightSource(ADSBFlightSource):
    name: str = "algorithm-nn"
    version: int = 3
    model_path: Path = DEFAULT_MODEL_PATH

    def adsb_icao_batch_size(self) -> int:
        return 256

    def artifact_cache_key(self) -> str:
        return file_artifact_cache_key("model", self.model_path)

    def adsb_columns(self):
        from flights.airport_nn.features import COLUMNS
        from flights.segmentation.algorithm import ADSB_ALGORITHM_COLUMNS

        return list(dict.fromkeys([*ADSB_ALGORITHM_COLUMNS, *COLUMNS]))

    def adsb_read_interval(self, dt: date) -> tuple[datetime, datetime]:
        return AlgorithmNearestFlightSource().adsb_read_interval(dt)

    @cached_property
    def model(self):
        from flights.airport_nn.model import load_model

        return load_model(self.model_path)

    def create_flights_for_day(
        self,
        dt: date,
        df_adsb: pl.DataFrame,
        airport_catalog: AirportCatalogView,
    ) -> pl.DataFrame:
        from flights.airport_nn.inference import predict_flights

        flights = AlgorithmNearestFlightSource().create_flights_for_day(
            dt,
            df_adsb,
            airport_catalog,
        )
        return predict_flights(
            flights,
            df_adsb,
            self.model,
            airport_catalog,
        )
