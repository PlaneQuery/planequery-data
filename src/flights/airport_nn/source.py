from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import polars as pl

from data_engineering.utils import OUTPUT_DIR
from flights.algorithm import AlgorithmNearestFlightSource
from flights.flight_source import ADSBFlightSource


DEFAULT_MODEL_PATH = (
    OUTPUT_DIR
    / "data/models/flights_airport_nn/algorithm-airport-refactored/model.pt"
)


@dataclass(frozen=True)
class AlgorithmNNFlightSource(ADSBFlightSource):
    name: str = "algorithm-nn"
    version: int = 1
    model_path: Path = DEFAULT_MODEL_PATH

    def adsb_columns(self):
        from flights.airport_nn.features import COLUMNS
        from flights.segmentation.algorithm import ADSB_ALGORITHM_COLUMNS

        return list(dict.fromkeys([*ADSB_ALGORITHM_COLUMNS, *COLUMNS]))

    def adsb_read_interval(self, dt: date) -> tuple[datetime, datetime]:
        return AlgorithmNearestFlightSource().adsb_read_interval(dt)

    def create_flights_for_day(
        self,
        dt: date,
        df_adsb: pl.DataFrame,
    ) -> pl.DataFrame:
        from flights.airport_nn.inference import predict_flights
        from flights.airport_nn.model import load_model

        flights = AlgorithmNearestFlightSource().create_flights_for_day(dt, df_adsb)
        return predict_flights(flights, df_adsb, load_model(self.model_path))
