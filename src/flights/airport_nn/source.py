import os
from dataclasses import dataclass
from datetime import date, datetime
from functools import cached_property
from pathlib import Path

import polars as pl

from airports.catalog import AirportCatalogView
from flights.algorithm import AlgorithmNearestFlightSource
from flights.flight_source import ADSBFlightSource


DEFAULT_MODEL_PATH = Path(
    "models/airport-nn-february-2026-10-21-days-20/model.pt"
)
REUSED_FLIGHT_BATCH_SIZE = 2048


@dataclass(frozen=True)
class AlgorithmNNFlightSource(ADSBFlightSource):
    name: str = "algorithm-nn"
    version: int = 10
    model_path: Path = DEFAULT_MODEL_PATH

    def adsb_icao_batch_size(self) -> int:
        return 256

    def adsb_columns(self) -> list[str]:
        from flights.airport_nn.features import COLUMNS
        from flights.segmentation.algorithm import ADSB_ALGORITHM_COLUMNS

        return list(dict.fromkeys([*ADSB_ALGORITHM_COLUMNS, *COLUMNS]))

    def reusable_flight_source(self) -> AlgorithmNearestFlightSource:
        return AlgorithmNearestFlightSource()

    def adsb_columns_for_reused_flights(self) -> list[str]:
        from flights.airport_nn.features import COLUMNS

        return COLUMNS

    def adsb_icao_batch_size_for_reused_flights(self) -> int:
        # Reuse needs fewer columns and does not materialize segmentation objects.
        return REUSED_FLIGHT_BATCH_SIZE

    def initialize_batch_worker(self, worker_count: int) -> None:
        import torch

        from flights.airport_nn.model import load_model

        # Separate processes parallelize Python feature construction. Keep their
        # small model evaluations on CPU so they do not contend for one MPS queue.
        torch.set_num_threads(max(1, (os.cpu_count() or 1) // worker_count))
        self.__dict__["model"] = load_model(self.model_path, device="cpu")

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

    def create_flights_from_reused_flights(
        self,
        dt: date,
        flights: pl.DataFrame,
        df_adsb: pl.DataFrame,
        airport_catalog: AirportCatalogView,
    ) -> pl.DataFrame:
        from flights.airport_nn.inference import predict_flights

        return predict_flights(
            flights,
            df_adsb,
            self.model,
            airport_catalog,
        )
