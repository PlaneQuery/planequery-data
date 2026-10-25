"""Plane-notify flight detection in the standard flight-source interface."""

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import polars as pl

from airports.catalog import AirportCatalogView
from data_engineering.adsb.read_adsb import ADSBSource
from flights.flight_source import ADSBFlightSource

from .segmentation import INPUT_COLUMNS, UPSTREAM_COMMIT, replay_flights


@dataclass(frozen=True)
class PlaneNotifyFlightSource(ADSBFlightSource):
    name: str = "plane-notify"
    version: int = 1

    def adsb_icao_batch_size(self) -> int:
        return 500

    def adsb_columns(self) -> tuple[str, ...]:
        return INPUT_COLUMNS

    def read_adsb(self, start, end=None, **kwargs):
        frame = super().read_adsb(start, end, **kwargs)
        start = start if isinstance(start, datetime) else datetime.combine(start, datetime.min.time())
        end = end or start + timedelta(days=1)
        # Keep the requested window even when a batch has no positions near
        # its edges. Observation min/max cannot establish data availability.
        return frame.with_columns(
            pl.lit(start).alias("_plane_notify_input_start"),
            pl.lit(end).alias("_plane_notify_input_end"),
        )

    def adsb_read_interval(self, dt: date) -> tuple[datetime, datetime]:
        start = datetime.combine(dt, datetime.min.time())
        return start - timedelta(days=1), start + timedelta(days=2)

    def adsb_read_interval_for_source(
        self, dt: date, adsb_src: ADSBSource
    ) -> tuple[datetime, datetime]:
        if adsb_src is ADSBSource.ADSBX:
            # Public ADSBX samples contain only the first day of each month.
            start = datetime.combine(dt, datetime.min.time())
            return start, start + timedelta(days=1)
        return self.adsb_read_interval(dt)

    def create_flights_for_day(
        self,
        dt: date,
        df_adsb: pl.DataFrame,
        airport_catalog: AirportCatalogView,
    ) -> pl.DataFrame:
        start, end = self.adsb_read_interval(dt)
        day_start = datetime.combine(dt, datetime.min.time())
        if self.target_day_adsb:
            start, end = day_start, day_start + timedelta(days=1)
        elif not df_adsb.is_empty() and "_plane_notify_input_start" in df_adsb.columns:
            start = df_adsb["_plane_notify_input_start"][0]
            end = df_adsb["_plane_notify_input_end"][0]
        return replay_flights(df_adsb, start, end, airport_catalog).filter(
            pl.col("takeoff_time").dt.date() == dt
        )
