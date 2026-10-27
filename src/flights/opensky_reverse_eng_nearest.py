"""Nearest-airport comparison using unchanged OpenSky reconstruction segments."""

import math
from dataclasses import dataclass
from datetime import date

import polars as pl

from airports.catalog import AirportCatalogView
from flights.opensky_reverse_eng import OpenSkyReverseEngFlightSource


def assign_nearest_airports(
    flights: pl.DataFrame,
    airport_catalog: AirportCatalogView,
) -> pl.DataFrame:
    """Replace only airport fields using the first/last retained coordinates.

    The shared matcher contains fixed-wing airports from the dated catalog.
    Neither distance, altitude, nor GPS-code/elevation availability restricts
    assignment. Segmentation, including its airport-based closure rule, has
    already completed before this function runs.
    """
    if flights.is_empty():
        return flights.clone()
    columns = []
    for endpoint, position in (("takeoff", "first"), ("landing", "last")):
        ids = []
        identifiers = []
        for lat, lon in flights.select(
            f"{position}_lat", f"{position}_lon"
        ).iter_rows():
            if (
                lat is None
                or lon is None
                or not math.isfinite(lat)
                or not math.isfinite(lon)
            ):
                ids.append(None)
                identifiers.append(None)
                continue
            match = airport_catalog.matcher.nearest(lat, lon)
            ids.append(match.id)
            identifiers.append(match.ident)
        columns.extend(
            [
                pl.Series(f"{endpoint}_airport_id", ids, dtype=pl.Int64),
                pl.Series(f"{endpoint}_airport_ident", identifiers, dtype=pl.String),
            ]
        )
    return flights.with_columns(columns)


@dataclass(frozen=True)
class OpenSkyReverseEngNearestFlightSource(OpenSkyReverseEngFlightSource):
    """Reuse the original segments and change only endpoint airport assignment.

    Inherit the segmentation version so changes to the parent source also
    invalidate this source's flight and evaluation caches.
    """

    name: str = "opensky-reverse-eng-nearest"

    def reusable_flight_source(self) -> OpenSkyReverseEngFlightSource:
        return OpenSkyReverseEngFlightSource()

    def adsb_columns_for_reused_flights(self) -> tuple[str, ...]:
        # Original flight records already contain both endpoint coordinates.
        return ()

    def create_flights_for_day(
        self, dt: date, df_adsb: pl.DataFrame, airport_catalog: AirportCatalogView
    ) -> pl.DataFrame:
        flights = super().create_flights_for_day(dt, df_adsb, airport_catalog)
        return assign_nearest_airports(flights, airport_catalog)

    def create_flights_from_reused_flights(
        self,
        dt: date,
        flights: pl.DataFrame,
        df_adsb: pl.DataFrame,
        airport_catalog: AirportCatalogView,
    ) -> pl.DataFrame:
        return assign_nearest_airports(flights, airport_catalog)
