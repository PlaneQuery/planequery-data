"""Assign nearest airports to published OpenSky flights using OpenSky positions."""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import date, timedelta
from operator import itemgetter

import polars as pl

from airports.catalog import AirportCatalogView
from data_engineering.adsb.read_adsb import ADSBSource, read_adsb
from flights.airport_nn.features import clean_messages
from flights.flight_source import NonADSBFlightSource, get_flights_for_catalog
from flights.providers.opensky import OpenSkyTrinoFlightSource


FLIGHT_BATCH_SIZE = 2048
ENDPOINT_WINDOW = timedelta(minutes=1)
COLUMNS = ["icao", "time", "lat", "lon"]


def assign_nearest_airports(
    flights: pl.DataFrame,
    messages: pl.DataFrame,
    airport_catalog: AirportCatalogView,
) -> pl.DataFrame:
    """Replace only airport fields, using one-minute endpoint windows.

    Choose the first/last valid position inside each segment and its one-minute
    endpoint window. Missing endpoint observations produce null airports; no
    distance or altitude threshold restricts the nearest-airport lookup.
    """
    tracks = {
        icao: frame.select("time", "lat", "lon").rows()
        for (icao,), frame in clean_messages(messages.select(COLUMNS)).group_by("icao")
    }
    airport_ids = {name: [] for name in ("takeoff", "landing")}
    airport_idents = {name: [] for name in ("takeoff", "landing")}
    for icao, start, end in flights.select(
        "icao", "takeoff_time", "landing_time"
    ).iter_rows():
        track = tracks.get(icao, [])
        for name, lo, hi in (
            ("takeoff", start, min(end, start + ENDPOINT_WINDOW)),
            ("landing", max(start, end - ENDPOINT_WINDOW), end),
        ):
            index = (
                bisect_left(track, lo, key=itemgetter(0))
                if name == "takeoff"
                else bisect_right(track, hi, key=itemgetter(0)) - 1
            )
            match = None
            if 0 <= index < len(track) and lo <= track[index][0] <= hi:
                _, lat, lon = track[index]
                match = airport_catalog.matcher.nearest(lat, lon)
            airport_ids[name].append(match.id if match is not None else None)
            airport_idents[name].append(match.ident if match is not None else None)
    return flights.with_columns(
        *[
            column
            for name in ("takeoff", "landing")
            for column in (
                pl.Series(f"{name}_airport_id", airport_ids[name], dtype=pl.Int64),
                pl.Series(f"{name}_airport_ident", airport_idents[name], dtype=pl.String),
            )
        ]
    )


@dataclass(frozen=True)
class OpenSkyNearestFlightSource(NonADSBFlightSource):
    name: str = "opensky-nearest"
    version: int = 2

    def enrichment_adsb_source(self) -> ADSBSource:
        return ADSBSource.OPENSKY

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

        parts = []
        for batch in flights.iter_slices(FLIGHT_BATCH_SIZE):
            messages = read_adsb(
                start=batch["takeoff_time"].min(),
                # Reads are end-exclusive; include lastseen and overnight arrivals.
                end=batch["landing_time"].max() + timedelta(milliseconds=1),
                source=ADSBSource.OPENSKY,
                icaos=batch["icao"].drop_nulls().unique().to_list(),
                columns=COLUMNS,
            )
            parts.append(assign_nearest_airports(batch, messages, airport_catalog))
        return pl.concat(parts)
