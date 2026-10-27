"""EUROCONTROL callsign enrichment from locally cached ADS-B observations."""

import logging
from datetime import timedelta

import polars as pl

from aircraft.callsign import is_valid_callsign
from data_engineering.adsb.canonical import (
    ADSBSource,
    adsb_day_is_complete,
    coerce_adsb_source,
    scan_adsb_parquet,
)


LOGGER = logging.getLogger(__name__)
MIN_CALLSIGN_MINUTES = 3
MIN_CALLSIGN_SHARE = 0.8


def assign_callsigns(
    flights: pl.DataFrame, observations: pl.LazyFrame
) -> pl.DataFrame:
    """Assign a dominant observed callsign within each [takeoff, landing) window.

    Each distinct callsign gets at most one vote per minute per flight. Require
    three supporting minutes and at least 80% of valid votes. No observations
    are filled across gaps or between flights. Unresolved callsigns stay empty.
    """
    indexed = flights.with_row_index("_flight_index")
    output_columns = list(flights.columns)
    if "callsign" not in output_columns:
        output_columns.append("callsign")
    windows = indexed.lazy().select(
        "_flight_index", "takeoff_time", "landing_time",
        pl.col("icao").str.to_lowercase(),
    )
    observations = (
        observations.select(
            pl.col("icao").str.to_lowercase(),
            "time",
            pl.col("callsign").str.to_uppercase().str.replace_all(r"\s+", ""),
        )
        .filter(is_valid_callsign().fill_null(False))
    )
    votes = (
        windows.join(observations, on="icao", how="inner")
        .filter(
            (pl.col("time") >= pl.col("takeoff_time"))
            & (pl.col("time") < pl.col("landing_time"))
        )
        .select("_flight_index", "callsign", pl.col("time").dt.truncate("1m"))
        .unique()
        .group_by("_flight_index", "callsign")
        .agg(pl.len().alias("_votes"))
        .with_columns(pl.col("_votes").sum().over("_flight_index").alias("_total"))
        .filter(
            (pl.col("_votes") >= MIN_CALLSIGN_MINUTES)
            & (pl.col("_votes") / pl.col("_total") >= MIN_CALLSIGN_SHARE)
        )
        .select("_flight_index", "callsign")
        .collect(engine="streaming")
    )
    return (
        indexed.drop("callsign", strict=False)
        .join(votes, on="_flight_index", how="left", validate="1:1")
        .sort("_flight_index")
        .drop("_flight_index")
        .with_columns(pl.col("callsign").fill_null(""))
        .select(output_columns)
    )


def enrich_eurocontrol_callsigns(
    flights: pl.DataFrame, source: ADSBSource
) -> pl.DataFrame:
    """Use complete local ADS-B day partitions, without triggering downloads."""
    source = coerce_adsb_source(source)
    windows = flights.filter(
        pl.col("icao").is_not_null()
        & (pl.col("icao") != "")
        & (pl.col("landing_time") > pl.col("takeoff_time"))
    )
    if windows.is_empty():
        return flights.with_columns(pl.lit("").alias("callsign"))

    start = windows["takeoff_time"].min()
    end = windows["landing_time"].max()
    last_day = (end - timedelta(microseconds=1)).date()
    days = [
        start.date() + timedelta(days=offset)
        for offset in range((last_day - start.date()).days + 1)
    ]
    available_days = [day for day in days if adsb_day_is_complete(source, day)]
    missing_days = sorted(set(days) - set(available_days))
    if missing_days:
        LOGGER.info("Skipping unavailable %s callsign partitions: %s", source.value, missing_days)
    observations = scan_adsb_parquet(
        source, available_days, ["icao", "time", "callsign"],
        icaos=windows["icao"].str.to_lowercase().unique().to_list(),
    ).filter((pl.col("time") >= start) & (pl.col("time") < end))
    return assign_callsigns(flights, observations)
