"""EUROCONTROL labels and off-block matching for European validation only."""

import polars as pl

from flights.airport_nn.data import TYPES, catalog
from flights.flight_source import flight_path, write_flights
from flights.providers.eurocontrol import EurocontrolFlightSource


def references(day):
    source = EurocontrolFlightSource()
    path = flight_path(source, day)
    if not path.exists():
        write_flights(source, day)
    fields = [
        "icao",
        "takeoff_time",
        "landing_time",
        "takeoff_airport_ident",
        "landing_airport_ident",
    ]
    raw = pl.read_parquet(path).select(fields).unique().sort(fields)
    airports = catalog()
    european = [
        k for k, a in airports.items() if a["type"] in TYPES and a["continent"] == "EU"
    ]
    refs = (
        raw.with_row_index("id")
        .with_columns(
            pl.col("icao").str.to_lowercase(),
            (pl.lit(f"{day}:EUROCONTROL:") + pl.col("id").cast(pl.String)).alias(
                "reference_id"
            ),
            pl.lit("EUROCONTROL").alias("source"),
            pl.lit("EUROCONTROL").alias("label_source"),
        )
        .drop("id")
    )
    valid = (
        pl.col("icao").str.contains(r"^[0-9a-f]{6}$")
        & (pl.col("takeoff_time").dt.date() == day)
        & (pl.col("landing_time").dt.date() == day)
        & (pl.col("landing_time") > pl.col("takeoff_time"))
        & pl.any_horizontal(
            pl.col(f"{end}_airport_ident").is_in(european)
            for end in ("takeoff", "landing")
        )
    ).fill_null(False)
    refs = refs.filter(valid)
    for end in ("takeoff", "landing"):
        refs = refs.with_columns(
            pl.when(pl.col(f"{end}_airport_ident").is_in(european))
            .then(pl.col(f"{end}_airport_ident"))
            .otherwise(pl.lit(""))
            .alias(f"{end}_truth"),
        )
    pairs = refs.join(refs, on="icao", suffix="_other").filter(
        (pl.col("reference_id") < pl.col("reference_id_other"))
        & (pl.col("takeoff_time") < pl.col("landing_time_other"))
        & (pl.col("takeoff_time_other") < pl.col("landing_time"))
    )
    conflicts = set(pairs["reference_id"]) | set(pairs["reference_id_other"])
    return (
        refs.with_columns(
            pl.col("reference_id").is_in(conflicts).alias("time_conflict")
        ),
        path,
    )


def match_eurocontrol(segments, refs):
    pairs = segments.join(
        refs.filter(~pl.col("time_conflict")), on="icao", suffix="_ref"
    ).filter(
        (pl.col("takeoff_time") - pl.col("takeoff_time_ref"))
        .dt.total_seconds()
        .is_between(-300, 3600)
        & (
            (pl.col("landing_time") - pl.col("landing_time_ref"))
            .dt.total_seconds()
            .abs()
            <= 900
        )
        & (
            pl.min_horizontal("landing_time", "landing_time_ref")
            > pl.max_horizontal("takeoff_time", "takeoff_time_ref")
        )
    )
    return pairs.filter(
        (pl.len().over("reference_id") == 1) & (pl.len().over("segment") == 1)
    )
