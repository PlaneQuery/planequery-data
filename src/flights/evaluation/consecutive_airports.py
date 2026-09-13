from typing import Protocol

import polars as pl

from airports.airport_lookup import AirportLookup
from planequery_data.utils import haversine


DEFAULT_DISPLACEMENT_THRESHOLD_KM = 50.0
REQUIRED_COLUMNS = {
    "icao",
    "takeoff_time",
    "landing_time",
    "takeoff_airport_ident",
    "landing_airport_ident",
}


class AirportCoordinatesLookup(Protocol):
    def get_airport_coordinates(
        self,
        airport_ident: str,
    ) -> tuple[float, float] | None: ...


def consecutive_airport_transitions(df_flights: pl.DataFrame) -> pl.DataFrame:
    """Return consecutive flight pairs whose airport idents can be compared."""
    missing_columns = REQUIRED_COLUMNS.difference(df_flights.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing columns required for consecutive-airport metrics: {missing}")

    previous_landing_ident = "_previous_landing_airport_ident"
    previous_takeoff_time = "_previous_takeoff_time"
    previous_landing_time = "_previous_landing_time"
    comparable_airports = (
        pl.col(previous_landing_ident).is_not_null()
        & (pl.col(previous_landing_ident).str.len_chars() > 0)
        & pl.col("takeoff_airport_ident").is_not_null()
        & (pl.col("takeoff_airport_ident").str.len_chars() > 0)
    )

    return (
        df_flights
        .sort(["icao", "takeoff_time", "landing_time"])
        .with_columns(
            pl.col("takeoff_time")
            .shift(1)
            .over("icao")
            .alias(previous_takeoff_time),
            pl.col("landing_airport_ident")
            .shift(1)
            .over("icao")
            .alias(previous_landing_ident),
            pl.col("landing_time")
            .shift(1)
            .over("icao")
            .alias(previous_landing_time),
        )
        .filter(comparable_airports)
        .select(
            "icao",
            pl.col(previous_takeoff_time).alias("previous_takeoff_time"),
            pl.col(previous_landing_time).alias("previous_landing_time"),
            "takeoff_time",
            pl.col(previous_landing_ident).alias(
                "previous_landing_airport_ident"
            ),
            "takeoff_airport_ident",
            (
                pl.col(previous_landing_ident) != pl.col("takeoff_airport_ident")
            ).alias("consecutive_airport_mismatch"),
        )
    )


def consecutive_airport_details(
    df_flights: pl.DataFrame,
    *,
    airport_lookup: AirportCoordinatesLookup | None = None,
) -> pl.DataFrame:
    """Return every comparable transition with its airport displacement."""
    transitions = consecutive_airport_transitions(df_flights)
    return _add_displacement(transitions, airport_lookup=airport_lookup)


def consecutive_airport_mismatches(
    df_flights: pl.DataFrame,
    *,
    airport_lookup: AirportCoordinatesLookup | None = None,
) -> pl.DataFrame:
    """Return mismatched consecutive airports and their great-circle distance."""
    return consecutive_airport_details(
        df_flights,
        airport_lookup=airport_lookup,
    ).filter(
        pl.col("consecutive_airport_mismatch")
    )


def _add_displacement(
    mismatches: pl.DataFrame,
    *,
    airport_lookup: AirportCoordinatesLookup | None,
) -> pl.DataFrame:
    if mismatches.is_empty():
        return mismatches.with_columns(
            pl.lit(None, dtype=pl.Float64).alias(
                "consecutive_airport_displacement_km"
            )
        )

    if airport_lookup is None:
        airport_lookup = AirportLookup()

    def displacement_km(row: dict[str, str | bool]) -> float | None:
        if not row["consecutive_airport_mismatch"]:
            return 0.0
        previous_coords = airport_lookup.get_airport_coordinates(
            row["previous_landing_airport_ident"]
        )
        takeoff_coords = airport_lookup.get_airport_coordinates(
            row["takeoff_airport_ident"]
        )
        if previous_coords is None or takeoff_coords is None:
            return None
        return haversine(*previous_coords, *takeoff_coords)

    return mismatches.with_columns(
        pl.struct(
            "previous_landing_airport_ident",
            "takeoff_airport_ident",
            "consecutive_airport_mismatch",
        )
        .map_elements(displacement_km, return_dtype=pl.Float64)
        .alias("consecutive_airport_displacement_km")
    )


def consecutive_airport_stats(
    df_flights: pl.DataFrame,
    *,
    displacement_threshold_km: float = DEFAULT_DISPLACEMENT_THRESHOLD_KM,
    airport_lookup: AirportCoordinatesLookup | None = None,
) -> dict[str, int | float | None]:
    """Calculate airport-continuity metrics for consecutive flights per ICAO."""
    if displacement_threshold_km < 0:
        raise ValueError("displacement_threshold_km must be non-negative")

    transitions = consecutive_airport_details(
        df_flights,
        airport_lookup=airport_lookup,
    )
    transition_count = transitions.height
    mismatch_count = transitions.filter(
        pl.col("consecutive_airport_mismatch")
    ).height
    mismatches = transitions.filter(pl.col("consecutive_airport_mismatch"))
    displacements = mismatches.get_column(
        "consecutive_airport_displacement_km"
    ).drop_nulls()
    over_threshold_count = int(
        (displacements > displacement_threshold_km).sum()
    )

    threshold_label = f"{displacement_threshold_km:g}"
    stats: dict[str, int | float | None] = {
        "consecutive_airport_comparison_count": transition_count,
        "consecutive_airport_mismatch_count": mismatch_count,
        "consecutive_airport_mismatch_rate": (
            mismatch_count / transition_count if transition_count else 0.0
        ),
        "consecutive_airport_displacement_km_p50": displacements.quantile(
            0.50, interpolation="linear"
        ),
        "consecutive_airport_displacement_km_p90": displacements.quantile(
            0.90, interpolation="linear"
        ),
        "consecutive_airport_displacement_km_p95": displacements.quantile(
            0.95, interpolation="linear"
        ),
        "consecutive_airport_displacement_km_p99": displacements.quantile(
            0.99, interpolation="linear"
        ),
        "consecutive_airport_displacement_km_max": displacements.max(),
        (
            "consecutive_airport_displacement_over_"
            f"{threshold_label}km_count"
        ): over_threshold_count,
        (
            "consecutive_airport_displacement_over_"
            f"{threshold_label}km_rate"
        ): (
            over_threshold_count / transition_count if transition_count else 0.0
        ),
    }
    return stats
