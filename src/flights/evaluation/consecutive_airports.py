import polars as pl

from airports.catalog import AirportCatalogView
from planequery_data.utils import haversine


DEFAULT_DISPLACEMENT_THRESHOLD_KM = 50.0
REQUIRED_COLUMNS = {
    "icao",
    "takeoff_time",
    "landing_time",
    "takeoff_airport_id",
    "landing_airport_id",
    "takeoff_airport_ident",
    "landing_airport_ident",
}


def consecutive_airport_transitions(df_flights: pl.DataFrame) -> pl.DataFrame:
    """Return consecutive flight pairs whose stable airport IDs can be compared."""
    missing_columns = REQUIRED_COLUMNS.difference(df_flights.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing columns required for consecutive-airport metrics: {missing}")

    previous_landing_ident = "_previous_landing_airport_ident"
    previous_landing_id = "_previous_landing_airport_id"
    previous_takeoff_time = "_previous_takeoff_time"
    previous_landing_time = "_previous_landing_time"
    comparable_airports = (
        pl.col(previous_landing_id).is_not_null()
        & pl.col("takeoff_airport_id").is_not_null()
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
            pl.col("landing_airport_id")
            .shift(1)
            .over("icao")
            .alias(previous_landing_id),
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
            pl.col(previous_landing_id).alias("previous_landing_airport_id"),
            "takeoff_airport_id",
            "takeoff_airport_ident",
            (
                pl.col(previous_landing_id) != pl.col("takeoff_airport_id")
            ).alias("consecutive_airport_mismatch"),
        )
    )


def consecutive_airport_details(
    df_flights: pl.DataFrame,
    *,
    airport_catalog: AirportCatalogView,
) -> pl.DataFrame:
    """Return every comparable transition with its airport displacement."""
    transitions = consecutive_airport_transitions(df_flights)
    return _add_displacement(transitions, airport_catalog=airport_catalog)


def consecutive_airport_mismatches(
    df_flights: pl.DataFrame,
    *,
    airport_catalog: AirportCatalogView,
) -> pl.DataFrame:
    """Return mismatched consecutive airports and their great-circle distance."""
    return consecutive_airport_details(
        df_flights,
        airport_catalog=airport_catalog,
    ).filter(
        pl.col("consecutive_airport_mismatch")
    )


def _add_displacement(
    mismatches: pl.DataFrame,
    *,
    airport_catalog: AirportCatalogView,
) -> pl.DataFrame:
    if mismatches.is_empty():
        return mismatches.with_columns(
            pl.lit(None, dtype=pl.Float64).alias(
                "consecutive_airport_displacement_km"
            )
        )

    def displacement_km(row: dict[str, int | bool]) -> float | None:
        if not row["consecutive_airport_mismatch"]:
            return 0.0
        previous_airport = airport_catalog.id_index.get(
            row["previous_landing_airport_id"]
        )
        takeoff_airport = airport_catalog.id_index.get(
            row["takeoff_airport_id"]
        )
        if previous_airport is None or takeoff_airport is None:
            return None
        return haversine(
            previous_airport.lat,
            previous_airport.lon,
            takeoff_airport.lat,
            takeoff_airport.lon,
        )

    return mismatches.with_columns(
        pl.struct(
            "previous_landing_airport_id",
            "takeoff_airport_id",
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
    airport_catalog: AirportCatalogView,
    displacement_threshold_km: float = DEFAULT_DISPLACEMENT_THRESHOLD_KM,
) -> dict[str, int | float | None]:
    """Calculate airport-continuity metrics for consecutive flights per ICAO."""
    if displacement_threshold_km < 0:
        raise ValueError("displacement_threshold_km must be non-negative")

    transitions = consecutive_airport_details(
        df_flights,
        airport_catalog=airport_catalog,
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
