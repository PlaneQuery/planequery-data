from datetime import date
import polars as pl

from airports.polars_exprs import airport_size_expr
from airports.runtime import airport_catalog_for_date
from data_engineering.adsb.read_adsb import ADSBSource
from data_profiling.metrics.flights import FlightCohort
from flights.evaluation.evaluation import main as run_evaluation
from flights.flight_source import get_flights
from flights.sources import ALGORITHM_NEAREST_SOURCE, SFDPS_SOURCE

TARGET_DATE = date(2026, 3, 1)
AIRPORT_CATALOG = airport_catalog_for_date(TARGET_DATE)

results, stats = run_evaluation(
    test_src=ALGORITHM_NEAREST_SOURCE,
    gold_src=SFDPS_SOURCE,
    test_dates=TARGET_DATE,
    adsb_src=ADSBSource.ADSBLOL,
    write_outputs=False
)
comparison: pl.DataFrame = results[0]

evaluation_columns = [
    "icao",
    "takeoff_time",
    "match_status",
    "same_takeoff_airport_id",
    "same_landing_airport_id",
    "same_airport_id",
    "takeoff_airport_ident_df1",
    "landing_airport_ident_df1",
]
evaluation_annotations = (
    comparison
    .filter(pl.col("match_status") != "df1_only")
    .select(evaluation_columns)
)

df_test = get_flights(
    ALGORITHM_NEAREST_SOURCE,
    TARGET_DATE,
    ADSBSource.ADSBLOL,
)
profile_df = df_test.join(
    evaluation_annotations,
    on=["icao", "takeoff_time"],
    how="left",
).with_columns(airport_size_expr(AIRPORT_CATALOG, "takeoff_airport_id"))

AIRPORT_SIZES = (
    "large_airport",
    "medium_airport",
    "small_airport",
    "unknown",
)


def add_airport_size_subsets(
    cohort: FlightCohort,
    airport_size_col: str,
    *,
    add_accuracy: bool = False,
) -> None:
    for airport_size in AIRPORT_SIZES:
        predicate = (
            pl.col(airport_size_col).is_null()
            if airport_size == "unknown"
            else pl.col(airport_size_col) == airport_size
        )
        size_cohort = cohort.add_subset(airport_size, predicate)
        if add_accuracy:
            size_cohort.add_subset(
                "Correct",
                pl.col("same_takeoff_airport_id"),
            )
            size_cohort.add_subset(
                "Incorrect",
                ~pl.col("same_takeoff_airport_id"),
            )

funnel = FlightCohort("All algorithm flights", profile_df)
evaluatable = funnel.add_subset(
    "Evaluatable flights",
    pl.col("match_status") == "both",
)
evaluatable.add_subset(
    "Correct takeoff airport",
    pl.col("same_takeoff_airport_id"),
)
evaluatable.add_subset(
    "Incorrect takeoff airport",
    ~pl.col("same_takeoff_airport_id"),
)
funnel.add_subset(
    "Not evaluatable",
    pl.col("match_status").fill_null("") != "both",
)

all_by_size = FlightCohort(
    "All algorithm flights by predicted takeoff-airport size",
    profile_df,
)
add_airport_size_subsets(all_by_size, "takeoff_airport_size")

evaluatable_by_predicted_size = FlightCohort(
    "Evaluatable flights by predicted takeoff-airport size",
    evaluatable.df,
)
add_airport_size_subsets(
    evaluatable_by_predicted_size,
    "takeoff_airport_size",
)

evaluatable_by_size_df = evaluatable.df.with_columns(
    airport_size_expr(AIRPORT_CATALOG, "takeoff_airport_id_df1").alias(
        "gold_takeoff_airport_size"
    )
)
accuracy_by_size = FlightCohort(
    "Evaluatable flights by gold takeoff-airport size",
    evaluatable_by_size_df,
)
add_airport_size_subsets(
    accuracy_by_size,
    "gold_takeoff_airport_size",
    add_accuracy=True,
)

print(funnel.to_text())
print()
print(all_by_size.to_text())
print()
print(
    evaluatable_by_predicted_size.to_text(
        root_count=funnel.count,
        parent_count=funnel.count,
    )
)
print()
print(
    accuracy_by_size.to_text(
        root_count=funnel.count,
        parent_count=funnel.count,
    )
)
