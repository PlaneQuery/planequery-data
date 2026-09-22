from datetime import date
import polars as pl

from airports.polars_exprs import airport_size_expr
from data_engineering.flights.flight_type import get_flights
from data_profiling.metrics.flights import FlightCohort
from flights.evaluation.evaluation import main as run_evaluation

TARGET_DATE = date(2026, 3, 1)

results, stats = run_evaluation(
    test_src="algorithm-no-airports",
    gold_src="sfdps",
    test_dates=TARGET_DATE,
    adsb_src="adsblol",  # Internal name for adsb.lol
    write_outputs=False
)
comparison: pl.DataFrame = results[0]

evaluation_columns = [
    "icao",
    "takeoff_time",
    "match_status",
    "same_takeoff_airport_ident",
    "same_landing_airport_ident",
    "same_airport_ident",
    "takeoff_airport_ident_df1",
    "landing_airport_ident_df1",
]
evaluation_annotations = (
    comparison
    .filter(pl.col("match_status") != "df1_only")
    .select(evaluation_columns)
)

df_test = get_flights(
    TARGET_DATE,
    adsb_src="adsblol",
    algorithm="algorithm",
    no_airports_model=True,
)
profile_df = df_test.join(
    evaluation_annotations,
    on=["icao", "takeoff_time"],
    how="left",
).with_columns(airport_size_expr("takeoff_airport_ident"))

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
                pl.col("same_takeoff_airport_ident"),
            )
            size_cohort.add_subset(
                "Incorrect",
                ~pl.col("same_takeoff_airport_ident"),
            )

funnel = FlightCohort("All algorithm flights", profile_df)
evaluatable = funnel.add_subset(
    "Evaluatable flights",
    pl.col("match_status") == "both",
)
evaluatable.add_subset(
    "Correct takeoff airport",
    pl.col("same_takeoff_airport_ident"),
)
evaluatable.add_subset(
    "Incorrect takeoff airport",
    ~pl.col("same_takeoff_airport_ident"),
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
    airport_size_expr("takeoff_airport_ident_df1").alias(
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
