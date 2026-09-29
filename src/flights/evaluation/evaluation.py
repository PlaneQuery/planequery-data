import argparse
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

from airports.runtime import airport_catalog_for_date
from data_engineering.adsb.read_adsb import ADSBSource, read_adsb
from data_engineering.filters import pia_or_american_ladd
from data_engineering.utils import OUTPUT_DIR
from flights.airport_nn.source import AlgorithmNNFlightSource
from flights.evaluation.consecutive_airports import (
    DEFAULT_DISPLACEMENT_THRESHOLD_KM,
    consecutive_airport_details,
    consecutive_airport_stats,
)
from flights.flights_comparison import df_flights_comparison, df_flights_comparison_stats
from flights.flights_match_adsb import ADSB_MATCHING_COLUMNS, get_matching_icaos_in_flights
from flights.flight_source import (
    ADSBFlightSource,
    FlightSource,
    get_flights,
)
from flights.sources import (
    ALGORITHM_NEAREST_SOURCE,
    BTS_SOURCE,
    FLIGHT_SOURCES_BY_NAME,
)
from planequery_data.utils import current_commit_hash

DEFAULT_TARGET_DATE = date(2026, 3, 1)
DEFAULT_ADSB_SRC_FOR_MATCHING = ADSBSource.ADSBX
DEFAULT_ADSB_SRC_FOR_FLIGHTS = ADSBSource.ADSBLOL


def default_stats_path() -> Path:
    run_time = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S-%f")
    commit_hash = current_commit_hash()
    filename = f"{run_time}_{commit_hash}_flights_evaluation.csv"
    return OUTPUT_DIR / "data" / "experiments" / "comparison" / filename


def is_helicopter_expr(
    aircraft_type_col: str = "aircraft_type",
    aircraft_description_col: str = "aircraft_description",
    category_col: str = "category",
) -> pl.Expr:
    HELICOPTER_PATTERN = (
    r"(?i)"
    r"HELICOPTER|HELIBUS|"
    r"SIKORSKY|ROBINSON|BELL|AGUSTA|AIRBUS HELICOPTERS|EUROCOPTER|"
    r"HUGHES|MD HELICOPTERS"
    )
    HELICOPTER_TYPE_CODES = [
        "A139", "A169", "A189",
        "B06", "B407", "B412", "B429",
        "EC20", "EC30", "EC35", "EC45",
        "H500",
        "R22", "R44", "R66",
        "S76", "S92",
    ]

    return (
        (pl.col(category_col) == "A7")
        | (pl.col(aircraft_type_col).is_in(HELICOPTER_TYPE_CODES))
        | (
            pl.col(aircraft_description_col)
            .fill_null("")
            .str.contains(HELICOPTER_PATTERN)
        )
    )


def resolve_source(source: FlightSource | str) -> FlightSource:
    if isinstance(source, str):
        try:
            return FLIGHT_SOURCES_BY_NAME[source]
        except KeyError:
            raise ValueError(f"Unknown flight source: {source}") from None
    return source


def source_flights_reader(
    source: FlightSource | str,
    adsb_src: ADSBSource = DEFAULT_ADSB_SRC_FOR_FLIGHTS,
):
    source = resolve_source(source)
    source_adsb = (
        ADSBSource(adsb_src) if isinstance(source, ADSBFlightSource) else None
    )

    def read_source(target_date: date) -> pl.DataFrame:
        if (
            isinstance(source, AlgorithmNNFlightSource)
            and not source.model_path.is_file()
        ):
            raise FileNotFoundError(
                f"Airport NN checkpoint not found: {source.model_path}. "
                "Pass --airport-nn-model with an existing checkpoint."
            )
        return get_flights(source, target_date, source_adsb)

    return read_source


def adsb_srcs_for_test_source(
    source: FlightSource | str,
    adsb_srcs: list[ADSBSource],
) -> list[ADSBSource]:
    if isinstance(resolve_source(source), ADSBFlightSource):
        return adsb_srcs
    return [DEFAULT_ADSB_SRC_FOR_FLIGHTS]


def add_average_rows(df_stats: pl.DataFrame) -> pl.DataFrame:
    metadata_cols = {
        "test_date",
        "adsb_src",
        "matching_adsb_src",
        "test_src",
        "gold_src",
        "use_all_icaos",
        "filter_rotorcraft",
        "pia_or_american_ladd_only",
    }
    stats_cols = [col for col in df_stats.columns if col not in metadata_cols]
    df_stats = df_stats.with_columns(pl.col(stats_cols).cast(pl.Float64))
    df_avg = (
        df_stats
        .group_by([
            "adsb_src",
            "matching_adsb_src",
            "test_src",
            "gold_src",
            "use_all_icaos",
            "filter_rotorcraft",
            "pia_or_american_ladd_only",
        ])
        .agg([pl.col(col).mean().alias(col) for col in stats_cols])
        .with_columns(pl.lit("average").alias("test_date"))
        .select(df_stats.columns)
    )
    return pl.concat([df_stats, df_avg])


def read_adsb_for_matching(
    df_flights: pl.DataFrame,
    target_date: date,
    *,
    adsb_src: ADSBSource = DEFAULT_ADSB_SRC_FOR_MATCHING,
    pia_or_american_ladd_only: bool,
) -> pl.DataFrame:
    return read_adsb(
        target_date,
        icaos=df_flights.get_column("icao").drop_nulls().unique().to_list(),
        columns=ADSB_MATCHING_COLUMNS,
        pia_or_american_ladd_only=pia_or_american_ladd_only,
        source=adsb_src,
    )


def main( # TODO: Maybe rename to run_evaluation
    test_src: FlightSource | str | list[FlightSource | str],
    gold_src: FlightSource | str | list[FlightSource | str],
    test_dates: date | list[date],
    adsb_src: ADSBSource | list[ADSBSource] = DEFAULT_ADSB_SRC_FOR_FLIGHTS,
    matching_adsb_src: ADSBSource = DEFAULT_ADSB_SRC_FOR_MATCHING,
    use_all_icaos = False,
    filter_rotorcraft = False,
    pia_or_american_ladd_only = False,
    include_consecutive_airport_metrics = False,
    consecutive_airport_displacement_threshold_km = (
        DEFAULT_DISPLACEMENT_THRESHOLD_KM
    ),
    write_outputs = True,
    airport_nn_model_path: Path | None = None,
):
    if isinstance(test_src, (FlightSource, str)):
        test_src = [test_src]
    if isinstance(gold_src, (FlightSource, str)):
        gold_src = [gold_src]
    test_sources = [resolve_source(source) for source in test_src]
    gold_sources = [resolve_source(source) for source in gold_src]
    if airport_nn_model_path is not None:
        test_sources = [
            replace(source, model_path=airport_nn_model_path)
            if isinstance(source, AlgorithmNNFlightSource)
            else source
            for source in test_sources
        ]
    if isinstance(adsb_src, str):
        adsb_src = [ADSBSource(adsb_src)]
    else:
        adsb_src = [ADSBSource(source) for source in adsb_src]
    matching_adsb_src = ADSBSource(matching_adsb_src)
    if isinstance(test_dates, date):
        test_dates = [test_dates]

    results = []
    rows = []
    stats_output = default_stats_path()
    matched_gold_by_date: dict[
        tuple[date, ADSBSource, str, bool], pl.DataFrame
    ] = {}
    adsb_by_date: dict[tuple[date, ADSBSource, str, bool], pl.DataFrame] = {}
    for source in test_sources:
        for source_adsb_src in adsb_srcs_for_test_source(source, adsb_src):
            get_source_flights = source_flights_reader(
                source,
                adsb_src=source_adsb_src,
            )
            for gold_source in gold_sources:
                get_gold_flights = source_flights_reader(gold_source)
                for test_date in test_dates:
                    airport_catalog = airport_catalog_for_date(test_date)
                    df_test = get_source_flights(test_date)
                    df_gold = get_gold_flights(test_date)
                    expr = (pl.col("landing_time").dt.date() == test_date) & (pl.col("takeoff_time").dt.date() == test_date)
                    df_test = df_test.filter(expr)
                    df_gold = df_gold.filter(expr)
                    if pia_or_american_ladd_only:
                        df_test = df_test.filter(pia_or_american_ladd())
                        df_gold = df_gold.filter(pia_or_american_ladd())
                    if not use_all_icaos:
                        matching_cache_key = (
                            test_date,
                            matching_adsb_src,
                            gold_source.name,
                            pia_or_american_ladd_only,
                        )
                        if matching_cache_key not in matched_gold_by_date:
                            adsb_cache_key = (
                                test_date,
                                matching_adsb_src,
                                gold_source.name,
                                pia_or_american_ladd_only,
                            )
                            if adsb_cache_key not in adsb_by_date:
                                adsb_by_date[adsb_cache_key] = read_adsb_for_matching(
                                    df_gold,
                                    test_date,
                                    adsb_src=matching_adsb_src,
                                    pia_or_american_ladd_only=pia_or_american_ladd_only,
                                )
                            matched_gold_by_date[matching_cache_key] = get_matching_icaos_in_flights(
                                df_gold,
                                adsb_by_date[adsb_cache_key],
                                datetime(test_date.year, test_date.month, test_date.day),
                            )
                        df_gold = matched_gold_by_date[matching_cache_key]
                        df_test = df_test.filter(pl.col("icao").is_in(df_gold.get_column("icao").implode()))
                    else:
                        df_test = df_test.filter(pl.col("icao").is_in(df_gold.get_column("icao").implode()))
                    if filter_rotorcraft:
                        df_test = df_test.filter(~is_helicopter_expr())
                        df_gold = df_gold.filter(~is_helicopter_expr())
                    if include_consecutive_airport_metrics:
                        consecutive_details = consecutive_airport_details(
                            df_test,
                            airport_catalog=airport_catalog,
                        )
                        df_test = df_test.join(
                            consecutive_details.drop(
                                "takeoff_airport_id",
                                "takeoff_airport_ident",
                            ),
                            on=["icao", "takeoff_time"],
                            how="left",
                        )
                    df = df_flights_comparison(
                        df_test,
                        df_gold,
                        datetime(test_date.year, test_date.month, test_date.day),
                        datetime(test_date.year, test_date.month, test_date.day)
                        + timedelta(days=1),
                        airport_catalog,
                        compare_airports=False,
                    )
                    results.append(df)
                    stats_adsb_src = (
                        source_adsb_src
                        if isinstance(source, ADSBFlightSource)
                        else None
                    )
                    stats = df_flights_comparison_stats(df)
                    if include_consecutive_airport_metrics:
                        stats.update(
                            consecutive_airport_stats(
                                df_test,
                                airport_catalog=airport_catalog,
                                displacement_threshold_km=(
                                    consecutive_airport_displacement_threshold_km
                                ),
                            )
                        )
                    rows.append({
                        "test_date": test_date.isoformat(),
                        "adsb_src": stats_adsb_src,
                        "matching_adsb_src": matching_adsb_src,
                        "test_src": source.name,
                        "gold_src": gold_source.name,
                        "use_all_icaos": use_all_icaos,
                        "filter_rotorcraft": filter_rotorcraft,
                        "pia_or_american_ladd_only": pia_or_american_ladd_only,
                        **stats,
                    })
                    print(
                        f"Stats for {source.name}:{gold_source.name}: {stats}"
                    )

    df_stats = pl.DataFrame(rows)
    if len(test_dates) > 1:
        df_stats = add_average_rows(df_stats)
    if write_outputs:
        stats_output.parent.mkdir(parents=True, exist_ok=True)
        df_stats.write_csv(stats_output)
        print(f"Stats CSV saved to {stats_output}")
    return results, df_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-src",
        nargs="+",
        choices=FLIGHT_SOURCES_BY_NAME,
        default=[ALGORITHM_NEAREST_SOURCE.name],
    )
    parser.add_argument(
        "--adsb-src",
        nargs="+",
        type=ADSBSource,
        choices=tuple(ADSBSource),
        default=[DEFAULT_ADSB_SRC_FOR_FLIGHTS],
    )
    parser.add_argument(
        "--matching-adsb-src",
        type=ADSBSource,
        choices=tuple(ADSBSource),
        default=DEFAULT_ADSB_SRC_FOR_MATCHING,
    )
    parser.add_argument(
        "--airport-nn-model",
        type=Path,
    )
    parser.add_argument(
        "--gold-src",
        nargs="+",
        choices=FLIGHT_SOURCES_BY_NAME,
        default=[BTS_SOURCE.name],
    )
    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument("--test-dates", nargs="+", dest="test_dates")
    test_date_sets = {
        name: value
        for name, value in globals().items()
        if name.startswith("TEST_DATES_")
    }
    for name, value in sorted(test_date_sets.items()):
        date_group.add_argument(f"--{name}", action="store_const", const=value, dest="test_dates")
    parser.add_argument("--use_all_icaos", action="store_true", default=False)
    parser.add_argument("--filter_rotorcraft", action="store_true", default=False)
    parser.add_argument("--pia-or-american-ladd-only", action="store_true", default=False)
    parser.add_argument(
        "--include-consecutive-airport-metrics",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--consecutive-airport-displacement-threshold-km",
        type=float,
        default=DEFAULT_DISPLACEMENT_THRESHOLD_KM,
    )
    args = parser.parse_args()

    test_dates = args.test_dates
    if test_dates is None:
        test_dates = [DEFAULT_TARGET_DATE]
    elif isinstance(test_dates, list) and test_dates and isinstance(test_dates[0], str):
        test_dates = [date.fromisoformat(d) for d in test_dates]

    main(
        test_src=args.test_src,
        adsb_src=args.adsb_src,
        airport_nn_model_path=args.airport_nn_model,
        matching_adsb_src=args.matching_adsb_src,
        gold_src=args.gold_src,
        test_dates=test_dates,
        use_all_icaos=args.use_all_icaos,
        filter_rotorcraft=args.filter_rotorcraft,
        pia_or_american_ladd_only=args.pia_or_american_ladd_only,
        include_consecutive_airport_metrics=(
            args.include_consecutive_airport_metrics
        ),
        consecutive_airport_displacement_threshold_km=(
            args.consecutive_airport_displacement_threshold_km
        ),
    )

# COLUMNS = ["icao", "takeoff_time", "landing_time", "takeoff_airport_ident", "landing_airport_ident", "takeoff_time_df1", "landing_time_df1", "takeoff_airport_ident_df1", "landing_airport_ident_df1", "match_status"] + ["consecutive_airport_displacement_km"]
