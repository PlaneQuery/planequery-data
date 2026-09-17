import argparse
import json
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
import polars as pl
from data_engineering.adsb.read_adsb import read_adsb
from data_engineering.bts.read_bts_ontime_to_flights import get_bts_flights_for_day
from data_engineering.filters import pia_or_american_ladd

from data_engineering.flights.flight_type import ALGORITHM_ADSB_SOURCES, get_flights
from data_engineering.eurocontrol.read import get_eurocontrol_flights
from data_engineering.opdi.read import get_opdi_flights
from data_engineering.openairframes.read import add_latest_icao_info
from flights.evaluation.consecutive_airports import (
    DEFAULT_DISPLACEMENT_THRESHOLD_KM,
    consecutive_airport_details,
    consecutive_airport_stats,
)
from flights.flights_comparison import df_flights_comparison, df_flights_comparison_stats
from flights.ground_truth.bts_and_sfdps_to_gold import (
    get_ground_truth_flights_for_day,
)
from flights.flights_match_adsb import ADSB_MATCHING_COLUMNS, get_matching_icaos_in_flights
from flights_airport_nn.derive_day import DEFAULT_MODEL_PATH as DEFAULT_AIRPORT_NN_MODEL
from planequery_data.utils import current_commit_hash
from data_engineering.flights.sfdps_to_flights import get_sfdps_flights_day
from data_engineering.flights.mike import get_mike_flights_for_day
from data_engineering.utils import OUTPUT_DIR

DEFAULT_TARGET_DATE = date(2026, 3, 1)
DEFAULT_ADSB_SRC_FOR_MATCHING = "adsbx"
DEFAULT_ADSB_SRC_FOR_FLIGHTS = "adsblol"


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


def read_fr24_flights(target_date: date) -> pl.DataFrame:
    from data_engineering.flights.flight_type import (
    with_flight_schema_columns,
    )
    return with_flight_schema_columns(
        pl.read_parquet(fr24_flights_parquet_path(target_date))
    )

def get_algorithm_flights_for_adsb_src(
    target_date: date,
    adsb_src: str,
    no_airports_model: bool = False,
) -> pl.DataFrame:
    from data_engineering.flights.flight_type import flights_algorithm_output_path
    from flights_algorithm.main import run_main

    output_path = flights_algorithm_output_path(target_date, adsb_src=adsb_src, no_airports_model=no_airports_model)
    if output_path.exists():
        return get_flights(
            target_date,
            no_airports_model=no_airports_model,
            adsb_src=adsb_src,
        )

    label = "algorithm (no airports model)" if no_airports_model else "algorithm"
    print(f"Missing {label} {adsb_src} flights for {target_date}; generating at {output_path}")
    run_main(
        target_date=target_date,
        no_airports_model=no_airports_model,
        source=adsb_src,
    )
    if not output_path.exists():
        raise FileNotFoundError(
            f"Failed to generate {label} {adsb_src} flights for {target_date}: {output_path}"
        )
    return get_flights(
        target_date,
        no_airports_model=no_airports_model,
        adsb_src=adsb_src,
    )


def get_algorithm_airports_nn_flights(
    target_date: date,
    adsb_src: str,
    model_path: Path = DEFAULT_AIRPORT_NN_MODEL,
) -> pl.DataFrame:
    """Assign airports to algorithm segments with the selected airport network."""
    if adsb_src not in ALGORITHM_ADSB_SOURCES:
        raise ValueError(f"algorithm-airports-nn does not support adsb_src={adsb_src!r}")
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Airport NN checkpoint not found: {model_path}. "
            "Pass --airport-nn-model with an existing checkpoint."
        )
    flights = get_algorithm_flights_for_adsb_src(
        target_date, adsb_src, no_airports_model=True
    )
    if flights.is_empty():
        return flights
    if adsb_src in ("adsblol", "adsbx"):
        from flights_airport_nn.derive_day import derive_day

        path = derive_day(
            target_date, source=adsb_src, model_path=model_path, flights=flights
        )
        return pl.read_parquet(path)
    from data_engineering.flights.flight_type import flights_to_flights_df
    from flights_airport_nn.inference import load_model, run_inference

    model, feature_config = load_model(model_path)
    predicted = run_inference(
        target_date,
        model,
        feature_config,
        adsb_src=adsb_src,
        flights=flights,
    )
    if len(predicted) != flights.height:
        raise ValueError("Airport NN inference did not return every flight")
    return flights_to_flights_df(list(predicted.values()))


def source_flights_reader(
    src: str,
    adsb_src: str = DEFAULT_ADSB_SRC_FOR_FLIGHTS,
    airport_nn_model_path: Path = DEFAULT_AIRPORT_NN_MODEL,
):
    if src == "adsbx":
        return lambda target_date: get_flights(target_date, algorithm="adsbx")
    if src == "algorithm":
        if adsb_src in ALGORITHM_ADSB_SOURCES:
            return lambda target_date: get_algorithm_flights_for_adsb_src(target_date, adsb_src)
        raise ValueError(f"algorithm source does not support adsb_src={adsb_src!r}")
    if src == "algorithm-no-airports":
        if adsb_src in ALGORITHM_ADSB_SOURCES:
            return lambda target_date: get_algorithm_flights_for_adsb_src(target_date, adsb_src, no_airports_model=True)
        raise ValueError(f"algorithm-no-airports source does not support adsb_src={adsb_src!r}")
    if src == "algorithm-airports-nn":
        if adsb_src in ALGORITHM_ADSB_SOURCES:
            @lru_cache(maxsize=2)
            def read_nn(target_date):
                return get_algorithm_airports_nn_flights(
                    target_date, adsb_src, airport_nn_model_path
                )

            return read_nn
        raise ValueError(f"algorithm-airports-nn does not support adsb_src={adsb_src!r}")
    if src == "sfdps":
        return get_sfdps_flights_day
    if src == "bts":
        return get_bts_flights_for_day
    if src == "ground_truth":
        return get_ground_truth_flights_for_day
    if src =="opensky":
        return lambda target_date: get_flights(target_date, algorithm="opensky")
    if src == "fr24":
        return read_fr24_flights
    if src == "eurocontrol":
        return get_eurocontrol_flights
    if src == "opdi":
        return get_opdi_flights
    if src == "mike":
        return get_mike_flights_for_day

    raise ValueError(f"Unknown flight source: {src}")

def adsb_srcs_for_test_source(src: str, adsb_srcs: list[str]) -> list[str]:
    if src in ("algorithm", "algorithm-no-airports", "algorithm-airports-nn"):
        return adsb_srcs
    return [src]

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
    adsb_src: str = DEFAULT_ADSB_SRC_FOR_MATCHING,
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
    test_src: str | list[str],
    gold_src: str | list[str],
    test_dates: date | list[date],
    adsb_src: str | list[str] = DEFAULT_ADSB_SRC_FOR_FLIGHTS,
    matching_adsb_src: str = DEFAULT_ADSB_SRC_FOR_MATCHING,
    use_all_icaos = False,
    filter_rotorcraft = False,
    pia_or_american_ladd_only = False,
    include_consecutive_airport_metrics = False,
    consecutive_airport_displacement_threshold_km = (
        DEFAULT_DISPLACEMENT_THRESHOLD_KM
    ),
    write_outputs = True,
    airport_nn_model_path: Path = DEFAULT_AIRPORT_NN_MODEL,
):
    if isinstance(test_src, str):
        test_src = [test_src]
    if isinstance(gold_src, str):
        gold_src = [gold_src]
    if isinstance(adsb_src, str):
        adsb_src = [adsb_src]
    if isinstance(test_dates, date):
        test_dates = [test_dates]

    results = []
    rows = []
    stats_output = default_stats_path()
    comparison_dir = stats_output.with_suffix("")
    comparison_index = []
    if write_outputs:
        comparison_dir.mkdir(parents=True, exist_ok=True)
    matched_gold_by_date: dict[tuple[date, str, str, bool], pl.DataFrame] = {}
    adsb_by_date: dict[tuple[date, str, str, bool], pl.DataFrame] = {}
    for src in test_src:
        for source_adsb_src in adsb_srcs_for_test_source(src, adsb_src):
            if src == "algorithm-airports-nn":
                get_source_flights = source_flights_reader(
                    src,
                    adsb_src=source_adsb_src,
                    airport_nn_model_path=airport_nn_model_path,
                )
            else:
                get_source_flights = source_flights_reader(
                    src,
                    adsb_src=source_adsb_src,
                )
            for gold in gold_src:
                if gold == "algorithm-airports-nn":
                    get_gold_flights = source_flights_reader(
                        gold, airport_nn_model_path=airport_nn_model_path
                    )
                else:
                    get_gold_flights = source_flights_reader(gold)
                for test_date in test_dates:
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
                            gold,
                            pia_or_american_ladd_only,
                        )
                        if matching_cache_key not in matched_gold_by_date:
                            adsb_cache_key = (
                                test_date,
                                matching_adsb_src,
                                gold,
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
                            df_test
                        )
                        df_test = df_test.join(
                            consecutive_details.drop(
                                "takeoff_airport_ident"
                            ),
                            on=["icao", "takeoff_time"],
                            how="left",
                        )
                    df = df_flights_comparison(df_test, df_gold, datetime(test_date.year, test_date.month, test_date.day), datetime(test_date.year, test_date.month, test_date.day) + timedelta(days=1), compare_airports=False)
                    results.append(df)
                    stats_adsb_src = (
                        source_adsb_src
                        if src in ("algorithm", "algorithm-no-airports", "algorithm-airports-nn")
                        else None
                    )
                    if write_outputs:
                        comparison_path = comparison_dir / (
                            f"comparison-{len(comparison_index):04d}.parquet"
                        )
                        temporary_path = comparison_path.with_name(
                            comparison_path.name + ".tmp"
                        )
                        df.write_parquet(temporary_path)
                        temporary_path.replace(comparison_path)
                        comparison_index.append({
                            "path": comparison_path.name,
                            "test_date": test_date.isoformat(),
                            "test_src": src,
                            "gold_src": gold,
                            "adsb_src": stats_adsb_src,
                            "matching_adsb_src": matching_adsb_src,
                            "use_all_icaos": use_all_icaos,
                            "filter_rotorcraft": filter_rotorcraft,
                            "pia_or_american_ladd_only": pia_or_american_ladd_only,
                            "rows": df.height,
                        })
                        index_path = comparison_dir / "index.json"
                        temporary_index_path = comparison_dir / "index.json.tmp"
                        temporary_index_path.write_text(
                            json.dumps(
                                {
                                    "stats_csv": f"../{stats_output.name}",
                                    "comparisons": comparison_index,
                                },
                                indent=2,
                            ) + "\n"
                        )
                        temporary_index_path.replace(index_path)
                        print(f"Comparison Parquet saved to {comparison_path}")
                    stats = df_flights_comparison_stats(df)
                    if include_consecutive_airport_metrics:
                        stats.update(
                            consecutive_airport_stats(
                                df_test,
                                displacement_threshold_km=(
                                    consecutive_airport_displacement_threshold_km
                                ),
                            )
                        )
                    rows.append({
                        "test_date": test_date.isoformat(),
                        "adsb_src": stats_adsb_src,
                        "matching_adsb_src": matching_adsb_src,
                        "test_src": src,
                        "gold_src": gold,
                        "use_all_icaos": use_all_icaos,
                        "filter_rotorcraft": filter_rotorcraft,
                        "pia_or_american_ladd_only": pia_or_american_ladd_only,
                        **stats,
                    })
                    print(f"Stats for {src}:{gold}: {stats}")

    df_stats = pl.DataFrame(rows)
    if len(test_dates) > 1:
        df_stats = add_average_rows(df_stats)
    if write_outputs:
        df_stats.write_csv(stats_output)
        print(f"Stats CSV saved to {stats_output}")
    return results, df_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-src", nargs="+", default=["adsbx"])
    parser.add_argument("--adsb-src", nargs="+", default=[DEFAULT_ADSB_SRC_FOR_FLIGHTS])
    parser.add_argument("--airport-nn-model", type=Path, default=DEFAULT_AIRPORT_NN_MODEL)
    parser.add_argument("--matching-adsb-src", default=DEFAULT_ADSB_SRC_FOR_MATCHING)
    parser.add_argument("--gold-src", nargs="+", default=["bts"])
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
