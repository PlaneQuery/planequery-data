import argparse
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
import polars as pl
from data_engineering.adsb.read_adsb import read_adsb
from data_engineering.bts.read_bts_ontime_to_flights import get_bts_flights_for_day
from data_engineering.opensky.create_flights_from_trino_parquets import read_opensky_flights

from data_engineering.flights.flight_type import get_flights
from data_engineering.eurocontrol.read import read_eurocontrol_flights
from data_engineering.openairframes.read import add_latest_icao_info
from flights.flights_comparison import df_flights_comparision, df_flights_comparison_stats
from flights.flights_match_adsb import get_matching_icaos_in_flights
from utils import current_commit_hash
from data_engineering.adsbx.get_flights import get_adsbx_flights_for_day
from data_engineering.flights.sfdps_to_flights import get_sfdps_flights_day
from data_engineering.utils import OUTPUT_DIR

DEFAULT_TARGET_DATE = date(2026, 3, 1)

def default_stats_path() -> Path:
    run_time = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M")
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


def is_american_icao_expr(icao_col: str = "icao") -> pl.Expr:
    icao = pl.col(icao_col).str.to_lowercase()
    return (icao >= "a00000") & (icao <= "afffff")


def pia_or_american_ladd_expr() -> pl.Expr:
    return (
        pl.col("pia").fill_null(False)
        | (is_american_icao_expr() & pl.col("ladd").fill_null(False))
    )



def read_fr24_flights(target_date: date) -> pl.DataFrame:
    from data_engineering.flights.flight_type import (
    with_flight_schema_columns,
    )
    return with_flight_schema_columns(
        pl.read_parquet(fr24_flights_parquet_path(target_date))
    )

def get_algorithm_adsbx_flights(target_date: date) -> pl.DataFrame:
    from data_engineering.flights.flight_type import (
        flights_algorithm_output_path,
        with_flight_schema_columns,
    )
    from flights_algorithm.main import run_main

    output_path = flights_algorithm_output_path(target_date, source="adsbx")
    if output_path.exists():
        return with_flight_schema_columns(pl.read_parquet(output_path))

    print(f"Missing algorithm ADSBX flights for {target_date}; generating at {output_path}")
    run_main(target_date=target_date, use_adsbx=True)
    if not output_path.exists():
        raise FileNotFoundError(
            f"Failed to generate algorithm ADSBX flights for {target_date}: {output_path}"
        )
    return with_flight_schema_columns(pl.read_parquet(output_path))


def source_flights_reader(
    src: str,
    adsb_src: str = "adsb.lol",
):
    if src == "adsbx":
        return get_adsbx_flights_for_day
    if src == "algorithm":
        if adsb_src == "adsbx":
            return get_algorithm_adsbx_flights
        if adsb_src != "adsb.lol":
            raise ValueError(f"algorithm source does not support adsb_src={adsb_src!r}")
        return get_flights
    if src == "sfdps":
        return get_sfdps_flights_day
    if src == "bts":
        return get_bts_flights_for_day
    if src =="opensky":
        return lambda target_date: read_opensky_flights(target_date)
    if src == "fr24":
        return read_fr24_flights
    if src == "eurocontrol":
        return read_eurocontrol_flights

    raise ValueError(f"Unknown flight source: {src}")

def adsb_srcs_for_test_source(src: str, adsb_srcs: list[str]) -> list[str]:
    if src == "algorithm":
        return adsb_srcs
    return [src]

def add_average_rows(df_stats: pl.DataFrame) -> pl.DataFrame:
    metadata_cols = {
        "test_date",
        "adsb_src",
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

def main(
    test_src: str | list[str],
    gold_src: str | list[str],
    test_dates: date | list[date],
    adsb_src: str | list[str] = "adsb.lol",
    use_all_icaos = False,
    filter_rotorcraft = False,
    pia_or_american_ladd_only = False,
):
    if isinstance(test_src, str):
        test_src = [test_src]
    if isinstance(gold_src, str):
        gold_src = [gold_src]
    if isinstance(adsb_src, str):
        adsb_src = [adsb_src]
    if isinstance(test_dates, date):
        test_dates = [test_dates]

    rows = []
    outputs = []
    adsb_by_date: dict[date, pl.DataFrame] = {}
    for src in test_src:
        for source_adsb_src in adsb_srcs_for_test_source(src, adsb_src):
            get_source_flights = source_flights_reader(
                src,
                adsb_src=source_adsb_src,
            )
            for gold in gold_src:
                get_gold_flights = source_flights_reader(
                    gold,
                )
                for test_date in test_dates:
                    df_test = get_source_flights(test_date)
                    df_gold = get_gold_flights(test_date)
                    expr = (pl.col("landing_time").dt.date() == test_date) & (pl.col("takeoff_time").dt.date() == test_date)
                    df_test = df_test.filter(expr)
                    df_gold = df_gold.filter(expr)
                    if pia_or_american_ladd_only:
                        df_test = df_test.filter(pia_or_american_ladd_expr())
                        df_gold = df_gold.filter(pia_or_american_ladd_expr())
                    if not use_all_icaos:
                        if test_date not in adsb_by_date:
                            adsb_by_date[test_date] = read_adsb(
                                test_date,
                                pia_or_american_ladd_only=pia_or_american_ladd_only,
                            )
                        df_adsb = adsb_by_date[test_date]
                        df_gold = get_matching_icaos_in_flights(df_gold, df_adsb)
                        df_test = df_test.filter(pl.col("icao").is_in(df_gold.get_column("icao").implode()))
                    else:
                        df_test = df_test.filter(pl.col("icao").is_in(df_gold.get_column("icao").implode()))
                    if filter_rotorcraft:
                        df_test = df_test.filter(~is_helicopter_expr())
                        df_gold = df_gold.filter(~is_helicopter_expr())
                    df = df_flights_comparision(df_test, df_gold, datetime(test_date.year, test_date.month, test_date.day), datetime(test_date.year, test_date.month, test_date.day) + timedelta(days=1), compare_airports=False)
                    stats = df_flights_comparison_stats(df)
                    outputs.append([df,stats])
                    rows.append({
                        "test_date": test_date.isoformat(),
                        "adsb_src": source_adsb_src,
                        "test_src": src,
                        "gold_src": gold,
                        "use_all_icaos": use_all_icaos,
                        "filter_rotorcraft": filter_rotorcraft,
                        "pia_or_american_ladd_only": pia_or_american_ladd_only,
                        **stats,
                    })
                    print(f"Stats for {src}:{gold}: {stats}")

    stats_output = default_stats_path()
    stats_output.parent.mkdir(parents=True, exist_ok=True)
    df_stats = pl.DataFrame(rows)
    if len(test_dates) > 1:
        df_stats = add_average_rows(df_stats)
    df_stats.write_csv(stats_output)
    print(f"Stats CSV saved to {stats_output}")
    return outputs, df_stats # TODO: Probably change this to make df_stats from outputs in another function.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-src", nargs="+", default=["adsbx"])
    parser.add_argument("--adsb-src", nargs="+", default=["adsb.lol"])
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
    args = parser.parse_args()

    test_dates = args.test_dates
    if test_dates is None:
        test_dates = [DEFAULT_TARGET_DATE]
    elif isinstance(test_dates, list) and test_dates and isinstance(test_dates[0], str):
        test_dates = [date.fromisoformat(d) for d in test_dates]

    main(
        test_src=args.test_src,
        adsb_src=args.adsb_src,
        gold_src=args.gold_src,
        test_dates=test_dates,
        use_all_icaos=args.use_all_icaos,
        filter_rotorcraft=args.filter_rotorcraft,
        pia_or_american_ladd_only=args.pia_or_american_ladd_only,
    )
