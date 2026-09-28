"""Identify PIA and LADD aircraft broadcasting the same non-tail callsign."""

import argparse
from datetime import date, datetime, time, timedelta
from pathlib import Path

import polars as pl

from data_engineering.adsb.canonical import ADSBSource, adsb_parquet_root
from data_engineering.openairframes.read import scan_latest_icao_info_for_join
from data_engineering.utils import OUTPUT_DIR


BASE_PATH = adsb_parquet_root(ADSBSource.ADSBLOL)
DEFAULT_START_DATE = date(2025, 1, 1)
DEFAULT_END_DATE = date(2026, 7, 1)
DEFAULT_TIME_TOLERANCE = timedelta(minutes=200)
PIA_CALLSIGN_SWITCH_CACHE_PATH = Path(
    "resources/pia/real_and_pia_icao_equal_callsign_close.csv"
)

# FAA registration format, excluding I and O, which are not permitted as suffixes.
# The characters following N may contain at most five digits, or four characters
# when there is one suffix letter, or three characters when there are two.
N_NUMBER_PATTERN = (
    r"^N(?:[1-9][0-9]{0,4}|[1-9][0-9]{0,3}[A-HJ-NP-Z]|"
    r"[1-9][0-9]{0,2}[A-HJ-NP-Z]{2})$"
)
# ADS-B flight identification is at most eight alphanumeric characters. For
# this non-registration search, require an alphabetic first character so that
# placeholders and malformed numeric values such as 00000000 are discarded.
VALID_CALLSIGN_PATTERN = r"^[A-Z][A-Z0-9]{0,7}$"

MATCH_COLUMNS = [
    "pia_icao",
    "ladd_icao",
    "callsign",
    "pia_time",
    "ladd_time",
    "time_difference",
]


def _normalised_callsign() -> pl.Expr:
    return pl.col("callsign").str.strip_chars().str.to_uppercase()


def _validate_date_range(start_date: date, end_date: date) -> None:
    if start_date >= end_date:
        raise ValueError("start_date must be before end_date")


def _date_range_filter(start_date: date, end_date: date) -> pl.Expr:
    """Return an inclusive-start, exclusive-end timestamp filter."""
    _validate_date_range(start_date, end_date)
    start = datetime.combine(start_date, time.min)
    end = datetime.combine(end_date, time.min)
    return (pl.col("time") >= start) & (pl.col("time") < end)


def find_pia_ladd_callsign_matches(
    adsb: pl.DataFrame,
    time_tolerance: timedelta = DEFAULT_TIME_TOLERANCE,
) -> pl.DataFrame:
    """Find PIA observations whose non-N-number callsign is used by LADD.

    The result contains the closest pair of observations for each unique
    ``(pia_icao, ladd_icao, callsign)`` combination. A callsign must be present
    on both aircraft no more than ``time_tolerance`` apart. Callsigns are
    stripped and upper-cased before comparison.
    """
    required_columns = ["time", "icao", "callsign", "pia", "ladd"]
    missing_columns = set(required_columns).difference(adsb.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"ADSB data is missing required columns: {missing}")
    if time_tolerance < timedelta(0):
        raise ValueError("time_tolerance must not be negative")

    observations = (
        adsb.select(required_columns)
        .with_columns(_normalised_callsign().alias("callsign"))
        .filter(
            pl.col("time").is_not_null()
            & pl.col("icao").is_not_null()
            & pl.col("callsign").is_not_null()
            & (pl.col("callsign") != "")
            & pl.col("callsign").str.contains(VALID_CALLSIGN_PATTERN)
            & ~pl.col("callsign").str.contains(N_NUMBER_PATTERN)
        )
    )

    pia_observations = (
        observations.filter(pl.col("pia").fill_null(False))
        .select(
            pl.col("icao").alias("pia_icao"),
            "callsign",
            pl.col("time").alias("pia_time"),
        )
        .unique()
    )
    ladd_observations = (
        observations.filter(pl.col("ladd").fill_null(False))
        .select(
            pl.col("icao").alias("ladd_icao"),
            "callsign",
            pl.col("time").alias("ladd_time"),
        )
        .unique()
    )

    # Expand each PIA observation only to the LADD ICAOs known to use the same
    # callsign. This lets join_asof find the nearest observation independently
    # for every possible PIA/LADD aircraft pair.
    possible_matches = (
        pia_observations.join(
            ladd_observations.select("callsign", "ladd_icao").unique(),
            on="callsign",
            how="inner",
        )
        .filter(pl.col("pia_icao") != pl.col("ladd_icao"))
        .sort("pia_time")
    )

    if possible_matches.is_empty() or ladd_observations.is_empty():
        time_dtype = adsb.schema["time"]
        duration_dtype = (
            pl.Duration(time_dtype.time_unit)
            if isinstance(time_dtype, pl.Datetime)
            else pl.Duration("us")
        )
        return pl.DataFrame(
            schema={
                "pia_icao": pl.String,
                "ladd_icao": pl.String,
                "callsign": pl.String,
                "pia_time": time_dtype,
                "ladd_time": time_dtype,
                "time_difference": duration_dtype,
            }
        )

    matches = (
        possible_matches.join_asof(
            ladd_observations.sort("ladd_time"),
            left_on="pia_time",
            right_on="ladd_time",
            by=["callsign", "ladd_icao"],
            strategy="nearest",
            tolerance=time_tolerance,
            check_sortedness=False,
        )
        .filter(pl.col("ladd_time").is_not_null())
        .with_columns(
            (pl.col("pia_time") - pl.col("ladd_time"))
            .abs()
            .alias("time_difference")
        )
        # Enforce the limit explicitly as well as through join_asof. This keeps
        # grouped as-of ordering quirks from admitting an out-of-range pair.
        .filter(pl.col("time_difference") <= time_tolerance)
        .sort(
            [
                "pia_icao",
                "ladd_icao",
                "callsign",
                "time_difference",
                "pia_time",
                "ladd_time",
            ]
        )
        .unique(
            subset=["pia_icao", "ladd_icao", "callsign"],
            keep="first",
            maintain_order=True,
        )
        .select(MATCH_COLUMNS)
    )
    return matches


def pia_callsign_switch_df(
    start_date: date = DEFAULT_START_DATE,
    end_date: date = DEFAULT_END_DATE,
    time_tolerance: timedelta = DEFAULT_TIME_TOLERANCE,
) -> pl.DataFrame:
    """Read an ADS-B date range and return shared-callsign matches.

    ``start_date`` is inclusive and ``end_date`` is exclusive.
    """
    _validate_date_range(start_date, end_date)
    adsb = (
        pl.scan_parquet(
            str(BASE_PATH / "year=*/month=*/day=*/icao_bucket=5/data.parquet"),
            hive_partitioning=True,
            missing_columns="insert",
            extra_columns="ignore",
        )
        .filter(_date_range_filter(start_date, end_date))
        .select("time", "icao", "callsign", "pia", "ladd")
        .filter(pl.col("pia").fill_null(False) | pl.col("ladd").fill_null(False))
        .collect(engine="streaming")
    )
    return find_pia_ladd_callsign_matches(adsb, time_tolerance=time_tolerance)


def get_pia_callsign_switch_icaos(
    start_date: date = DEFAULT_START_DATE,
    end_date: date = DEFAULT_END_DATE,
    use_cache: bool = True,
) -> pl.DataFrame:
    """Return PIA-to-LADD mappings enriched with LADD registrations."""
    output_path = (
        OUTPUT_DIR
        / "tmp"
        / f"pia_callsign_switch_{start_date.isoformat()}_{end_date.isoformat()}.parquet"
    )
    if use_cache:
        df = pl.read_csv(
            PIA_CALLSIGN_SWITCH_CACHE_PATH,
            schema_overrides={
                "pia_icao": pl.String,
                "ladd_icao": pl.String,
            },
        )
    else:
        df = pia_callsign_switch_df(
            start_date=start_date,
            end_date=end_date,
        )

    registrations = scan_latest_icao_info_for_join().select(
        pl.col("icao").str.strip_chars().str.to_lowercase().alias("real_icao"),
        pl.col("registration").str.strip_chars().str.to_uppercase(),
    )
    df = (
        df.lazy()
        .rename({"ladd_icao": "real_icao"})
        .with_columns(pl.col("real_icao").str.strip_chars().str.to_lowercase())
        .join(registrations, on="real_icao", how="left")
        .select("pia_icao", "registration", "real_icao")
        .unique()
        .collect()
    )
    unmatched = df.filter(
        pl.col("registration").is_null() | (pl.col("registration") == "")
    )
    if not unmatched.is_empty():
        print(
            f"{unmatched.height} PIA callsign-switch mappings did not "
            "join to an OpenAirframes registration: "
            f"{unmatched.select('pia_icao', 'real_icao').sort(['pia_icao', 'real_icao']).to_dicts()}"
        )
    df = df.filter(
        pl.col("registration").is_not_null() & (pl.col("registration") != "")
    )

    conflicting_pia_icaos = (
        df.group_by("pia_icao")
        .agg(pl.col("real_icao").n_unique().alias("real_icao_count"))
        .filter(pl.col("real_icao_count") > 1)
        .sort("pia_icao")
    )
    if not conflicting_pia_icaos.is_empty():
        print(
            "Callsign-switch mappings contain PIA ICAOs with multiple "
            f"real ICAOs: {conflicting_pia_icaos.to_dicts()}"
        )

    if not use_cache:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output_path)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find PIA/LADD shared callsigns in an ADS-B date range."
    )
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        default=DEFAULT_START_DATE,
        help=f"inclusive YYYY-MM-DD date (default: {DEFAULT_START_DATE})",
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=DEFAULT_END_DATE,
        help=f"exclusive YYYY-MM-DD date (default: {DEFAULT_END_DATE})",
    )
    args = parser.parse_args()
    print(
        get_pia_callsign_switch_icaos(
            start_date=args.start_date,
            end_date=args.end_date,
            use_cache=False,
        )
    )
