from data_engineering.openairframes.read import get_openairframes_faa_df
from data_engineering.utils import OUTPUT_DIR
import polars as pl

'''
Get percentage of pia icaos seen that broadcast an N-number 
Compare to Jack Sweeney PIA matchs. 
Checks that a pia icao isen't assigned to multiple N-numbers
'''

ACCIDENTAL_CALLSIGNS = ["N12345", "N123AB"]

BASE_PATH = OUTPUT_DIR / "data/raw/adsblol/parquet_output/v6"

def longest_period_between_icao_usage(df):
    df_icao_first_last = (
        df
        .group_by("icao")
        .agg(
            pl.col("time").min().alias("first_occurrence"),
            pl.col("time").max().alias("last_occurrence"),
        )
        .sort("icao")
    )
    df_icao_first_last = df_icao_first_last.with_columns(
        (pl.col("last_occurrence") - pl.col("first_occurrence")).alias("duration")
    )
    longest_row = df_icao_first_last.sort("duration", descending=True).row(0, named=True)
    print(f"Icao with longest duration: {longest_row['icao']}, duration: {longest_row['duration']}")

    duration_days = df_icao_first_last.get_column("duration").dt.total_seconds() / 86400
    stats = duration_days.describe()
    print("Duration statistics (days):")
    print(stats)
    for pct in [0.05, 0.25, 0.5, 0.75, 0.95]:
        print(f"{int(pct * 100)}th percentile: {duration_days.quantile(pct)}")

def pia_to_reg_broadcast_df(print_longest_period_between_icao_usage: bool = False) -> pl.DataFrame:
    df_faa = get_openairframes_faa_df()
    df_faa = df_faa.rename({"registration_number": "registration"})
    df_pia_adsb = (
        pl.scan_parquet(
            str(BASE_PATH / "year=*/month=*/day=*/icao_bucket=5/data.parquet"),
            hive_partitioning=True,
            missing_columns='insert'
        )
        .filter(
            pl.col("pia")
        )
        .collect()
    )
    if print_longest_period_between_icao_usage:
        longest_period_between_icao_usage(df_pia_adsb)
    total_pia_icaos = df_pia_adsb.get_column("icao").unique().len()
    print(f"Total pia icaos: {total_pia_icaos}")
    df_pia_n_number_df = df_pia_adsb.filter(pl.col("callsign").str.strip_chars().str.to_uppercase().str.starts_with("N"))
    df_pia_n_number_df = df_pia_n_number_df.unique(["icao", "callsign"])
    df_pia_n_number_df = df_pia_n_number_df.filter(~pl.col("callsign").is_in(ACCIDENTAL_CALLSIGNS))

    df = map_pia_n_numbers_to_real_icaos(df_pia_n_number_df, df_faa)
    df = df.rename({"icao": "pia_icao", "callsign": "registration", "real_icao": "real_icao"})
    print(f"Total PIA ICAOs with an unambiguous FAA mapping: {len(df)}")
    return df


def map_reg_to_icao(df_faa: pl.DataFrame) -> pl.DataFrame:
    """Return only registrations that have exactly one FAA transponder ICAO."""
    registration_column = (
        "registration" if "registration" in df_faa.columns else "registration_number"
    )
    return (
        df_faa
        .select([registration_column, "transponder_code_hex"])
        .rename(
            {
                registration_column: "registration",
                "transponder_code_hex": "real_icao",
            }
        )
        .with_columns(
            pl.col("registration").str.strip_chars().str.to_uppercase(),
            pl.col("real_icao").str.strip_chars().str.to_lowercase(),
        )
        .filter(
            pl.col("registration").is_not_null()
            & (pl.col("registration") != "")
            & pl.col("real_icao").is_not_null()
            & (pl.col("real_icao") != "")
        )
        .unique()
        .group_by("registration")
        .agg(pl.col("real_icao").unique().alias("real_icaos"))
        .filter(pl.col("real_icaos").list.len() == 1)
        .select(
            "registration",
            pl.col("real_icaos").list.first().alias("real_icao"),
        )
    )


def map_pia_n_numbers_to_real_icaos(
    df_pia_n_number_df: pl.DataFrame,
    df_faa: pl.DataFrame,
) -> pl.DataFrame:
    """Map PIA ICAOs to FAA ICAOs when their broadcast tail number is unique.

    A PIA ICAO is excluded when it broadcast more than one tail number, its tail
    number is missing from the FAA data, or that FAA registration has no unique
    transponder ICAO.
    """
    df_pia_n_number_df = df_pia_n_number_df.with_columns(
        pl.col("callsign").str.strip_chars().str.to_uppercase().alias("callsign")
    )
    unique_pia_icaos = (
        df_pia_n_number_df
        .group_by("icao")
        .agg(pl.col("callsign").unique().alias("callsigns"))
        .filter(pl.col("callsigns").list.len() == 1)
        .select("icao", pl.col("callsigns").list.first().alias("callsign"))
    )
    return unique_pia_icaos.join(
        map_reg_to_icao(df_faa),
        left_on="callsign",
        right_on="registration",
        how="inner",
    )

def get_pia_to_reg_broadcast_icaos(use_cache: bool = True):
    output_path = OUTPUT_DIR / "tmp" / "pia_to_reg_broadcast_df.csv"
    if use_cache and output_path.exists():
        return pl.read_csv(output_path)
    df = pia_to_reg_broadcast_df()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(output_path)
    return df

if __name__ == "__main__":
    print(get_pia_to_reg_broadcast_icaos(use_cache=False))
