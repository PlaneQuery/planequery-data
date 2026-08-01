from data_engineering.openairframes.read import get_openairframes_faa_df
from data_engineering.utils import OUTPUT_DIR
import polars as pl

from pia.evaluation.ground_control import load_ground_control_mappings

'''
Get percentage of pia icaos seen that broadcast an N-number 
Compare to Jack Sweeney PIA matchs. 
Checks that a pia icao isen't assigned to multiple N-numbers
'''

ACCIDENTAL_CALLSIGNS = ["N12345", "N123AB"]

BASE_PATH = OUTPUT_DIR / "data/raw/adsblol/parquet_output/v6"

def pia_to_reg_broadcast_df():
    df_faa = get_openairframes_faa_df()
    df_faa = df_faa.rename({"registration_number": "registration"})
    df_pia_adsb = (
        pl.scan_parquet(
            str(BASE_PATH / "year=2026/month=*/day=*/icao_bucket=5/data.parquet"),
            hive_partitioning=True,
            missing_columns='insert'
        )
        .filter(
            pl.col("pia")
        )
        .collect()
    )

    total_pia_icaos = df_pia_adsb.get_column("icao").unique().len()
    print(f"Total pia icaos: {total_pia_icaos}")
    df_pia_n_number_df = df_pia_adsb.filter(pl.col("callsign").str.strip_chars().str.to_uppercase().str.starts_with("N"))
    df_pia_n_number_df = df_pia_n_number_df.unique(["icao", "callsign"])
    faa_n_numbers = df_faa.get_column("registration").unique().to_list()
    df_pia_n_number_df = df_pia_n_number_df.filter(pl.col("callsign").is_in(faa_n_numbers))
    df_pia_n_number_df = df_pia_n_number_df.filter(~pl.col("callsign").is_in(ACCIDENTAL_CALLSIGNS))

    icao_occurrences = (
        df_pia_n_number_df
        .group_by("icao")
        .len()
        .rename({"len": "occurrences"})
        .sort("occurrences", descending=True)
    )
    assert icao_occurrences.get_column("occurrences").max() == 1, "Some pia icaos are assigned to multiple N-numbers"
    print(f"Total pia icaos that broadcast an N-number: {len(df_pia_n_number_df)}")
        
    # check
    df_groundcontrol = load_ground_control_mappings()
    df = df_pia_n_number_df.select(["time", "icao", "callsign"])
    df = df.join(df_groundcontrol, left_on="icao", right_on="pia_icao").with_columns((pl.col("callsign") == pl.col("registration")).alias("reg_match"))
    assert df.get_column("reg_match").sum() == len(df)
    return df_pia_n_number_df

